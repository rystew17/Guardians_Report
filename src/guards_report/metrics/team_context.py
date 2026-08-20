"""Team-level context: standings, run differential, and league-wide ranks.

This is the layer that answers "who is favoured and why", which the per-player
pages cannot. Two ideas do most of the work here.

Run differential over record. A club four games above .500 with a negative run
differential is a different opponent from one four games above with a positive
one; the gap between actual and Pythagorean record is itself the story, so both
are shown rather than just the record.

Ranks over raw values. "4.31 runs per game" means nothing without knowing that
it is fourth best in baseball. We already fetch all thirty team lines to derive
league averages, so the ranking costs no extra request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from guards_report.config import (
    COMPETITIVE_GAME_TYPES,
    REGULAR_SEASON_GAME_TYPE,
)
from guards_report.metrics import formulas as f


# ---------------------------------------------------------------------------
# Standings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeamRecord:
    team_id: int
    name: str
    wins: int
    losses: int
    win_pct: str
    division_rank: str
    league_rank: str
    games_back: str
    wildcard_games_back: str
    streak: str
    splits: dict[str, tuple[int, int]] = field(default_factory=dict)

    def split(self, key: str) -> str:
        """A record split rendered as W-L, or a dash when absent."""
        record = self.splits.get(key)
        return f"{record[0]}-{record[1]}" if record else "–"

    @property
    def pythagorean(self) -> str:
        return self.split("xWinLoss")

    @property
    def luck(self) -> int | None:
        """Actual wins minus expected wins.

        Positive means the club has won more than its run scoring and
        prevention imply -- often a signal that its record overstates it.
        """
        expected = self.splits.get("xWinLoss")
        if not expected:
            return None
        return self.wins - expected[0]


def parse_standings(payload: dict[str, Any]) -> dict[int, TeamRecord]:
    records: dict[int, TeamRecord] = {}

    for division in payload.get("records", []):
        for entry in division.get("teamRecords", []):
            team = entry.get("team") or {}
            team_id = team.get("id")
            if team_id is None:
                continue

            splits: dict[str, tuple[int, int]] = {}
            for group in (entry.get("records") or {}).values():
                if not isinstance(group, list):
                    continue
                for item in group:
                    key = item.get("type")
                    if key and "wins" in item and "losses" in item:
                        splits[key] = (int(item["wins"]), int(item["losses"]))

            records[team_id] = TeamRecord(
                team_id=team_id,
                name=team.get("name", ""),
                wins=int(entry.get("wins", 0)),
                losses=int(entry.get("losses", 0)),
                win_pct=entry.get("winningPercentage", ""),
                division_rank=entry.get("divisionRank", ""),
                league_rank=entry.get("leagueRank", ""),
                games_back=entry.get("gamesBack", ""),
                wildcard_games_back=entry.get("wildCardGamesBack", ""),
                streak=(entry.get("streak") or {}).get("streakCode", ""),
                splits=splits,
            )

    return records


# ---------------------------------------------------------------------------
# Team statistical ranks
# ---------------------------------------------------------------------------


@dataclass
class RankedStat:
    """One team's value for a metric, with its rank among all thirty clubs."""

    key: str
    label: str
    value: float | None
    rank: int | None
    lower_is_better: bool = False

    @property
    def rank_class(self) -> str:
        """Bucket for colour coding. Rank 1 is always best, by construction."""
        if self.rank is None:
            return "rank-none"
        if self.rank <= 5:
            return "rank-great"
        if self.rank <= 12:
            return "rank-good"
        if self.rank <= 20:
            return "rank-avg"
        if self.rank <= 26:
            return "rank-poor"
        return "rank-bad"


def _team_splits(payload: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = payload.get("stats") or []
    if not blocks:
        return []
    return blocks[0].get("splits") or []


def _int(stat: dict[str, Any], key: str) -> int:
    value = stat.get(key)
    if value in (None, ""):
        return 0
    return int(value)


def _team_id(split: dict[str, Any]) -> int | None:
    return (split.get("team") or {}).get("id")


def hitting_metrics(payload: dict[str, Any]) -> dict[int, dict[str, float | None]]:
    """Per-team offensive rates, computed from each club's counting stats."""
    out: dict[int, dict[str, float | None]] = {}

    for split in _team_splits(payload):
        team_id = _team_id(split)
        if team_id is None:
            continue
        stat = split.get("stat") or {}

        games = _int(stat, "gamesPlayed") or 1
        pa = _int(stat, "plateAppearances")
        ab = _int(stat, "atBats")
        hits = _int(stat, "hits")
        doubles, triples = _int(stat, "doubles"), _int(stat, "triples")
        home_runs = _int(stat, "homeRuns")
        walks, strikeouts = _int(stat, "baseOnBalls"), _int(stat, "strikeOuts")
        hbp, sac_flies = _int(stat, "hitByPitch"), _int(stat, "sacFlies")
        runs = _int(stat, "runs")

        singles = f.singles(hits, doubles, triples, home_runs)
        total_bases = f.total_bases(singles, doubles, triples, home_runs)
        avg = f.batting_average(hits, ab)
        obp = f.on_base_pct(hits, walks, hbp, ab, sac_flies)
        slg = f.slugging(total_bases, ab)

        out[team_id] = {
            "runsPerGame": runs / games,
            "runs": float(runs),
            "avg": avg,
            "obp": obp,
            "slg": slg,
            "ops": f.ops(obp, slg),
            "iso": f.iso(slg, avg),
            "homeRuns": float(home_runs),
            "kPct": f.k_pct(strikeouts, pa),
            "bbPct": f.bb_pct(walks, pa),
            "stolenBases": float(_int(stat, "stolenBases")),
        }

    return out


def pitching_metrics(
    payload: dict[str, Any], *, fip_constant: float
) -> dict[int, dict[str, float | None]]:
    """Per-team run prevention rates."""
    out: dict[int, dict[str, float | None]] = {}

    for split in _team_splits(payload):
        team_id = _team_id(split)
        if team_id is None:
            continue
        stat = split.get("stat") or {}

        games = _int(stat, "gamesPlayed") or 1
        outs = _int(stat, "outs")
        runs = _int(stat, "runs")
        earned = _int(stat, "earnedRuns")
        hits = _int(stat, "hits")
        walks, strikeouts = _int(stat, "baseOnBalls"), _int(stat, "strikeOuts")
        home_runs, hbp = _int(stat, "homeRuns"), _int(stat, "hitByPitch")
        batters_faced = _int(stat, "battersFaced")

        out[team_id] = {
            "runsAllowedPerGame": runs / games,
            "runsAllowed": float(runs),
            "era": f.era(earned, outs),
            "fip": f.fip(
                home_runs=home_runs, walks=walks, hit_by_pitch=hbp,
                strikeouts=strikeouts, outs=outs, fip_constant_=fip_constant,
            ),
            "whip": f.whip(walks, hits, outs),
            "kPct": f.k_pct(strikeouts, batters_faced),
            "bbPct": f.bb_pct(walks, batters_faced),
            "hrPer9": f.per_nine(home_runs, outs),
            "saves": float(_int(stat, "saves")),
            "blownSaves": float(_int(stat, "blownSaves")),
            "holds": float(_int(stat, "holds")),
        }

    return out


def rank_all(
    metrics: dict[int, dict[str, float | None]], *, lower_is_better: set[str]
) -> dict[int, dict[str, int]]:
    """Rank every team on every metric, 1 being best.

    Teams with no value for a metric are left unranked rather than sorted to
    the bottom, so a missing number never masquerades as a poor one.
    """
    ranks: dict[int, dict[str, int]] = {team: {} for team in metrics}
    if not metrics:
        return ranks

    keys = set()
    for values in metrics.values():
        keys.update(values)

    for key in keys:
        scored = [
            (team, values[key])
            for team, values in metrics.items()
            if values.get(key) is not None
        ]
        scored.sort(key=lambda pair: pair[1], reverse=key not in lower_is_better)
        for position, (team, _) in enumerate(scored, start=1):
            ranks[team][key] = position

    return ranks


# Metrics where a lower value is the better outcome.
HITTING_LOWER_BETTER = {"kPct"}
PITCHING_LOWER_BETTER = {
    "runsAllowedPerGame", "runsAllowed", "era", "fip", "whip", "bbPct",
    "hrPer9", "blownSaves",
}


@dataclass(frozen=True)
class SeriesContext:
    """Where today's game sits in the season series and the current set."""

    games_played: int
    home_wins: int
    away_wins: int
    series_game_number: int | None
    games_in_series: int | None
    recent: list[str] = field(default_factory=list)

    def record_for(self, *, home: bool) -> str:
        wins = self.home_wins if home else self.away_wins
        losses = self.away_wins if home else self.home_wins
        return f"{wins}-{losses}"

    @property
    def leader(self) -> str:
        if self.home_wins > self.away_wins:
            return "home"
        if self.away_wins > self.home_wins:
            return "away"
        return "even"


def parse_series(
    payload: dict[str, Any], *, home_team_id: int, today_game_pk: int
) -> SeriesContext:
    """Season series record between the two clubs, excluding today's game.

    Only completed games count. A postponed or in-progress game has no winner
    and would otherwise be silently scored as a loss for somebody.
    """
    home_wins = away_wins = 0
    recent: list[str] = []
    series_game_number = games_in_series = None

    for day in payload.get("dates", []):
        for game in day.get("games", []):
            # Spring training and exhibitions share this endpoint with real
            # games. The source is asked to filter them too, but a record is
            # wrong rather than merely incomplete if one slips through.
            if game.get("gameType") not in COMPETITIVE_GAME_TYPES:
                continue

            teams = game.get("teams") or {}
            if game.get("gamePk") == today_game_pk:
                series_game_number = game.get("seriesGameNumber")
                games_in_series = game.get("gamesInSeries")
                continue

            state = (game.get("status") or {}).get("abstractGameState")
            if state != "Final":
                continue

            home, away = teams.get("home") or {}, teams.get("away") or {}
            home_is_our_home = (home.get("team") or {}).get("id") == home_team_id

            if home.get("isWinner"):
                winner_is_home_team = home_is_our_home
            elif away.get("isWinner"):
                winner_is_home_team = not home_is_our_home
            else:
                continue

            if winner_is_home_team:
                home_wins += 1
            else:
                away_wins += 1

            recent.append(
                f"{game.get('officialDate', '')} "
                f"{away.get('score', '')}-{home.get('score', '')}"
            )

    return SeriesContext(
        games_played=home_wins + away_wins,
        home_wins=home_wins,
        away_wins=away_wins,
        series_game_number=series_game_number,
        games_in_series=games_in_series,
        recent=recent[-5:],
    )


@dataclass(frozen=True)
class SeriesRecord:
    """How a club has fared across whole series, not individual games.

    A team can be under .500 in games and still win most of its series, or the
    reverse. Series outcomes are how a season is actually experienced, and they
    are not derivable from the standings.
    """

    won: int = 0
    lost: int = 0
    split: int = 0
    sweeps_for: int = 0
    sweeps_against: int = 0
    completed: int = 0

    @property
    def record(self) -> str:
        return f"{self.won}-{self.lost}" + (f"-{self.split}" if self.split else "")

    @property
    def win_pct(self) -> float | None:
        decided = self.won + self.lost
        return f.rate(self.won, decided) if decided else None


def parse_series_records(
    payload: dict[str, Any], *, team_id: int
) -> SeriesRecord:
    """Group a season's games into series and score each one.

    Series boundaries come from the source's own `seriesGameNumber`, which
    resets to 1 at the start of each set. Only regular-season games count, and
    the final group is dropped when it is still in progress -- a club leading
    2-0 in a three-game set has not won it yet.

    A sweep requires at least two games, so a one-game make-up set cannot be
    counted as one.
    """
    games: list[tuple[str, int, bool | None]] = []

    for day in payload.get("dates", []):
        for game in day.get("games", []):
            if game.get("gameType") != REGULAR_SEASON_GAME_TYPE:
                continue

            teams = game.get("teams") or {}
            side = next(
                (
                    s for s in ("home", "away")
                    if ((teams.get(s) or {}).get("team") or {}).get("id") == team_id
                ),
                None,
            )
            if side is None:
                continue

            final = (game.get("status") or {}).get("abstractGameState") == "Final"
            won: bool | None = None
            if final:
                other = "away" if side == "home" else "home"
                if (teams[side] or {}).get("isWinner"):
                    won = True
                elif (teams.get(other) or {}).get("isWinner"):
                    won = False
                else:
                    continue

            games.append((
                game.get("officialDate", ""),
                int(game.get("seriesGameNumber") or 1),
                won,
            ))

    games.sort(key=lambda g: (g[0], g[1]))

    # Split into series wherever the source's game number restarts.
    series: list[list[bool | None]] = []
    for _, number, won in games:
        if number == 1 or not series:
            series.append([])
        series[-1].append(won)

    record = SeriesRecord()
    counted = []
    for group in series:
        # Skip any set with a game still unplayed -- including today's.
        if any(result is None for result in group):
            continue
        counted.append(group)

    won = lost = split = sweeps_for = sweeps_against = 0
    for group in counted:
        wins = sum(1 for result in group if result)
        losses = len(group) - wins
        if wins > losses:
            won += 1
            if losses == 0 and len(group) >= 2:
                sweeps_for += 1
        elif losses > wins:
            lost += 1
            if wins == 0 and len(group) >= 2:
                sweeps_against += 1
        else:
            split += 1

    return SeriesRecord(
        won=won, lost=lost, split=split,
        sweeps_for=sweeps_for, sweeps_against=sweeps_against,
        completed=len(counted),
    )


@dataclass
class TeamProfile:
    """Everything the comparison page needs about one club."""

    team_id: int
    name: str
    abbreviation: str
    record: TeamRecord | None
    hitting: list[RankedStat] = field(default_factory=list)
    pitching: list[RankedStat] = field(default_factory=list)
    run_differential: int | None = None
    bullpen_pitches_last_3: int = 0
    bullpen_available: int = 0
    bullpen_total: int = 0
    lineup_hand_counts: dict[str, int] = field(default_factory=dict)
    series_record: SeriesRecord | None = None


HITTING_DISPLAY = [
    ("runsPerGame", "Runs/G"),
    ("ops", "OPS"),
    ("avg", "AVG"),
    ("obp", "OBP"),
    ("slg", "SLG"),
    ("iso", "ISO"),
    ("homeRuns", "HR"),
    ("kPct", "K%"),
    ("bbPct", "BB%"),
    ("stolenBases", "SB"),
]

PITCHING_DISPLAY = [
    ("runsAllowedPerGame", "Runs allowed/G"),
    ("era", "ERA"),
    ("fip", "FIP"),
    ("whip", "WHIP"),
    ("kPct", "K%"),
    ("bbPct", "BB%"),
    ("hrPer9", "HR/9"),
    ("saves", "Saves"),
    ("blownSaves", "Blown saves"),
]


def build_ranked(
    team_id: int,
    metrics: dict[int, dict[str, float | None]],
    ranks: dict[int, dict[str, int]],
    display: list[tuple[str, str]],
    lower_is_better: set[str],
) -> list[RankedStat]:
    values = metrics.get(team_id, {})
    team_ranks = ranks.get(team_id, {})
    return [
        RankedStat(
            key=key,
            label=label,
            value=values.get(key),
            rank=team_ranks.get(key),
            lower_is_better=key in lower_is_better,
        )
        for key, label in display
    ]
