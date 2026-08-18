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
