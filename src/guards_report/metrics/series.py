"""Per-game box scores and per-player lines for the current series.

Two views of the same games. The team pages get a full box score for each
completed game -- line score, decisions, and the players who actually decided
it. Each player box gets his own line across the series, which is the number a
broadcaster reaches for first: how has this man swung the bat *in this set*,
rather than over a rolling thirty days.

Everything here is arithmetic over the source's own box score rows. Nothing is
estimated, and the one ranking in the module states its formula in the report
so a reader can check why a name is on the list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from guards_report.metrics import formulas as f

BATTING_COUNTS = (
    "atBats", "hits", "doubles", "triples", "homeRuns", "rbi", "runs",
    "baseOnBalls", "strikeOuts", "stolenBases", "hitByPitch", "totalBases",
    "plateAppearances",
)
PITCHING_COUNTS = (
    "outs", "earnedRuns", "runs", "hits", "homeRuns", "baseOnBalls",
    "strikeOuts", "battersFaced",
)


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Per-player series lines
# ---------------------------------------------------------------------------


@dataclass
class SeriesLine:
    """One player's aggregate line across the games of a series."""

    games: int = 0
    stat: dict[str, int] = field(default_factory=dict)
    pitching: bool = False

    @property
    def has_data(self) -> bool:
        key = "battersFaced" if self.pitching else "plateAppearances"
        return self.stat.get(key, 0) > 0

    @property
    def avg(self) -> float | None:
        return f.batting_average(self.stat.get("hits", 0), self.stat.get("atBats", 0))

    @property
    def summary(self) -> str:
        """The line as a broadcaster reads it: "2-6, 2B, HR, 3 RBI"."""
        if self.pitching:
            return pitching_summary(self.stat)

        s = self.stat
        parts = [f"{s.get('hits', 0)}-{s.get('atBats', 0)}"]

        # Extra-base hits are named by type, with a count only when repeated.
        for key, label in (("doubles", "2B"), ("triples", "3B"), ("homeRuns", "HR")):
            count = s.get(key, 0)
            if count == 1:
                parts.append(label)
            elif count > 1:
                parts.append(f"{count} {label}")

        for key, label in (
            ("rbi", "RBI"), ("runs", "R"), ("baseOnBalls", "BB"),
            ("strikeOuts", "K"), ("stolenBases", "SB"),
        ):
            count = s.get(key, 0)
            if count:
                parts.append(f"{count} {label}")
        return ", ".join(parts)


def pitching_summary(stat: dict[str, int]) -> str:
    """A pitching line: "6.1 IP, 4 H, 2 R, 2 ER, 1 BB, 7 K"."""
    parts = [f"{f.outs_to_ip(stat.get('outs', 0))} IP"]
    for key, label in (
        ("hits", "H"), ("runs", "R"), ("earnedRuns", "ER"),
        ("baseOnBalls", "BB"), ("strikeOuts", "K"),
    ):
        parts.append(f"{stat.get(key, 0)} {label}")
    return ", ".join(parts)


def _accumulate(
    into: dict[int, SeriesLine],
    pid: int,
    stat: dict[str, Any] | None,
    fields: tuple[str, ...],
    is_pitching: bool,
) -> None:
    if not stat:
        return
    # The source returns an empty stat block for players who dressed but did
    # not appear. Counting those would inflate the games total.
    if not any(_int(stat.get(name)) for name in fields):
        return

    line = into.setdefault(pid, SeriesLine(pitching=is_pitching))
    line.games += 1
    for name in fields:
        line.stat[name] = line.stat.get(name, 0) + _int(stat.get(name))


def player_series_lines(
    boxscores: list[dict[str, Any]],
) -> tuple[dict[int, SeriesLine], dict[int, SeriesLine]]:
    """Aggregate every player's batting and pitching across a set of games.

    Returns (batting, pitching), each keyed by player id. A player who did not
    appear has no entry at all, which the renderer shows as "no series line
    yet" rather than as a line of zeroes.
    """
    batting: dict[int, SeriesLine] = {}
    pitching: dict[int, SeriesLine] = {}

    for box in boxscores:
        for side in ("home", "away"):
            players = ((box.get("teams") or {}).get(side) or {}).get("players") or {}
            for entry in players.values():
                pid = (entry.get("person") or {}).get("id")
                if pid is None:
                    continue
                stats = entry.get("stats") or {}
                _accumulate(batting, pid, stats.get("batting"), BATTING_COUNTS, False)
                _accumulate(pitching, pid, stats.get("pitching"), PITCHING_COUNTS, True)

    return batting, pitching


# ---------------------------------------------------------------------------
# Per-game box scores
# ---------------------------------------------------------------------------


@dataclass
class Performer:
    """A player worth naming in a recap, with the line that earned it."""

    player_id: int
    name: str
    team: str
    summary: str
    score: float


@dataclass
class GameBox:
    """One completed game, summarised the way a box score page shows it."""

    game_pk: int
    game_date: date
    series_game_number: int | None
    away_abbr: str
    home_abbr: str
    away_runs: int
    home_runs: int
    innings: list[dict[str, Any]] = field(default_factory=list)
    away_totals: dict[str, int] = field(default_factory=dict)
    home_totals: dict[str, int] = field(default_factory=dict)
    winning_pitcher: str = ""
    winning_line: str = ""
    losing_pitcher: str = ""
    losing_line: str = ""
    save_pitcher: str = ""
    performers: list[Performer] = field(default_factory=list)

    @property
    def final(self) -> str:
        return f"{self.away_abbr} {self.away_runs}, {self.home_abbr} {self.home_runs}"

    @property
    def winner_abbr(self) -> str:
        return self.home_abbr if self.home_runs > self.away_runs else self.away_abbr


# How standout performances are ranked. Both are plain counting formulas over
# the box score, and the report prints them, because "top performers" is
# otherwise just an opinion presented as data.
BATTER_SCORE = "total bases + RBI + runs + walks + stolen bases"
PITCHER_SCORE = "outs recorded + strikeouts - 2 x earned runs"


def _batter_score(s: dict[str, Any]) -> float:
    return float(
        _int(s.get("totalBases")) + _int(s.get("rbi")) + _int(s.get("runs"))
        + _int(s.get("baseOnBalls")) + _int(s.get("stolenBases"))
    )


def _pitcher_score(s: dict[str, Any]) -> float:
    return float(
        _int(s.get("outs")) + _int(s.get("strikeOuts")) - _int(s.get("earnedRuns")) * 2
    )


def top_performers(box: dict[str, Any], *, limit: int = 3) -> list[Performer]:
    """The players who most shaped the game, across both clubs."""
    found: list[Performer] = []

    for side in ("away", "home"):
        team = (box.get("teams") or {}).get(side) or {}
        abbr = ((team.get("team") or {}).get("abbreviation")) or ""

        for entry in (team.get("players") or {}).values():
            person = entry.get("person") or {}
            pid, name = person.get("id"), person.get("fullName") or ""
            if pid is None:
                continue
            stats = entry.get("stats") or {}

            batting = stats.get("batting") or {}
            if _int(batting.get("plateAppearances")):
                line = SeriesLine(
                    games=1, stat={k: _int(batting.get(k)) for k in BATTING_COUNTS}
                )
                found.append(
                    Performer(pid, name, abbr, line.summary, _batter_score(batting))
                )

            pitching = stats.get("pitching") or {}
            if _int(pitching.get("battersFaced")):
                line = SeriesLine(
                    games=1, pitching=True,
                    stat={k: _int(pitching.get(k)) for k in PITCHING_COUNTS},
                )
                found.append(
                    Performer(pid, name, abbr, line.summary, _pitcher_score(pitching))
                )

    # Ties break on name, so a rebuild produces the same list every time.
    found.sort(key=lambda p: (-p.score, p.name))
    return found[:limit]


def _decision_lines(
    boxscore: dict[str, Any], winner: str, loser: str
) -> tuple[str, str]:
    """The pitching lines belonging to the winning and losing pitchers."""
    wanted: dict[str, str] = {name: "" for name in (winner, loser) if name}

    for side in ("away", "home"):
        players = ((boxscore.get("teams") or {}).get(side) or {}).get("players") or {}
        for entry in players.values():
            name = (entry.get("person") or {}).get("fullName")
            if name not in wanted or wanted[name]:
                continue
            pitching = (entry.get("stats") or {}).get("pitching") or {}
            if not _int(pitching.get("battersFaced")):
                continue
            wanted[name] = pitching_summary(
                {k: _int(pitching.get(k)) for k in PITCHING_COUNTS}
            )

    return wanted.get(winner, ""), wanted.get(loser, "")


def parse_game_box(
    *,
    schedule_game: dict[str, Any],
    boxscore: dict[str, Any],
    linescore: dict[str, Any],
) -> GameBox:
    """Assemble one finished game from its schedule row, box score and line score."""
    teams = schedule_game.get("teams") or {}
    away, home = teams.get("away") or {}, teams.get("home") or {}
    ls_teams = linescore.get("teams") or {}
    decisions = schedule_game.get("decisions") or {}

    game = GameBox(
        game_pk=schedule_game.get("gamePk"),
        game_date=date.fromisoformat(schedule_game["officialDate"]),
        series_game_number=schedule_game.get("seriesGameNumber"),
        away_abbr=(away.get("team") or {}).get("abbreviation", ""),
        home_abbr=(home.get("team") or {}).get("abbreviation", ""),
        away_runs=_int((ls_teams.get("away") or {}).get("runs")),
        home_runs=_int((ls_teams.get("home") or {}).get("runs")),
        innings=[
            {
                "num": inning.get("num"),
                "away": (inning.get("away") or {}).get("runs"),
                "home": (inning.get("home") or {}).get("runs"),
            }
            for inning in linescore.get("innings") or []
        ],
        away_totals={
            k: _int((ls_teams.get("away") or {}).get(k))
            for k in ("runs", "hits", "errors")
        },
        home_totals={
            k: _int((ls_teams.get("home") or {}).get(k))
            for k in ("runs", "hits", "errors")
        },
        winning_pitcher=(decisions.get("winner") or {}).get("fullName", ""),
        losing_pitcher=(decisions.get("loser") or {}).get("fullName", ""),
        save_pitcher=(decisions.get("save") or {}).get("fullName", ""),
        performers=top_performers(boxscore),
    )

    game.winning_line, game.losing_line = _decision_lines(
        boxscore, game.winning_pitcher, game.losing_pitcher
    )
    return game
