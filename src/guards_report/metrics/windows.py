"""Recent-form windows: aggregate game logs into season / L5 / L15 / L30 lines.

The one rule that matters here: sum the counting stats over the window first,
then compute rates from those sums. Averaging per-game rates is a different and
wrong number -- a 1-for-1 game and an 0-for-5 game average to .500 that way,
when the player actually went 1 for 6 (.167). Every rate in the report comes
out of `aggregate`, so that mistake has one place to not happen.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from guards_report.metrics import formulas as f


@dataclass(frozen=True)
class WindowSpec:
    """A named recent-form window.

    Exactly one of `games` or `days` is set. Game-count windows are the natural
    unit for hitters, who play nearly every day. Day-count windows are the
    honest unit for starting pitchers: a starter's "last 30 games" would span
    two seasons, which is not what anyone means by recent form.
    """

    label: str
    games: int | None = None
    days: int | None = None

    def __post_init__(self) -> None:
        if (self.games is None) == (self.days is None):
            raise ValueError(
                f"window {self.label!r} must specify exactly one of games or days"
            )


SEASON = WindowSpec(label="Season", days=400)
HITTER_WINDOWS: tuple[WindowSpec, ...] = (
    WindowSpec(label="L5", games=5),
    WindowSpec(label="L15", games=15),
    WindowSpec(label="L30", games=30),
)
# A starter makes roughly one appearance every five days, so these cover
# about the last 3, 6, and 12 starts respectively.
STARTER_WINDOWS: tuple[WindowSpec, ...] = (
    WindowSpec(label="Last 15d", days=15),
    WindowSpec(label="Last 30d", days=30),
    WindowSpec(label="Last 60d", days=60),
)
RELIEVER_WINDOWS: tuple[WindowSpec, ...] = (
    WindowSpec(label="L5 app", games=5),
    WindowSpec(label="L15 app", games=15),
    WindowSpec(label="Last 30d", days=30),
)


# Counting stats summed across a window. Anything not listed here is either a
# rate (recomputed) or not additive (e.g. a season-to-date average).
HITTING_COUNTS = (
    "gamesPlayed", "plateAppearances", "atBats", "hits", "doubles", "triples",
    "homeRuns", "runs", "rbi", "baseOnBalls", "intentionalWalks", "strikeOuts",
    "hitByPitch", "sacFlies", "sacBunts", "stolenBases", "caughtStealing",
    "groundOuts", "airOuts", "totalBases", "leftOnBase", "numberOfPitches",
)
PITCHING_COUNTS = (
    "gamesPlayed", "gamesStarted", "battersFaced", "outs", "hits", "runs",
    "earnedRuns", "homeRuns", "baseOnBalls", "intentionalWalks", "strikeOuts",
    "hitByPitch", "wins", "losses", "saves", "holds", "blownSaves",
    "groundOuts", "airOuts", "numberOfPitches", "pitchesThrown",
)


@dataclass(frozen=True)
class GameLogRow:
    """One player's line from one game."""

    game_date: date
    game_pk: int
    is_home: bool
    opponent_team_id: int | None
    stat: dict[str, Any]


def _to_int(value: Any) -> int:
    if value in (None, "", "-"):
        return 0
    if isinstance(value, str):
        # A few fields arrive as strings; ".---" style placeholders mean zero.
        try:
            return int(float(value))
        except ValueError:
            return 0
    return int(value)


def parse_game_logs(payload: dict[str, Any]) -> list[GameLogRow]:
    """Turn a statsapi gameLog response into rows, newest first.

    An empty splits array is a legitimate answer, not an error -- it is what
    the API returns for a pitcher asked for hitting stats, or for a player who
    has not yet appeared.
    """
    blocks = payload.get("stats") or []
    if not blocks:
        return []

    rows: list[GameLogRow] = []
    for split in blocks[0].get("splits") or []:
        raw_date = split.get("date")
        if not raw_date:
            continue
        game = split.get("game") or {}
        opponent = split.get("opponent") or {}
        rows.append(
            GameLogRow(
                game_date=date.fromisoformat(raw_date),
                game_pk=_to_int(game.get("gamePk")),
                is_home=bool(split.get("isHome", False)),
                opponent_team_id=opponent.get("id"),
                stat=split.get("stat") or {},
            )
        )

    rows.sort(key=lambda r: (r.game_date, r.game_pk), reverse=True)
    return rows


def select(
    rows: Sequence[GameLogRow], spec: WindowSpec, *, as_of: date
) -> list[GameLogRow]:
    """The rows falling inside a window, counting back from `as_of`.

    `as_of` is the report date and rows on or after it are excluded: a preview
    written the morning of a game must not include that game, and must not
    include anything the report's reader could not have known yet.
    """
    eligible = [r for r in rows if r.game_date < as_of]
    if spec.games is not None:
        return eligible[: spec.games]
    cutoff = as_of - timedelta(days=spec.days or 0)
    return [r for r in eligible if r.game_date >= cutoff]


def _sum_counts(rows: Iterable[GameLogRow], fields: Sequence[str]) -> dict[str, int]:
    totals = {name: 0 for name in fields}
    for row in rows:
        for name in fields:
            if name in row.stat:
                totals[name] += _to_int(row.stat[name])
    return totals


def aggregate_hitting(rows: Sequence[GameLogRow]) -> dict[str, Any]:
    """Sum a hitter's counting stats, then derive rates from the sums."""
    totals = _sum_counts(rows, HITTING_COUNTS)

    hits = totals["hits"]
    at_bats = totals["atBats"]
    doubles, triples, home_runs = (
        totals["doubles"], totals["triples"], totals["homeRuns"],
    )
    walks, hbp, sac_flies = (
        totals["baseOnBalls"], totals["hitByPitch"], totals["sacFlies"],
    )
    strikeouts = totals["strikeOuts"]
    plate_appearances = totals["plateAppearances"]

    singles_ = f.singles(hits, doubles, triples, home_runs)
    total_bases = f.total_bases(singles_, doubles, triples, home_runs)

    avg = f.batting_average(hits, at_bats)
    obp = f.on_base_pct(hits, walks, hbp, at_bats, sac_flies)
    slg = f.slugging(total_bases, at_bats)

    return {
        **totals,
        "games": len(rows),
        "singles": singles_,
        "computedTotalBases": total_bases,
        "avg": avg,
        "obp": obp,
        "slg": slg,
        "ops": f.ops(obp, slg),
        "iso": f.iso(slg, avg),
        "babip": f.babip(hits, home_runs, at_bats, strikeouts, sac_flies),
        "kPct": f.k_pct(strikeouts, plate_appearances),
        "bbPct": f.bb_pct(walks, plate_appearances),
    }


def aggregate_pitching(
    rows: Sequence[GameLogRow], *, fip_constant: float | None = None
) -> dict[str, Any]:
    """Sum a pitcher's counting stats, then derive rates from the sums.

    `outs` is summed directly rather than parsed back out of the innings-pitched
    string, which sidesteps the "5.1 innings" notation entirely.
    """
    totals = _sum_counts(rows, PITCHING_COUNTS)

    outs = totals["outs"]
    walks, hbp = totals["baseOnBalls"], totals["hitByPitch"]
    strikeouts, home_runs = totals["strikeOuts"], totals["homeRuns"]
    hits, earned_runs = totals["hits"], totals["earnedRuns"]
    batters_faced = totals["battersFaced"]

    result: dict[str, Any] = {
        **totals,
        "games": len(rows),
        "inningsPitchedDisplay": f.outs_to_ip(outs),
        "era": f.era(earned_runs, outs),
        "whip": f.whip(walks, hits, outs),
        "kPer9": f.per_nine(strikeouts, outs),
        "bbPer9": f.per_nine(walks, outs),
        "hrPer9": f.per_nine(home_runs, outs),
        "kPct": f.k_pct(strikeouts, batters_faced),
        "bbPct": f.bb_pct(walks, batters_faced),
        "kMinusBbPct": f.k_minus_bb_pct(strikeouts, walks, batters_faced),
    }

    # FIP is only meaningful with the season's constant. Omitting the key
    # entirely, rather than storing None, keeps "we could not compute this"
    # distinguishable from "this computed to nothing".
    if fip_constant is not None:
        result["fip"] = f.fip(
            home_runs=home_runs,
            walks=walks,
            hit_by_pitch=hbp,
            strikeouts=strikeouts,
            outs=outs,
            fip_constant_=fip_constant,
        )

    return result


# ---------------------------------------------------------------------------
# Bullpen availability
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BullpenAvailability:
    """Workload signals for one reliever, as of a report date.

    These are facts derived from the game log, not a judgement about whether a
    manager will actually use the pitcher. The report presents the workload and
    leaves the inference to the reader.
    """

    days_rest: int | None
    pitches_last_1: int
    pitches_last_2: int
    pitches_last_3: int
    appearances_last_7: int
    back_to_back: bool
    last_appearance: date | None


def bullpen_availability(
    rows: Sequence[GameLogRow], *, as_of: date
) -> BullpenAvailability:
    prior = [r for r in rows if r.game_date < as_of]
    if not prior:
        return BullpenAvailability(
            days_rest=None,
            pitches_last_1=0,
            pitches_last_2=0,
            pitches_last_3=0,
            appearances_last_7=0,
            back_to_back=False,
            last_appearance=None,
        )

    def pitches_within(days: int) -> int:
        cutoff = as_of - timedelta(days=days)
        return sum(
            _to_int(r.stat.get("numberOfPitches") or r.stat.get("pitchesThrown"))
            for r in prior
            if r.game_date >= cutoff
        )

    last = prior[0].game_date
    appearance_dates = {r.game_date for r in prior}

    return BullpenAvailability(
        days_rest=(as_of - last).days - 1,
        pitches_last_1=pitches_within(1),
        pitches_last_2=pitches_within(2),
        pitches_last_3=pitches_within(3),
        appearances_last_7=sum(
            1 for r in prior if r.game_date >= as_of - timedelta(days=7)
        ),
        # Pitched on two consecutive calendar days immediately before today.
        back_to_back=(
            as_of - timedelta(days=1) in appearance_dates
            and as_of - timedelta(days=2) in appearance_dates
        ),
        last_appearance=last,
    )
