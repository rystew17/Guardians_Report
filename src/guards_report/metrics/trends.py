"""Trend series and sample-size reliability.

Two presentation problems that are really data problems.

Recent form as three static rows (L5, L15, L30) makes the reader difference
them in their head to see whether a player is climbing or falling -- which is
the entire reason to show them. A rolling series answers it at a glance and
costs no extra requests, since the game logs are already fetched.

Sample size is the more serious one. A .203 OPS in 39 plate appearances renders
with the same visual weight as a 500-PA line unless something is done about it,
which invites confident conclusions from noise. We attach a reliability band to
every rate so the renderer can de-emphasize thin samples rather than hide them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from guards_report.metrics import formulas as f
from guards_report.metrics.windows import GameLogRow, _to_int


# ---------------------------------------------------------------------------
# Sample size
# ---------------------------------------------------------------------------
# Thresholds are the widely used points at which a rate stabilizes -- the
# sample where roughly half the observed variation is signal. They are rules of
# thumb, not precise constants, so they are used only to grade confidence for
# display and never to alter a number.

STABILIZATION_PA = {
    "kPct": 60,
    "bbPct": 120,
    "iso": 160,
    "obp": 460,
    "slg": 320,
    "avg": 910,
    "ops": 400,
    "babip": 800,
}

STABILIZATION_BF = {
    "kPct": 70,
    "bbPct": 170,
    "era": 400,
    "fip": 400,
    "whip": 400,
    "hrPer9": 1320,
}


@dataclass(frozen=True)
class Reliability:
    """How much weight a rate deserves, given its denominator."""

    denominator: int
    threshold: int
    band: str  # "solid" | "thin" | "noise"

    @property
    def ratio(self) -> float:
        return self.denominator / self.threshold if self.threshold else 0.0


def reliability(
    stat_key: str, denominator: int, *, pitching: bool = False
) -> Reliability | None:
    table = STABILIZATION_BF if pitching else STABILIZATION_PA
    threshold = table.get(stat_key)
    if threshold is None:
        return None

    ratio = denominator / threshold if threshold else 0.0
    if ratio >= 1.0:
        band = "solid"
    elif ratio >= 0.35:
        band = "thin"
    else:
        band = "noise"
    return Reliability(denominator=denominator, threshold=threshold, band=band)


# ---------------------------------------------------------------------------
# Rolling trend series
# ---------------------------------------------------------------------------


@dataclass
class TrendSeries:
    """A rolling value per game, oldest to newest, for a sparkline."""

    label: str
    points: list[float]
    window: int

    @property
    def has_data(self) -> bool:
        return len(self.points) >= 3

    @property
    def direction(self) -> str:
        """Compare the last third of the series with the first third.

        A blunt instrument deliberately: this drives an arrow glyph, not a
        claim, and a subtler estimator would imply precision the sample does
        not support.
        """
        if len(self.points) < 6:
            return "flat"
        third = max(2, len(self.points) // 3)
        early = sum(self.points[:third]) / third
        late = sum(self.points[-third:]) / third
        if abs(late - early) < 0.02:
            return "flat"
        return "up" if late > early else "down"


def _rolling(
    rows: Sequence[GameLogRow], window: int, compute
) -> list[float]:
    """Apply `compute` to each trailing window of games, oldest first."""
    ordered = list(reversed(rows))  # rows arrive newest-first
    points: list[float] = []
    for end in range(window, len(ordered) + 1):
        value = compute(ordered[end - window : end])
        if value is not None:
            points.append(value)
    return points


def hitter_ops_trend(
    rows: Sequence[GameLogRow], *, as_of: date, window: int = 10, cap: int = 40
) -> TrendSeries:
    """Rolling OPS over a trailing window of games.

    OPS rather than a single component because it moves for both the on-base
    and power reasons a reader cares about, and it is the number they already
    have a feel for.
    """
    prior = [r for r in rows if r.game_date < as_of][:cap]

    def compute(chunk: Sequence[GameLogRow]) -> float | None:
        ab = sum(_to_int(r.stat.get("atBats")) for r in chunk)
        hits = sum(_to_int(r.stat.get("hits")) for r in chunk)
        doubles = sum(_to_int(r.stat.get("doubles")) for r in chunk)
        triples = sum(_to_int(r.stat.get("triples")) for r in chunk)
        home_runs = sum(_to_int(r.stat.get("homeRuns")) for r in chunk)
        walks = sum(_to_int(r.stat.get("baseOnBalls")) for r in chunk)
        hbp = sum(_to_int(r.stat.get("hitByPitch")) for r in chunk)
        sac_flies = sum(_to_int(r.stat.get("sacFlies")) for r in chunk)

        singles = f.singles(hits, doubles, triples, home_runs)
        total_bases = f.total_bases(singles, doubles, triples, home_runs)
        obp = f.on_base_pct(hits, walks, hbp, ab, sac_flies)
        slg = f.slugging(total_bases, ab)
        return f.ops(obp, slg)

    return TrendSeries(
        label=f"Rolling {window}-game OPS",
        points=_rolling(prior, window, compute),
        window=window,
    )


def pitcher_era_trend(
    rows: Sequence[GameLogRow], *, as_of: date, window: int = 5, cap: int = 25
) -> TrendSeries:
    """Rolling ERA over a trailing window of appearances.

    Inverted for display so that, as with the hitter series, up means better.
    Without inverting, an improving pitcher's sparkline would fall while an
    improving hitter's rises, which reads as the opposite of what it means.
    """
    prior = [r for r in rows if r.game_date < as_of][:cap]

    def compute(chunk: Sequence[GameLogRow]) -> float | None:
        outs = sum(_to_int(r.stat.get("outs")) for r in chunk)
        earned = sum(_to_int(r.stat.get("earnedRuns")) for r in chunk)
        era = f.era(earned, outs)
        if era is None:
            return None
        # Clamp before inverting so one blow-up start cannot flatten the rest
        # of the series into a single pixel.
        return -min(era, 12.0)

    return TrendSeries(
        label=f"Rolling {window}-appearance ERA",
        points=_rolling(prior, window, compute),
        window=window,
    )
