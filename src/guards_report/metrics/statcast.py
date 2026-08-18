"""Aggregate pitch-level Statcast into zone grids and spray charts.

This is the one place the project uses pitch-level data. It is fetched per
player for the ~50 people in today's game, never league-wide, and never stored
in the warehouse -- the aggregates below are what the report displays.

Everything here is a hard-coded aggregation over raw event rows. The zone
values are counted from `zone`, `description` and `events` columns exactly as
Statcast records them, which is what makes a figure like "OPS by zone against
left-handers" verifiable: it can be recomputed from the archived CSV.
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from guards_report.metrics import formulas as f

# Statcast's `description` values that mean the batter swung. Anything else --
# called strikes, balls, blocked pitches -- is a take.
SWING_DESCRIPTIONS = {
    "hit_into_play", "foul", "swinging_strike", "swinging_strike_blocked",
    "foul_tip", "foul_bunt", "missed_bunt", "bunt_foul_tip",
}
WHIFF_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "missed_bunt", "foul_tip",
}

# Events that end a plate appearance and count as an at-bat.
HIT_EVENTS = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
NON_AB_EVENTS = {
    "walk", "intent_walk", "hit_by_pitch", "sac_fly", "sac_bunt",
    "catcher_interf", "sac_fly_double_play", "sac_bunt_double_play",
}

# The nine in-zone cells plus the four out-of-zone regions, matching the layout
# MLB uses in hotColdZones so the two sets of maps read identically.
IN_ZONE = ("1", "2", "3", "4", "5", "6", "7", "8", "9")
OUT_ZONE = ("11", "12", "13", "14")
ALL_ZONES = IN_ZONE + OUT_ZONE


def parse_pitches(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def _f(row: dict[str, str], key: str) -> float | None:
    raw = (row.get(key) or "").strip()
    if not raw or raw.lower() in ("null", "nan"):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Zone aggregation
# ---------------------------------------------------------------------------


@dataclass
class ZoneStats:
    """Counting stats accumulated for one zone."""

    pitches: int = 0
    swings: int = 0
    whiffs: int = 0
    strikeouts: int = 0
    plate_appearances: int = 0
    at_bats: int = 0
    hits: int = 0
    total_bases: int = 0
    walks: int = 0
    hit_by_pitch: int = 0
    sac_flies: int = 0
    xwoba_sum: float = 0.0
    xwoba_count: int = 0

    @property
    def avg(self) -> float | None:
        return f.batting_average(self.hits, self.at_bats)

    @property
    def obp(self) -> float | None:
        return f.on_base_pct(
            self.hits, self.walks, self.hit_by_pitch, self.at_bats, self.sac_flies
        )

    @property
    def slg(self) -> float | None:
        return f.slugging(self.total_bases, self.at_bats)

    @property
    def ops(self) -> float | None:
        return f.ops(self.obp, self.slg)

    @property
    def xwoba(self) -> float | None:
        if not self.xwoba_count:
            return None
        return self.xwoba_sum / self.xwoba_count

    @property
    def swing_pct(self) -> float | None:
        return f.rate(self.swings, self.pitches)

    @property
    def whiff_pct(self) -> float | None:
        return f.rate(self.whiffs, self.swings)

    @property
    def k_pct(self) -> float | None:
        return f.rate(self.strikeouts, self.plate_appearances)

    def value(self, metric: str) -> float | None:
        return {
            "ops": self.ops, "avg": self.avg, "xwoba": self.xwoba,
            "swing": self.swing_pct, "k": self.k_pct, "whiff": self.whiff_pct,
        }.get(metric)


@dataclass
class ZoneChart:
    """One metric's 13-cell grid, for one handedness split."""

    metric: str
    label: str
    hand: str
    cells: dict[str, ZoneStats] = field(default_factory=dict)
    pitches: int = 0

    @property
    def has_data(self) -> bool:
        return self.pitches >= 50

    def value(self, code: str) -> float | None:
        stats = self.cells.get(code)
        return stats.value(self.metric) if stats else None

    def sample(self, code: str) -> int:
        """Denominator behind a cell, so thin cells can be de-emphasised."""
        stats = self.cells.get(code)
        if not stats:
            return 0
        if self.metric in ("swing", "whiff"):
            return stats.pitches
        return stats.plate_appearances

    def scale(self) -> tuple[float, float]:
        values = [
            v for code in IN_ZONE
            if (v := self.value(code)) is not None and self.sample(code) >= 5
        ]
        if not values:
            return (0.0, 1.0)
        low, high = min(values), max(values)
        if high - low < 1e-9:
            return (low, low + 1.0)
        return (low, high)

    def intensity(self, code: str) -> float:
        value = self.value(code)
        if value is None:
            return 0.5
        low, high = self.scale()
        return max(0.0, min(1.0, (value - low) / (high - low)))


METRIC_LABELS = {
    "ops": "OPS", "avg": "AVG", "xwoba": "xwOBA",
    "swing": "Swing%", "k": "K%", "whiff": "Whiff%",
}
DEFAULT_METRICS = ("ops", "avg", "xwoba", "swing", "k")


def build_zone_charts(
    rows: list[dict[str, str]],
    *,
    perspective: str,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> dict[str, dict[str, ZoneChart]]:
    """Aggregate pitches into zone charts, split by opposing handedness.

    `perspective` is "batter" or "pitcher". For a batter the split is the
    pitcher's throwing hand; for a pitcher it is the batter's side. Returns
    {hand: {metric: chart}}, hand being "L", "R" or "all".
    """
    hand_field = "p_throws" if perspective == "batter" else "stand"

    buckets: dict[tuple[str, str], ZoneStats] = defaultdict(ZoneStats)
    pitch_counts: dict[str, int] = defaultdict(int)

    for row in rows:
        zone = (row.get("zone") or "").strip()
        if zone not in ALL_ZONES:
            continue
        hand = (row.get(hand_field) or "").strip()
        if hand not in ("L", "R"):
            continue

        description = (row.get("description") or "").strip()
        event = (row.get("events") or "").strip()

        for key in (hand, "all"):
            stats = buckets[(key, zone)]
            stats.pitches += 1
            pitch_counts[key] += 1

            if description in SWING_DESCRIPTIONS:
                stats.swings += 1
            if description in WHIFF_DESCRIPTIONS:
                stats.whiffs += 1

            # `events` is populated only on the final pitch of a plate
            # appearance, so counting it here attributes the outcome to the
            # zone of the pitch that ended it -- which is the convention
            # Savant's own zone charts use.
            if not event:
                continue

            stats.plate_appearances += 1
            if event in HIT_EVENTS:
                stats.at_bats += 1
                stats.hits += 1
                stats.total_bases += HIT_EVENTS[event]
            elif event in ("walk", "intent_walk"):
                stats.walks += 1
            elif event == "hit_by_pitch":
                stats.hit_by_pitch += 1
            elif event in ("sac_fly", "sac_fly_double_play"):
                stats.sac_flies += 1
            elif event not in NON_AB_EVENTS:
                stats.at_bats += 1
                if event.startswith("strikeout"):
                    stats.strikeouts += 1

            xwoba = _f(row, "estimated_woba_using_speedangle")
            if xwoba is not None:
                stats.xwoba_sum += xwoba
                stats.xwoba_count += 1

    charts: dict[str, dict[str, ZoneChart]] = {}
    for hand in ("L", "R", "all"):
        by_metric: dict[str, ZoneChart] = {}
        for metric in metrics:
            chart = ZoneChart(
                metric=metric, label=METRIC_LABELS.get(metric, metric),
                hand=hand, pitches=pitch_counts.get(hand, 0),
            )
            for code in ALL_ZONES:
                if (hand, code) in buckets:
                    chart.cells[code] = buckets[(hand, code)]
            by_metric[metric] = chart
        charts[hand] = by_metric

    return charts


# ---------------------------------------------------------------------------
# Spray chart
# ---------------------------------------------------------------------------

# Statcast records hit coordinates in a fixed pixel space where home plate sits
# at roughly (125.42, 198.27) and y increases toward the outfield wall. These
# constants are the standard conversion used to put a batted ball on a field.
HOME_X = 125.42
HOME_Y = 198.27


@dataclass
class SprayPoint:
    x: float          # feet, negative toward left field
    y: float          # feet, toward centre field
    outcome: str      # single | double | triple | home_run | out
    bb_type: str
    exit_velocity: float | None
    distance: float


@dataclass
class SprayChart:
    points: list[SprayPoint] = field(default_factory=list)
    pull_pct: float | None = None
    straight_pct: float | None = None
    oppo_pct: float | None = None

    @property
    def has_data(self) -> bool:
        return len(self.points) >= 10


OUTCOME_ORDER = ("out", "single", "double", "triple", "home_run")


def build_spray(rows: list[dict[str, str]], *, bats: str | None) -> SprayChart:
    """Turn batted-ball coordinates into plottable field positions.

    Coordinates are converted to feet from home plate, with x negative toward
    left field. For a left-handed hitter the chart is mirrored so that "pull"
    is always the same side of the image, which is what makes two hitters'
    charts comparable at a glance.
    """
    chart = SprayChart()
    pull = straight = oppo = 0

    for row in rows:
        hc_x, hc_y = _f(row, "hc_x"), _f(row, "hc_y")
        if hc_x is None or hc_y is None:
            continue
        bb_type = (row.get("bb_type") or "").strip()
        if not bb_type:
            continue

        # Statcast pixels are about 2.5 feet each.
        x = (hc_x - HOME_X) * 2.5
        y = (HOME_Y - hc_y) * 2.5
        if y <= 0:
            continue

        # Mirror left-handed hitters so pull is always to the same side.
        if bats == "L":
            x = -x

        event = (row.get("events") or "").strip()
        outcome = event if event in HIT_EVENTS else "out"

        chart.points.append(SprayPoint(
            x=x, y=y, outcome=outcome, bb_type=bb_type,
            exit_velocity=_f(row, "launch_speed"),
            distance=(x * x + y * y) ** 0.5,
        ))

        # Thirds of the field by horizontal angle. After the mirror above,
        # negative x is always the pull side regardless of which way the
        # hitter bats.
        if x < -y * 0.36:
            pull += 1
        elif x > y * 0.36:
            oppo += 1
        else:
            straight += 1

    total = pull + straight + oppo
    if total:
        chart.pull_pct = 100.0 * pull / total
        chart.straight_pct = 100.0 * straight / total
        chart.oppo_pct = 100.0 * oppo / total

    return chart
