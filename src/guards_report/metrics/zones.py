"""Strike-zone heat maps from MLB's hotColdZones data.

MLB reports zone performance on its standard 13-cell layout:

    zones 01 02 03      the 3x3 strike zone, read left to right,
          04 05 06      top to bottom, from the catcher's view
          07 08 09

    zones 11 12         the four regions outside the zone:
          13 14         up-and-in, up-and-away, down-and-in, down-and-away

We render the 3x3 as a grid with the four outside regions as corner gutters,
which is how Savant and MLB present it and therefore what a reader already
knows how to interpret.

The values come from the source already computed. This module only reshapes
them and attaches a colour scale; it does no averaging of its own, because a
zone value is a rate over a denominator we are not given and therefore cannot
be recombined correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# The nine in-zone cells in display order, and the four outside regions.
IN_ZONE_CODES = ("01", "02", "03", "04", "05", "06", "07", "08", "09")
OUT_ZONE_CODES = ("11", "12", "13", "14")

METRIC_LABELS = {
    "battingAverage": "AVG",
    "onBasePlusSlugging": "OPS",
    "sluggingPercentage": "SLG",
    "onBasePercentage": "OBP",
    "exitVelocity": "Exit velo",
    "earnedRunAverage": "ERA",
    "numberOfPitches": "Pitches",
    "numberOfStrikes": "Strikes",
}


@dataclass
class ZoneCell:
    code: str
    raw: str
    value: float | None
    temp: str
    # MLB ships a colour with every zone, already on its own hot/cold scale.
    # Using it rather than a scale of our own keeps the map identical to what
    # the same player looks like on MLB and Savant, and keeps the shading a
    # sourced value rather than an invented one.
    color: str = ""


@dataclass
class ZoneGrid:
    """One metric's 13-cell grid for one player."""

    metric: str
    label: str
    cells: dict[str, ZoneCell] = field(default_factory=dict)

    @property
    def in_zone(self) -> list[ZoneCell]:
        return [self.cells[c] for c in IN_ZONE_CODES if c in self.cells]

    @property
    def out_zone(self) -> list[ZoneCell]:
        return [self.cells[c] for c in OUT_ZONE_CODES if c in self.cells]

    @property
    def has_data(self) -> bool:
        return any(cell.value is not None for cell in self.cells.values())

    def scale(self) -> tuple[float, float]:
        """Min and max across in-zone cells, for relative shading.

        Shading is scaled to the player's own range rather than a fixed league
        scale: the question a heat map answers is "where is this hitter strong
        relative to himself", and a fixed scale washes that out for anyone who
        is uniformly good or uniformly bad.
        """
        values = [c.value for c in self.in_zone if c.value is not None]
        if not values:
            return (0.0, 1.0)
        low, high = min(values), max(values)
        if high - low < 1e-9:
            return (low, low + 1.0)
        return (low, high)

    def intensity(self, cell: ZoneCell) -> float:
        """Position of a cell within the player's own range, 0.0 to 1.0."""
        if cell.value is None:
            return 0.0
        low, high = self.scale()
        return max(0.0, min(1.0, (cell.value - low) / (high - low)))


def _parse_value(raw: str | None) -> float | None:
    """Zone values arrive as strings like '.356' or '89.2', or as placeholders.

    MLB uses '-.--' and '.---' for "no data in this zone", which must stay
    distinct from a genuine zero.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or set(text) <= {".", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_zones(person: dict[str, Any]) -> dict[str, ZoneGrid]:
    """Extract every zone metric for one player from a hydrated people payload.

    Returns a mapping of metric name to grid. A player with too little playing
    time simply has no zone block, which is a legitimate answer.
    """
    grids: dict[str, ZoneGrid] = {}

    for block in person.get("stats") or []:
        if (block.get("type") or {}).get("displayName") != "hotColdZones":
            continue
        for split in block.get("splits") or []:
            stat = split.get("stat") or {}
            metric = stat.get("name")
            zones = stat.get("zones") or []
            if not metric or not zones:
                continue

            grid = ZoneGrid(metric=metric, label=METRIC_LABELS.get(metric, metric))
            for zone in zones:
                code = zone.get("zone")
                if not code:
                    continue
                grid.cells[code] = ZoneCell(
                    code=code,
                    raw=zone.get("value", ""),
                    value=_parse_value(zone.get("value")),
                    temp=zone.get("temp", ""),
                    color=zone.get("color", ""),
                )
            if grid.cells:
                grids[metric] = grid

    return grids
