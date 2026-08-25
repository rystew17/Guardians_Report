"""Everything the page shows, assembled once so the template only formats.

The section is deliberately flat: one row per priced selection, carrying every
number that went into its verdict. A reader who disagrees with a conclusion
should be able to recompute it by hand from the row itself rather than having to
trust the total, which is the same rule the rest of this report follows.

Both sides of every market are shown, not only the side we lean. Printing only
the attractive half would hide that the other half is the same disagreement
with its sign flipped, and would make a 6-point gap look like a discovery rather
than an arithmetic necessity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from guards_report.betting import clv as clv_module, verdict as verdict_module
from guards_report.betting.edge import Play
from guards_report.betting.guide import Belief, Night
from guards_report.odds import types

MARKET_LABELS = {
    types.MONEYLINE: "Moneyline",
    types.TOTAL: "Total runs",
    types.RUNLINE: "Run line",
    types.F5_MONEYLINE: "First five",
    types.F5_TOTAL: "First five total",
    types.STRIKEOUTS: "Strikeouts",
}


def american(value: float) -> str:
    """As a book prints it, with the plus that a bare number loses."""
    return f"{value:+.0f}"


@dataclass(frozen=True)
class Row:
    """One priced selection, with its verdict and everything behind it."""

    market: str
    market_label: str
    selection: str
    line: float | None
    american: str
    p_model: float
    p_market: float
    break_even: float
    disagreement: float
    edge: float
    sigma: float
    z: float
    confidence: float
    stake: float
    basis: str
    verdict: verdict_module.Verdict

    @property
    def is_bet(self) -> bool:
        return self.verdict.is_bet


@dataclass
class Section:
    """The whole block, including what it could not do."""

    disclaimer: str = verdict_module.DISCLAIMER
    rows: list[Row] = field(default_factory=list)
    summary: str = ""
    tau: float = 0.0
    z_threshold: float = 0.0
    devig: str = "shin"
    record: Any | None = None
    warnings: list[str] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    sigma_note: str = ""

    @property
    def bets(self) -> list[Row]:
        return [r for r in self.rows if r.is_bet]

    @property
    def has_prices(self) -> bool:
        return bool(self.rows)


# Why sigma is what it is. Worth stating on the page rather than burying,
# because it is a limit of the evidence rather than of the model: with 11,668
# held-out games, miscalibration below about 1.4 points cannot be told from
# sampling noise, and the standard error is floored there. More seasons would
# lower the floor and let more disagreements through.
SIGMA_NOTE = (
    "Our standard error is currently set by how much history we have, not by "
    "how good the model is. Across {n:,} held-out games, miscalibration below "
    "{floor:.1f} points cannot be separated from sampling noise, so that is "
    "the floor. More seasons would tighten it and let smaller edges through."
)


def build(
    night: Night,
    beliefs: dict[tuple[str, str], Belief],
    *,
    tau: float,
    z_threshold: float,
    devig: str = "shin",
    record: Any | None = None,
    lines: dict[tuple[str, str], float] | None = None,
    calibration: list[dict] | None = None,
) -> Section:
    """Turn a priced night into the rows the template renders."""
    lines = lines or {}
    rows: list[Row] = []

    for play in night.considered:
        key = (play.market, play.selection.lower())
        belief = beliefs.get(key)
        measured = bool(belief and belief.measured)
        basis = belief.basis if belief else ""

        rows.append(Row(
            market=play.market,
            market_label=MARKET_LABELS.get(play.market, play.market.title()),
            selection=play.selection,
            line=play.line if play.line is not None else lines.get(key),
            american=american(play.american),
            p_model=play.p_model,
            p_market=play.p_market,
            break_even=play.break_even,
            disagreement=play.disagreement,
            edge=play.edge,
            sigma=play.sigma,
            z=play.z,
            confidence=play.confidence,
            stake=play.stake if measured else 0.0,
            basis=basis,
            verdict=verdict_module.decide(
                play, measured=measured, z_threshold=z_threshold, basis=basis),
        ))

    # Bets first, then near misses, then everything else -- so the reader meets
    # the decision before the arithmetic. Within a group, by confidence.
    order = {verdict_module.BET: 0, verdict_module.PASS_INSIDE_ERROR: 1,
             verdict_module.PASS_UNMEASURED: 2, verdict_module.PASS_PRICED_IN: 3}
    rows.sort(key=lambda r: (order.get(r.verdict.action, 9), -r.z))

    return Section(
        rows=rows,
        summary=verdict_module.summarize([r.verdict for r in rows]),
        tau=tau,
        z_threshold=z_threshold,
        devig=devig,
        record=record,
        warnings=list(night.warnings),
        unmatched=list(night.unmatched),
        sigma_note=_sigma_note(calibration),
    )


def _sigma_note(calibration: list[dict] | None) -> str:
    if not calibration:
        return ""
    games = sum(int(entry.get("n", 0)) for entry in calibration)
    floors = [float(entry.get("resolution", 0.0)) for entry in calibration]
    if not games or not floors:
        return ""
    return SIGMA_NOTE.format(n=games, floor=(sum(floors) / len(floors)) * 100)
