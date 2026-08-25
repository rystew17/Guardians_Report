"""Assembling priced markets and our own numbers into a night's plays.

Pure: given markets and a lookup of what we believe, this produces `Play`
records and nothing else. It reads no files, calls no models, and decides
nothing about presentation.

Three rules it enforces, each of which exists because the failure is silent:

**A market we cannot match is reported, never dropped.** A mistyped team code
would otherwise turn into "no plays tonight", which is indistinguishable from a
night with no edge. Unmatched selections come back in `unmatched` so the page
can say what it could not read.

**A belief without a measured standard error cannot be staked.** Only the win
model has calibration data behind it; totals, first five and strikeout props
have none yet. They are still priced and still shown -- the disagreement is
real information -- but a stake computed against a floor sigma would be a
confident number derived from no evidence about how wrong we tend to be.

**The vig comes out before anything is called a play.** Beating the book's
opinion is not enough; the bar is beating the price, which sits about 2.4
points worse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from guards_report.betting import edge as edge_module, prices
from guards_report.betting.edge import Play

# Beyond this gap between our number and the market's, the price is more likely
# wrong than we are. Nothing legitimate on a major market disagrees by thirty
# points with a book that has taken money on it.
IMPLAUSIBLE_DISAGREEMENT = 0.30


@dataclass(frozen=True)
class Belief:
    """What we think about one outcome, and how sure we are allowed to be."""

    probability: float
    sigma: float
    measured: bool
    basis: str = ""


@dataclass
class Night:
    """Everything the page needs, including what could not be priced."""

    plays: list[Play] = field(default_factory=list)
    considered: list[Play] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    unstakeable: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def quiet(self) -> bool:
        """A night that priced markets and recommended nothing.

        Worth naming, because it is the expected outcome most nights and should
        read as the system working rather than as a failure to load.
        """
        return bool(self.considered) and not self.plays


def build(
    markets,
    beliefs: dict[tuple[str, str], Belief],
    *,
    tau: float,
    z_threshold: float,
    devig: str = "shin",
    kelly_fraction: float = edge_module.KELLY_FRACTION,
) -> Night:
    """Price every market against what we believe, and select what clears."""
    night = Night()

    for market in markets:
        if not market.complete:
            night.warnings.append(
                f"{market.name}: incomplete market, cannot remove the vig")
            continue

        decimals = market.decimal()
        if decimals is None:
            night.warnings.append(
                f"{market.name}: a price could not be read as American odds")
            continue

        fair = prices.fair_probabilities(decimals, devig)
        if fair is None:
            night.warnings.append(f"{market.name}: prices could not be de-vigged")
            continue

        overround = prices.overround(decimals)
        if overround is not None and overround <= 0:
            # Either the feed arrived de-vigged or it is not a real market.
            # Both make every price look better than it is.
            night.warnings.append(
                f"{market.name}: no margin in these prices ({overround:+.3f}) "
                f"-- they may already have been de-vigged")

        for quote, p_market in zip(market.quotes, fair):
            # The line is part of the key. Books post alternate numbers on the
            # same bet, and without it the second silently answers for the first.
            key = (market.name, quote.selection.lower(), quote.line)
            belief = beliefs.get(key)
            if belief is None:
                night.unmatched.append(f"{market.name}: {quote.selection}")
                continue

            play = edge_module.assess(
                market=market.name,
                selection=quote.selection,
                american=quote.american,
                line=quote.line,
                p_model=belief.probability,
                sigma=belief.sigma,
                p_market=p_market,
                tau=tau,
                kelly_fraction=kelly_fraction,
            )
            if play is None:
                night.warnings.append(
                    f"{market.name}: {quote.selection} could not be priced")
                continue

            if abs(play.disagreement) > IMPLAUSIBLE_DISAGREEMENT:
                # Past this the likely explanation is a bad price, not an edge.
                # In-play odds reaching a pregame projection is the case that
                # produced it, but a mis-mapped market or a stale line does the
                # same thing, and every one of them looks like free money.
                night.warnings.append(
                    f"{market.name}: {quote.selection} at {quote.american:+.0f} "
                    f"differs from our number by "
                    f"{abs(play.disagreement) * 100:.0f} points, which is far "
                    f"more likely a bad price than an edge -- not staked")
                night.considered.append(play)
                night.unstakeable.append(f"{market.name}: {quote.selection}")
                continue

            night.considered.append(play)

            if not belief.measured:
                # Priced and shown, never staked. The disagreement is real
                # information; a stake against an unmeasured sigma is not.
                night.unstakeable.append(f"{market.name}: {quote.selection}")
                continue
            if play.expected_value > 0 and play.z >= z_threshold:
                night.plays.append(play)

    night.plays.sort(key=lambda p: p.z, reverse=True)
    night.considered.sort(key=lambda p: p.z, reverse=True)
    return night
