"""Turning a projected probability and a posted price into a decision.

The arithmetic is short. The reason it is arranged this way is not, so it is
worth setting down.

Our model's probability and the market's disagree on most games. That
disagreement is not an edge -- an edge is a disagreement where we are the ones
who are right, and the subtraction cannot tell the two apart. Ranking plays by
the size of the gap makes it worse rather than better: our estimate is roughly
`true + error`, so sorting on the gap sorts largely on our own error and
reliably surfaces the games we understand worst.

The fix is to treat the market as a prior and our model as an observation.
With a normal prior on the true probability centred on the market, and a
normal likelihood for our estimate, the posterior is conjugate:

    w        = tau^2 / (tau^2 + sigma^2)
    posterior mean     = w * p_model + (1 - w) * p_market
    posterior variance = sigma^2 * w

`tau` is the typical size of a genuine market error and `sigma` is the standard
error of our own estimate. When our estimate is weak, w falls, our probability
collapses onto the market's, the edge vanishes and no play is produced. That
happens by construction rather than by anyone's judgement, which is the point.

Note the posterior standard deviation is `sigma * sqrt(w)`, not `sigma`. Using
the raw sigma -- as an earlier sketch of this did -- understates our confidence
after the prior has been folded in, because incorporating information can only
tighten the estimate. It errs conservative, but it is the wrong number and it
makes the reported z uninterpretable.

That matters because with the posterior in hand, z has a plain meaning:
`Phi(z)` is the probability that this bet is genuinely positive expected value.
That is a far more useful thing to show a reader than a test statistic, so both
are carried.

Everything here is a pure function. Nothing decides whether to place a bet.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt

from guards_report.betting import prices

# Conventional Kelly discount. Full Kelly is optimal only when the probability
# is known exactly; ours is an estimate, and Kelly's failure mode under an
# overconfident input is not mild -- it stakes hardest precisely where we are
# most wrong.
KELLY_FRACTION = 0.25


def _phi(z: float) -> float:
    """Standard normal CDF. scipy is not a dependency of this project."""
    return 0.5 * erfc(-z / sqrt(2.0))


@dataclass(frozen=True)
class Play:
    """One priced opportunity, with everything needed to recompute it by hand.

    Every field on the page comes from here. A reader who disagrees with the
    conclusion can check each step rather than having to trust the total.
    """

    market: str
    selection: str
    american: float
    decimal: float

    p_model: float
    sigma: float
    p_market: float
    tau: float

    weight: float          # w, how much of our own number survived
    p_used: float          # posterior mean
    posterior_sd: float    # sigma * sqrt(w)

    break_even: float      # 1/d, the number to clear -- vig included
    edge: float            # p_used - break_even
    expected_value: float  # per unit staked
    z: float
    confidence: float      # Phi(z): probability this is genuinely +EV
    stake: float           # fraction of bankroll, after the Kelly discount

    # The posted number for totals, run lines and props. Carried because
    # "over" at 8.5 and "over" at 9 are different bets that share a selection
    # name, and a record keyed without it silently keeps only one of them.
    line: float | None = None

    @property
    def disagreement(self) -> float:
        """Our probability minus the market's, before any shrinkage.

        Carried because it is what a reader intuitively expects "edge" to mean,
        and showing it beside the much smaller post-shrinkage edge is the
        clearest way to make the difference legible.
        """
        return self.p_model - self.p_market

    @property
    def is_playable(self) -> bool:
        """Positive expected value after the vig. Necessary, not sufficient --
        the z threshold is applied by the caller."""
        return self.expected_value > 0.0


def shrinkage_weight(sigma: float, tau: float) -> float | None:
    """How much of our own estimate survives contact with the market.

    Returns 0 when tau is 0: a market that is never wrong leaves nothing for us
    to add, and the correct number of bets is then zero forever. That is a
    legitimate answer rather than an edge case to work around.
    """
    if sigma is None or tau is None or sigma < 0 or tau < 0:
        return None
    denominator = tau * tau + sigma * sigma
    if denominator <= 0:
        return None
    return (tau * tau) / denominator


def assess(
    *,
    market: str,
    selection: str,
    american: float,
    p_model: float,
    line: float | None = None,
    sigma: float,
    p_market: float,
    tau: float,
    kelly_fraction: float = KELLY_FRACTION,
) -> Play | None:
    """Price one opportunity. Returns None if the inputs cannot support one.

    `p_market` must be the *de-vigged* market probability -- what the book
    believes. The break-even comparison is made separately against 1/d, which
    still carries the vig. Passing a de-vigged number in both places would make
    every price look about 2.4 points better than it is.
    """
    decimal = prices.american_to_decimal(american)
    if decimal is None:
        return None
    break_even = prices.break_even_probability(decimal)
    if break_even is None:
        return None
    if not (0.0 < p_model < 1.0) or not (0.0 < p_market < 1.0):
        return None
    if sigma is None or sigma <= 0.0:
        # A zero standard error claims we know the probability exactly. Nothing
        # downstream survives that: the weight goes to 1, the posterior width
        # goes to 0, and z goes to infinity on any edge at all.
        return None

    weight = shrinkage_weight(sigma, tau)
    if weight is None:
        return None

    p_used = weight * p_model + (1.0 - weight) * p_market
    posterior_sd = sigma * sqrt(weight)

    edge = p_used - break_even
    expected_value = edge * decimal

    if posterior_sd <= 0.0:
        # tau == 0: we have deferred entirely to the market and have no opinion
        # left to express. Report the play with no confidence rather than
        # dividing by zero to manufacture one.
        z = 0.0
        confidence = 0.0
    else:
        z = edge / posterior_sd
        confidence = _phi(z)

    return Play(
        market=market,
        selection=selection,
        american=float(american),
        decimal=decimal,
        line=line,
        p_model=p_model,
        sigma=sigma,
        p_market=p_market,
        tau=tau,
        weight=weight,
        p_used=p_used,
        posterior_sd=posterior_sd,
        break_even=break_even,
        edge=edge,
        expected_value=expected_value,
        z=z,
        confidence=confidence,
        stake=kelly_stake(p_used, decimal, kelly_fraction),
    )


def kelly_stake(
    p_used: float,
    decimal_odds: float,
    fraction: float = KELLY_FRACTION,
) -> float:
    """Fraction of bankroll to stake, after the Kelly discount.

    Full Kelly is `(p*d - 1) / (d - 1)`, which is the edge divided by the net
    price. Negative results are clamped to zero rather than returned as a
    signal to bet the other side -- the other side is its own market with its
    own vig, and it is almost never positive either.
    """
    if decimal_odds is None or decimal_odds <= 1.0:
        return 0.0
    full = (p_used * decimal_odds - 1.0) / (decimal_odds - 1.0)
    if full <= 0.0:
        return 0.0
    return full * fraction


def required_disagreement(
    *,
    sigma: float,
    tau: float,
    break_even: float,
    p_market: float,
    z_threshold: float,
) -> float | None:
    """How far our probability must sit from the market's to clear the bar.

    Solving the z definition for the raw disagreement:

        z = (w * D - v) / (sigma * sqrt(w))     with v = break_even - p_market
        D = (z * sigma * sqrt(w) + v) / w

    This exists to be shown rather than used. The figure it returns is the
    honest answer to "what would it actually take", and on a sharp market with
    a noisy model it comes back well above anything that has ever happened --
    which is worth knowing before a season of empty pages, not after.
    """
    weight = shrinkage_weight(sigma, tau)
    if weight is None or weight <= 0.0 or sigma <= 0.0:
        return None
    vig_share = break_even - p_market
    return (z_threshold * sigma * sqrt(weight) + vig_share) / weight
