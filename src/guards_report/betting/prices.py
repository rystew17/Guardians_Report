"""Odds conversion and removing the bookmaker's margin.

Pure functions, same rules as metrics/formulas.py: no I/O, no state, and a
None return means "not computable from these inputs" rather than a zero.

The distinction this module exists to enforce is that a posted price carries
*two* different probabilities, and using the wrong one silently costs money in
opposite directions:

* `break_even_probability` is what the price charges us. It includes the vig,
  it is what a bet has to clear to be worth making, and it is the number the
  bet decision compares against. Sum it across both sides of a market and you
  get roughly 1.048, not 1.0 -- that excess is the bookmaker's margin.

* `fair_probabilities` is what the market *believes*, with the margin stripped
  out. It sums to 1 by construction. This is what we shrink our own estimate
  toward, and what closing line value is measured against.

Confusing them is not a rounding error. De-vigging before the bet decision
makes every price look about 2.4 points better than it is, which is more than
the entire edge anyone is realistically hunting. Comparing our probability to
the vigged number when measuring the market's opinion biases the market's
belief toward whichever side we happened to price.

The three de-vig methods disagree by a fraction of a point on a two-way
moneyline and by a great deal on a long price, so the choice is load-bearing
for props and nearly irrelevant for sides. It is left to the caller rather
than decided here.
"""

from __future__ import annotations

from typing import Iterable, Sequence

# Below this the bisections stop; well past the precision of any posted price.
_TOLERANCE = 1e-12
_MAX_STEPS = 200


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
# American odds are quoted as "risk this to win 100" when negative and "win
# this on 100" when positive, with no legal values between -100 and +100.
# Decimal odds are the total return per unit staked, including the stake, so
# they are always greater than 1.


def american_to_decimal(odds: float) -> float | None:
    """Convert American odds to decimal. -110 -> 1.909, +150 -> 2.50."""
    if odds is None:
        return None
    odds = float(odds)
    if -100 < odds < 100:
        # No book quotes these. Rather than pick an interpretation, refuse.
        return None
    if odds > 0:
        return odds / 100.0 + 1.0
    return 100.0 / abs(odds) + 1.0


def decimal_to_american(decimal_odds: float) -> float | None:
    """Convert decimal odds to American. The inverse of the above."""
    if decimal_odds is None or decimal_odds <= 1.0:
        return None
    if decimal_odds >= 2.0:
        return (decimal_odds - 1.0) * 100.0
    return -100.0 / (decimal_odds - 1.0)


def break_even_probability(decimal_odds: float) -> float | None:
    """The win rate a bet at this price needs just to break even.

    This is 1/d, and it includes the vig. At -110 it is 0.5238: we have to be
    right 52.4% of the time to lose nothing. Every bet decision compares our
    probability against *this* number, never against the de-vigged one.
    """
    if decimal_odds is None or decimal_odds <= 1.0:
        return None
    return 1.0 / decimal_odds


def overround(decimal_odds: Sequence[float]) -> float | None:
    """How much more than 100% the book has priced across a full market.

    Returns 0.048 for a typical -110/-110 two-way market. A market that comes
    back at or below zero has either been de-vigged already by whoever sent it
    or is not a complete market, and both are worth catching loudly -- a
    de-vigged feed passed off as raw makes every play look profitable.
    """
    raw = _raw_probabilities(decimal_odds)
    if raw is None:
        return None
    return sum(raw) - 1.0


def _raw_probabilities(decimal_odds: Sequence[float]) -> list[float] | None:
    """Break-even probabilities for every outcome in a market, unnormalized."""
    if not decimal_odds:
        return None
    out = []
    for price in decimal_odds:
        implied = break_even_probability(price)
        if implied is None:
            return None
        out.append(implied)
    return out


# ---------------------------------------------------------------------------
# Removing the margin
# ---------------------------------------------------------------------------


def fair_probabilities(
    decimal_odds: Sequence[float],
    method: str = "multiplicative",
) -> list[float] | None:
    """The market's implied beliefs with the bookmaker's margin removed.

    Sums to 1 by construction. `method` is one of:

    * `multiplicative` -- divide each price by the book sum. Assumes the
      margin is spread proportionally, which is close enough on a two-way
      moneyline and known to overprice favorites on a long board.
    * `shin` -- solves Hyun Song Shin's model, in which the margin exists to
      protect the book against bettors with private information. It takes more
      probability off the longshot than off the favorite, which is the
      direction the empirical bias actually runs.
    * `power` -- raises each price to a common exponent. A middle position,
      with no story behind it beyond fitting.

    Which one is right is an empirical question about a given market, not a
    matter of principle, so this returns whichever was asked for.
    """
    raw = _raw_probabilities(decimal_odds)
    if raw is None or len(raw) < 2:
        return None

    total = sum(raw)
    if total <= 0:
        return None

    if method == "multiplicative":
        return [p / total for p in raw]
    if method == "power":
        return _devig_power(raw)
    if method == "shin":
        return _devig_shin(raw)
    raise ValueError(f"unknown de-vig method: {method!r}")


def _devig_power(raw: Sequence[float]) -> list[float] | None:
    """Solve for k with sum(q_i ** k) == 1.

    Every q_i is below 1 and they sum above it, so raising them to a common
    k > 1 shrinks the total monotonically and the root is unique. Bisection is
    slower than Newton and cannot diverge, which matters more here -- this runs
    a few dozen times a night, and a solver that occasionally returns nonsense
    would produce a plausible-looking price rather than an error.
    """
    if any(p <= 0 or p >= 1 for p in raw):
        # A price at or beyond certainty has no exponent that normalizes it.
        return None

    low, high = 1.0, 2.0
    for _ in range(_MAX_STEPS):
        if sum(p ** high for p in raw) <= 1.0:
            break
        high *= 2.0
    else:
        return None

    for _ in range(_MAX_STEPS):
        mid = (low + high) / 2.0
        total = sum(p ** mid for p in raw)
        if abs(total - 1.0) < _TOLERANCE:
            break
        if total > 1.0:
            low = mid
        else:
            high = mid
    k = (low + high) / 2.0
    out = [p ** k for p in raw]
    return _normalized(out)


def _devig_shin(raw: Sequence[float]) -> list[float] | None:
    """Solve Shin's model for the insider proportion z, then invert it.

    For book sum P and raw price q_i, the fair probability under Shin is

        p_i = [sqrt(z^2 + 4(1-z) * q_i^2 / P) - z] / (2(1-z))

    and z is whatever makes those sum to 1. At z = 0 this reduces to dividing
    by sqrt(P), which still sums above 1, and the sum falls as z rises -- so
    the root is bracketed on [0, 1) and bisection finds it.

    z is worth reading: it is the share of money the model attributes to
    better-informed bettors. A market where it comes back near zero is one the
    book is pricing as if nobody knows anything it does not.
    """
    total = sum(raw)
    if total <= 1.0:
        # No margin to remove. Either already fair or not a complete market;
        # either way Shin has nothing to solve and would divide by zero trying.
        return _normalized(list(raw))

    def summed(z: float) -> float:
        acc = 0.0
        for q in raw:
            inner = z * z + 4.0 * (1.0 - z) * q * q / total
            if inner < 0.0:
                return float("inf")
            acc += (inner ** 0.5 - z) / (2.0 * (1.0 - z))
        return acc

    low, high = 0.0, 0.99
    if summed(low) < 1.0:
        return _normalized(list(raw))
    for _ in range(_MAX_STEPS):
        mid = (low + high) / 2.0
        value = summed(mid)
        if abs(value - 1.0) < _TOLERANCE:
            break
        if value > 1.0:
            low = mid
        else:
            high = mid
    z = (low + high) / 2.0

    out = []
    for q in raw:
        inner = z * z + 4.0 * (1.0 - z) * q * q / total
        out.append((inner ** 0.5 - z) / (2.0 * (1.0 - z)))
    return _normalized(out)


def shin_z(decimal_odds: Sequence[float]) -> float | None:
    """The insider share Shin's model attributes to this market.

    Exposed on its own because it is a diagnostic worth showing: a market
    priced with a high z is one the book thinks it is being picked off in, and
    that is precisely where our own confidence deserves the most suspicion.
    """
    raw = _raw_probabilities(decimal_odds)
    if raw is None or len(raw) < 2:
        return None
    total = sum(raw)
    if total <= 1.0:
        return 0.0

    def summed(z: float) -> float:
        acc = 0.0
        for q in raw:
            inner = z * z + 4.0 * (1.0 - z) * q * q / total
            if inner < 0.0:
                return float("inf")
            acc += (inner ** 0.5 - z) / (2.0 * (1.0 - z))
        return acc

    low, high = 0.0, 0.99
    if summed(low) < 1.0:
        return 0.0
    for _ in range(_MAX_STEPS):
        mid = (low + high) / 2.0
        value = summed(mid)
        if abs(value - 1.0) < _TOLERANCE:
            break
        if value > 1.0:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def _normalized(values: Iterable[float]) -> list[float] | None:
    """Force a sum of exactly 1, absorbing the solver's last few bits.

    The bisections land within 1e-12, which is far below anything that could
    matter, but a fair probability set that does not sum to 1 is a wrong claim
    however small the discrepancy, and later code is entitled to rely on it.
    """
    values = list(values)
    total = sum(values)
    if total <= 0:
        return None
    return [v / total for v in values]
