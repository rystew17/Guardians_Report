"""Where each market's probability comes from, and how sure we may be about it.

One market at a time, because they do not share a model and they do not share
an evidence base:

* **Moneyline** -- Model A, the calibrated logistic. The only market with a
  measured standard error behind it, so the only one that can currently be
  staked.
* **Total** -- Model B's simulated joint score distribution.
* **First five** -- the dedicated F5 fit.
* **Strikeouts** -- the starter's own distribution from the props model.

The last three are marked unmeasured. That is not a placeholder: no calibration
data exists for them, so there is no honest basis for a standard error, and
`guide.build` refuses to stake anything carrying that flag. They still appear on
the page, because a disagreement is worth seeing even when it cannot be sized.

Selections are keyed lower-case so a hand-typed `cle` matches `CLE`. Anything
that fails to match is surfaced by the guide rather than dropped, since a
mistyped team code and a night with no edge otherwise look identical.
"""

from __future__ import annotations

from typing import Any

from guards_report.betting import uncertainty
from guards_report.betting.guide import Belief
from guards_report.odds import types

# What a market's probability rests on, for the page to show beside it.
BASIS = {
    types.MONEYLINE: "Model A, calibrated logistic",
    types.TOTAL: "Model B, simulated score distribution",
    types.F5_MONEYLINE: "first-five model",
    types.F5_TOTAL: "first-five model",
    types.STRIKEOUTS: "starter strikeout distribution",
}


def moneyline(
    outcome_model, features: dict[str, float], *, home: str, away: str,
) -> dict[tuple[str, str], Belief]:
    """P(home win) and its complement, with the measured standard error."""
    p_home = outcome_model.win_probability(features)
    sigma = uncertainty.estimate(outcome_model, features)

    # The same sigma applies to both sides: they are one number and its
    # complement, so an error in one is exactly an error in the other.
    return {
        (types.MONEYLINE, home.lower()): Belief(
            probability=p_home, sigma=sigma.value,
            measured=sigma.measured, basis=BASIS[types.MONEYLINE]),
        (types.MONEYLINE, away.lower()): Belief(
            probability=1.0 - p_home, sigma=sigma.value,
            measured=sigma.measured, basis=BASIS[types.MONEYLINE]),
    }


def total(simulation: dict[str, Any], line: float) -> dict[tuple[str, str], Belief]:
    """P(total over the posted number), from the simulated distribution.

    Half-integer lines only in practice, but whole numbers are handled: a total
    landing exactly on the number is a push, which is neither a win nor a loss,
    so those draws are removed from the denominator rather than counted as
    either. Counting a push as a loss would understate the over by the entire
    probability mass sitting on the line, which on a total of 9 is not small.
    """
    distribution = simulation.get("total_distribution") or {}
    if not distribution:
        return {}

    over = pushed = under = 0.0
    for runs, weight in distribution.items():
        runs = float(runs)
        if runs > line:
            over += weight
        elif runs < line:
            under += weight
        else:
            pushed += weight

    live = over + under
    if live <= 0:
        return {}
    p_over = over / live

    return {
        (types.TOTAL, "over"): Belief(
            probability=p_over, sigma=uncertainty.MINIMUM_SIGMA,
            measured=False, basis=BASIS[types.TOTAL]),
        (types.TOTAL, "under"): Belief(
            probability=1.0 - p_over, sigma=uncertainty.MINIMUM_SIGMA,
            measured=False, basis=BASIS[types.TOTAL]),
    }


def first_five(projection, *, home: str, away: str) -> dict[tuple[str, str], Belief]:
    """First-five moneyline, with the tie removed.

    F5 is quoted three ways at most books but two ways on the main line, where
    a tie after five is a push. Same treatment as a total landing on the
    number: the tied mass leaves the denominator instead of being handed to one
    side.
    """
    if projection is None:
        return {}
    home_leads = float(getattr(projection, "home_leads", 0.0))
    away_leads = float(getattr(projection, "away_leads", 0.0))
    live = home_leads + away_leads
    if live <= 0:
        return {}
    p_home = home_leads / live

    return {
        (types.F5_MONEYLINE, home.lower()): Belief(
            probability=p_home, sigma=uncertainty.MINIMUM_SIGMA,
            measured=False, basis=BASIS[types.F5_MONEYLINE]),
        (types.F5_MONEYLINE, away.lower()): Belief(
            probability=1.0 - p_home, sigma=uncertainty.MINIMUM_SIGMA,
            measured=False, basis=BASIS[types.F5_MONEYLINE]),
    }


def strikeouts(prop, line: float) -> dict[tuple[str, str], Belief]:
    """P(starter goes over the posted strikeout number).

    `at_least(k)` is inclusive, so clearing a line of 5.5 means at least 6 --
    the ceiling of the line, not the line rounded. Getting that wrong by one
    moves the probability by a whole distribution bar, which on a strikeout
    prop is worth several points.
    """
    if prop is None or not getattr(prop, "distribution", None):
        return {}
    import math

    needed = math.floor(line) + 1
    p_over = float(prop.at_least(needed))
    name = (getattr(prop, "name", "") or "").strip().lower()
    if not name:
        return {}

    return {
        (types.STRIKEOUTS, f"{name} over"): Belief(
            probability=p_over, sigma=uncertainty.MINIMUM_SIGMA,
            measured=False, basis=BASIS[types.STRIKEOUTS]),
        (types.STRIKEOUTS, f"{name} under"): Belief(
            probability=1.0 - p_over, sigma=uncertainty.MINIMUM_SIGMA,
            measured=False, basis=BASIS[types.STRIKEOUTS]),
    }
