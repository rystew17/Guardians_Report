"""Shrinkage, edge after the vig, and staking.

The failure modes here are quiet. Every one of them returns a number in a
plausible range, renders without complaint, and is wrong in a direction that
costs money:

* comparing against the de-vigged probability instead of 1/d makes every price
  look about 2.4 points better than it is -- more than the whole edge anyone
  is realistically hunting;
* dividing by sigma rather than the posterior standard deviation makes the
  reported confidence mean nothing in particular;
* an inverted shrinkage weight trusts our model most exactly when it is least
  reliable.

So the properties are pinned, not the arithmetic.
"""

from __future__ import annotations

import pytest

from guards_report.betting import edge, prices

STANDARD = prices.american_to_decimal(-110)
FAIR_AT_STANDARD = 0.5   # both sides of a -110/-110 market, de-vigged


def _play(**overrides):
    kwargs = dict(
        market="moneyline",
        selection="CLE",
        american=-110,
        p_model=0.58,
        sigma=0.03,
        p_market=FAIR_AT_STANDARD,
        tau=0.03,
    )
    kwargs.update(overrides)
    return edge.assess(**kwargs)


# ---------------------------------------------------------------------------
# The economics the whole feature has to clear
# ---------------------------------------------------------------------------

def test_agreeing_with_the_book_loses_money():
    """The thesis of the entire design, stated as a test.

    When our probability matches the market's exactly, we are not neutral --
    we are down the vig. Being right is not what gets paid; the price being
    wrong is. If this ever passes with a positive expected value, the
    comparison has been made against the de-vigged number instead of 1/d.
    """
    play = _play(p_model=FAIR_AT_STANDARD)
    assert play.edge < 0
    assert play.expected_value < 0
    assert play.stake == 0.0
    assert not play.is_playable


def test_the_break_even_is_the_vigged_number_not_the_market_belief():
    """1/d at -110 is 52.4%, while the de-vigged market belief is 50%.

    Those 2.4 points are the margin, and they must sit between the two.
    """
    play = _play()
    assert play.break_even == pytest.approx(0.5238, abs=1e-4)
    assert play.p_market == pytest.approx(0.5)
    assert play.break_even > play.p_market


def test_expected_value_turns_positive_only_once_the_vig_is_cleared():
    small = _play(p_model=0.53)      # ahead of the market, behind the vig
    large = _play(p_model=0.80)      # clear of both
    assert small.expected_value < 0
    assert large.expected_value > 0


# ---------------------------------------------------------------------------
# Shrinkage
# ---------------------------------------------------------------------------

def test_a_market_that_is_never_wrong_leaves_us_nothing_to_add():
    """tau = 0 means the closing line is exact, so the correct number of bets
    is zero forever. A legitimate answer, not an edge case."""
    assert edge.shrinkage_weight(0.03, 0.0) == 0.0
    play = _play(tau=0.0)
    assert play.weight == 0.0
    assert play.p_used == pytest.approx(play.p_market)
    assert play.expected_value < 0


def test_a_noisier_model_keeps_less_of_its_own_opinion():
    """The direction that matters. Inverted, this would trust our estimate
    most exactly where it is least reliable."""
    confident = edge.shrinkage_weight(0.01, 0.03)
    vague = edge.shrinkage_weight(0.10, 0.03)
    assert confident > vague
    assert 0.0 < vague < confident < 1.0


def test_the_shrunk_probability_lands_between_ours_and_the_market():
    play = _play(p_model=0.58, p_market=0.50)
    assert 0.50 < play.p_used < 0.58


def test_shrinkage_shrinks_the_disagreement_by_exactly_the_weight():
    play = _play(p_model=0.58, p_market=0.50)
    assert play.disagreement == pytest.approx(0.08)
    assert play.p_used - play.p_market == pytest.approx(play.weight * 0.08)


# ---------------------------------------------------------------------------
# The posterior, and what z means
# ---------------------------------------------------------------------------

def test_the_posterior_is_tighter_than_our_own_estimate():
    """Folding in a prior can only reduce uncertainty. Reporting sigma here
    instead of sigma*sqrt(w) understates our confidence and makes z
    uninterpretable as a probability."""
    play = _play()
    assert play.posterior_sd == pytest.approx(play.sigma * play.weight ** 0.5)
    assert play.posterior_sd < play.sigma


def test_confidence_is_the_probability_the_bet_is_actually_positive():
    """The number worth showing a reader, rather than a test statistic."""
    play = _play(p_model=0.80)
    assert 0.0 < play.confidence < 1.0
    assert play.confidence > 0.5, "a comfortably +EV play should read above half"

    losing = _play(p_model=0.50)
    assert losing.confidence < 0.5


def test_confidence_rises_with_the_edge_and_falls_with_our_noise():
    assert _play(p_model=0.70).confidence > _play(p_model=0.60).confidence
    assert _play(sigma=0.01).confidence > _play(sigma=0.08).confidence


# ---------------------------------------------------------------------------
# Staking
# ---------------------------------------------------------------------------

def test_a_losing_price_is_staked_at_nothing():
    """Never returned as a signal to bet the other side -- that side is its own
    market with its own vig, and is almost never positive either."""
    assert edge.kelly_stake(0.40, STANDARD) == 0.0
    assert edge.kelly_stake(0.5238, STANDARD) == pytest.approx(0.0, abs=1e-6)


def test_the_stake_is_the_kelly_fraction_of_full_kelly():
    full = edge.kelly_stake(0.70, STANDARD, fraction=1.0)
    quarter = edge.kelly_stake(0.70, STANDARD, fraction=0.25)
    assert quarter == pytest.approx(full * 0.25)


def test_the_default_discount_is_a_quarter():
    assert edge.KELLY_FRACTION == 0.25


def test_the_stake_is_computed_from_the_shrunk_probability():
    """Staking on the raw model output would size every bet as though the
    market contributed nothing."""
    play = _play(p_model=0.80)
    expected = edge.kelly_stake(play.p_used, play.decimal, edge.KELLY_FRACTION)
    assert play.stake == pytest.approx(expected)
    raw = edge.kelly_stake(play.p_model, play.decimal, edge.KELLY_FRACTION)
    assert play.stake < raw


def test_a_bigger_edge_stakes_more():
    assert _play(p_model=0.85).stake > _play(p_model=0.70).stake


# ---------------------------------------------------------------------------
# What it would actually take
# ---------------------------------------------------------------------------

def test_the_required_disagreement_round_trips():
    """Solve for the disagreement that clears a threshold, then feed it back
    and confirm z lands on that threshold."""
    sigma, tau, z_threshold = 0.03, 0.03, 2.5
    market = 0.50
    break_even = prices.break_even_probability(STANDARD)

    needed = edge.required_disagreement(
        sigma=sigma, tau=tau, break_even=break_even,
        p_market=market, z_threshold=z_threshold)

    play = _play(p_model=market + needed, sigma=sigma, tau=tau)
    assert play.z == pytest.approx(z_threshold, abs=1e-6)


def test_a_sharper_model_needs_a_smaller_disagreement():
    """The finding that decides whether this feature ever fires: the bar is set
    by our own standard error far more than by the threshold."""
    break_even = prices.break_even_probability(STANDARD)
    kwargs = dict(tau=0.03, break_even=break_even, p_market=0.50, z_threshold=2.5)
    noisy = edge.required_disagreement(sigma=0.03, **kwargs)
    sharp = edge.required_disagreement(sigma=0.015, **kwargs)
    assert sharp < noisy
    assert noisy > 0.10, "a 3-point standard error demands a disagreement no moneyline produces"


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_a_zero_standard_error_is_refused():
    """It claims we know the probability exactly: the weight goes to 1, the
    posterior width to 0, and z to infinity on any edge at all."""
    assert _play(sigma=0.0) is None
    assert _play(sigma=-0.01) is None


def test_impossible_probabilities_and_prices_are_refused():
    assert _play(p_model=0.0) is None
    assert _play(p_model=1.0) is None
    assert _play(p_market=1.5) is None
    assert _play(american=50) is None
