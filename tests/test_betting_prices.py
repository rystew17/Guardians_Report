"""Odds conversion and margin removal.

These are the numbers every downstream decision is built on, and each has a
failure mode that produces a plausible price rather than an error. A de-vig
that runs backwards still returns probabilities summing to 1; a conversion off
by the American sign convention still returns a number in the right range.
Nothing downstream would flag either.

So the de-vig methods are pinned by the properties they must satisfy rather
than by figures transcribed from a formula, because a figure transcribed
wrongly and a test written to match it agree with each other perfectly.
"""

from __future__ import annotations

import pytest

from guards_report.betting import prices


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def test_the_standard_price_converts_the_way_the_industry_quotes_it():
    """-110 is the reference price for almost every market on the board."""
    assert prices.american_to_decimal(-110) == pytest.approx(1.9090909, abs=1e-6)
    assert prices.american_to_decimal(+150) == pytest.approx(2.50)
    assert prices.american_to_decimal(-200) == pytest.approx(1.50)
    assert prices.american_to_decimal(+100) == pytest.approx(2.00)


def test_conversion_round_trips_in_both_directions():
    for odds in (-500, -200, -110, -101, +100, +105, +150, +400, +1200):
        back = prices.decimal_to_american(prices.american_to_decimal(odds))
        assert back == pytest.approx(odds, abs=1e-6), odds


def test_odds_no_book_would_quote_are_refused_rather_than_guessed():
    """There is no legal American price between -100 and +100.

    Interpreting one would mean choosing a sign convention on the caller's
    behalf, and the two readings differ by the whole margin.
    """
    for odds in (0, 50, -50, 99, -99):
        assert prices.american_to_decimal(odds) is None
    assert prices.american_to_decimal(None) is None


def test_a_decimal_price_at_or_below_evens_is_refused():
    """Decimal odds include the stake, so 1.0 is a bet that returns nothing."""
    assert prices.decimal_to_american(1.0) is None
    assert prices.decimal_to_american(0.5) is None
    assert prices.break_even_probability(1.0) is None
    assert prices.break_even_probability(None) is None


# ---------------------------------------------------------------------------
# The number a bet has to clear
# ---------------------------------------------------------------------------

def test_the_break_even_rate_at_the_standard_price():
    """52.4% is the whole reason this is hard, so it is worth pinning."""
    decimal = prices.american_to_decimal(-110)
    assert prices.break_even_probability(decimal) == pytest.approx(0.5238, abs=1e-4)


def test_the_break_even_rate_still_carries_the_margin():
    """It is 1/d, not a fair probability.

    Both sides of a -110 market break even above 50%, and that is not a
    contradiction -- it is the vig, stated per side.
    """
    decimal = prices.american_to_decimal(-110)
    assert prices.break_even_probability(decimal) > 0.5


def test_the_margin_on_a_standard_two_way_market():
    both = [prices.american_to_decimal(-110)] * 2
    assert prices.overround(both) == pytest.approx(0.0476, abs=1e-4)


def test_a_market_with_no_margin_is_reported_as_such():
    """A feed that arrives already de-vigged would make every play look
    profitable, so a zero or negative overround has to be visible."""
    fair = [2.0, 2.0]
    assert prices.overround(fair) == pytest.approx(0.0, abs=1e-9)
    assert prices.overround([]) is None
    assert prices.overround([1.0, 2.0]) is None


# ---------------------------------------------------------------------------
# De-vigging: properties, not transcribed figures
# ---------------------------------------------------------------------------

METHODS = ("multiplicative", "shin", "power")

BALANCED = [prices.american_to_decimal(-110), prices.american_to_decimal(-110)]
TYPICAL = [prices.american_to_decimal(-155), prices.american_to_decimal(+135)]
LOPSIDED = [prices.american_to_decimal(-900), prices.american_to_decimal(+600)]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("market", [BALANCED, TYPICAL, LOPSIDED])
def test_every_method_returns_a_genuine_probability_distribution(method, market):
    fair = prices.fair_probabilities(market, method)
    assert fair is not None
    assert sum(fair) == pytest.approx(1.0, abs=1e-9)
    assert all(0.0 < p < 1.0 for p in fair)


@pytest.mark.parametrize("method", METHODS)
def test_removing_the_margin_lowers_every_side(method):
    """The book prices above 100%, so stripping it can only take probability
    away. A method that raised one side has run backwards."""
    raw = [prices.break_even_probability(d) for d in TYPICAL]
    fair = prices.fair_probabilities(TYPICAL, method)
    for before, after in zip(raw, fair):
        assert after < before


@pytest.mark.parametrize("method", METHODS)
def test_the_favorite_stays_the_favorite(method):
    """Ordering is the one thing no margin model may rearrange."""
    fair = prices.fair_probabilities(LOPSIDED, method)
    assert fair[0] > fair[1]


def test_the_methods_agree_on_a_balanced_market():
    """Nothing to disagree about when both sides are priced identically."""
    results = [prices.fair_probabilities(BALANCED, m) for m in METHODS]
    for fair in results:
        assert fair[0] == pytest.approx(0.5, abs=1e-6)


def test_shin_takes_more_from_the_longshot_than_proportional_scaling_does():
    """The substantive claim, and the reason to carry Shin at all.

    Bettors overbet longshots, so a posted long price implies more probability
    than the outcome deserves. Shin models the margin as protection against
    informed money and removes it unevenly; multiplicative scaling removes the
    same fraction from both sides and so preserves the bias it should correct.

    If this assertion ever flips, the solver is inverted -- and it would still
    be returning a tidy distribution summing to 1 while doing it.
    """
    mult = prices.fair_probabilities(LOPSIDED, "multiplicative")
    shin = prices.fair_probabilities(LOPSIDED, "shin")
    assert shin[1] < mult[1], "Shin must discount the longshot further"
    assert shin[0] > mult[0], "and hand that probability to the favorite"


def test_the_choice_of_method_barely_matters_on_a_moneyline():
    """Which is why it is not worth arguing about for sides and totals."""
    mult = prices.fair_probabilities(TYPICAL, "multiplicative")
    shin = prices.fair_probabilities(TYPICAL, "shin")
    assert abs(mult[0] - shin[0]) < 0.01


def test_the_choice_of_method_matters_a_great_deal_on_a_long_price():
    """And why it is load-bearing for props, where prices run long."""
    mult = prices.fair_probabilities(LOPSIDED, "multiplicative")
    shin = prices.fair_probabilities(LOPSIDED, "shin")
    assert abs(mult[1] - shin[1]) > 0.005


def test_an_unknown_method_raises_rather_than_falling_back():
    """A typo that silently selected a default would change every price on the
    page without changing anything visible on it."""
    with pytest.raises(ValueError):
        prices.fair_probabilities(TYPICAL, "proportional")


def test_a_one_sided_or_empty_market_yields_nothing():
    assert prices.fair_probabilities([1.9], "shin") is None
    assert prices.fair_probabilities([], "shin") is None


def test_an_already_fair_market_survives_de_vigging():
    """Opening day for a new feed: if it arrives without margin, every method
    must return it unchanged rather than dividing by zero."""
    for method in METHODS:
        fair = prices.fair_probabilities([2.0, 2.0], method)
        assert fair == pytest.approx([0.5, 0.5], abs=1e-9)


# ---------------------------------------------------------------------------
# The Shin diagnostic
# ---------------------------------------------------------------------------

def test_the_insider_share_is_a_fraction_and_rises_with_the_margin():
    """z is worth showing: a market priced against informed money is exactly
    where our own confidence deserves the most suspicion."""
    thin = [prices.american_to_decimal(-105), prices.american_to_decimal(-105)]
    fat = [prices.american_to_decimal(-130), prices.american_to_decimal(-130)]
    z_thin = prices.shin_z(thin)
    z_fat = prices.shin_z(fat)
    assert 0.0 <= z_thin < 1.0
    assert 0.0 <= z_fat < 1.0
    assert z_fat > z_thin


def test_a_market_with_no_margin_reports_no_insiders():
    assert prices.shin_z([2.0, 2.0]) == pytest.approx(0.0)
    assert prices.shin_z([1.9]) is None
