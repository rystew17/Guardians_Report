"""Reading typed prices, grouping them into markets, and pricing a night.

The bugs this file is written against are all silent ones. Each produces a page
that renders cleanly and says something false:

* American odds handed to a function expecting decimals make every market
  unpriceable, which is indistinguishable from a night with no edge;
* a mistyped team code drops its selection, which looks the same again;
* an incomplete market normalizes to 1.0, which reads as a certainty;
* a probability with no measured standard error produces a confident stake from
  no evidence at all.
"""

from __future__ import annotations

from datetime import date

import pytest

from guards_report.betting import guide, sources
from guards_report.betting.guide import Belief
from guards_report.odds import manual, types

TODAY = date(2026, 8, 25)


def _markets(text: str):
    return manual.parse_markets(text, game_date=TODAY)


# ---------------------------------------------------------------------------
# Reading what was typed
# ---------------------------------------------------------------------------

def test_one_line_carries_both_sides():
    market, = _markets("ml CLE -135 DET +115")
    assert market.name == types.MONEYLINE
    assert market.selections() == ["CLE", "DET"]
    assert market.american() == [-135, 115]


def test_the_same_market_split_over_two_lines_is_the_same_market():
    one, = _markets("ml CLE -135 DET +115")
    two, = _markets("moneyline CLE -135\nmoneyline DET +115")
    assert one.american() == two.american()


def test_the_spellings_a_person_actually_types_are_understood():
    """Typed at speed, on a phone, in a hurry."""
    for text in ("ML  cle  -135   det  +115", "h2h CLE -135 / DET +115",
                 "Moneyline CLE -135 DET +115"):
        market, = _markets(text)
        assert market.name == types.MONEYLINE, text


def test_a_total_keeps_its_number_and_normalizes_the_side():
    market, = _markets("total o8.5 -105 u8.5 -115")
    assert market.line == 8.5
    assert market.selections() == ["over", "under"]


def test_a_prop_keeps_the_subject_on_both_sides():
    """Nobody types the pitcher's name twice, and a price with no player
    attached is worse than useless on a page."""
    market, = _markets("k Bibee o5.5 -120 u5.5 +100")
    assert market.selections() == ["Bibee over", "Bibee under"]
    assert market.line == 5.5


def test_comments_and_blank_lines_are_ignored():
    market, = _markets("# tonight, DraftKings\n\nml CLE -135 DET +115\n")
    assert len(market.quotes) == 2


def test_two_different_totals_are_two_different_markets():
    """Pairing the over on 8.5 with the under on 9 would produce a plausible
    overround describing a bet nobody can place."""
    markets = manual.parse_markets(
        "total o8.5 -105 u8.5 -115\ntotal o9 -120 u9 +100", game_date=TODAY)
    assert len(markets) == 2
    assert {m.line for m in markets} == {8.5, 9.0}


# ---------------------------------------------------------------------------
# What must be refused
# ---------------------------------------------------------------------------

def test_one_side_of_a_two_way_market_is_refused():
    """The input that must never pass. De-vigging a lone price normalizes it to
    1.0, which reads as a certainty rather than as a missing counterpart."""
    with pytest.raises(manual.ParseError, match="incomplete"):
        _markets("ml CLE -135")


@pytest.mark.parametrize("text,fragment", [
    ("xx CLE -135 DET +115", "unknown market"),
    ("ml -135", "no selection"),
    ("ml CLE", "no price"),
])
def test_a_typo_names_its_own_line(text, fragment):
    with pytest.raises(manual.ParseError, match=fragment):
        _markets(text)


def test_prices_convert_to_decimals_as_a_set_or_not_at_all():
    """The bug this cost an hour to find: everything in betting.prices works in
    decimals, quotes are stored American, and -135 handed to a decimal function
    reads as a price below evens rather than as an error."""
    market, = _markets("ml CLE -135 DET +115")
    decimals = market.decimal()
    assert decimals == pytest.approx([1.7407, 2.15], abs=1e-4)
    assert all(d > 1.0 for d in decimals)


def test_a_price_that_cannot_be_converted_takes_the_whole_market_with_it():
    broken = types.Market(name=types.MONEYLINE, quotes=[
        types.Quote(TODAY, types.MONEYLINE, "CLE", -135),
        types.Quote(TODAY, types.MONEYLINE, "DET", 50),   # no book quotes this
    ])
    assert broken.decimal() is None


# ---------------------------------------------------------------------------
# Pricing a night
# ---------------------------------------------------------------------------

MEASURED = Belief(probability=0.617, sigma=0.0142, measured=True, basis="test")
COMPLEMENT = Belief(probability=0.383, sigma=0.0142, measured=True, basis="test")


def test_a_market_is_actually_priced():
    night = guide.build(
        _markets("ml CLE -135 DET +115"),
        {(types.MONEYLINE, "cle"): MEASURED, (types.MONEYLINE, "det"): COMPLEMENT},
        tau=0.03, z_threshold=2.5)
    assert len(night.considered) == 2
    assert not night.warnings, night.warnings


def test_a_selection_we_cannot_match_is_reported_rather_than_dropped():
    """A mistyped team code and a night with no edge must not look the same."""
    night = guide.build(
        _markets("ml CLW -135 DET +115"),
        {(types.MONEYLINE, "cle"): MEASURED, (types.MONEYLINE, "det"): COMPLEMENT},
        tau=0.03, z_threshold=2.5)
    assert night.unmatched == ["moneyline: CLW"]


def test_an_unmeasured_standard_error_is_priced_but_never_staked():
    """The guard that matters most right now.

    Totals, first five and strikeout props have no calibration behind them, so
    a stake computed for them would be a confident number derived from no
    evidence about how wrong we tend to be.
    """
    guess = Belief(probability=0.65, sigma=0.005, measured=False, basis="test")
    other = Belief(probability=0.35, sigma=0.005, measured=False, basis="test")
    night = guide.build(
        _markets("total o8.5 -105 u8.5 -115"),
        {(types.TOTAL, "over"): guess, (types.TOTAL, "under"): other},
        tau=0.03, z_threshold=2.5)

    assert night.considered, "it must still be priced and shown"
    assert night.plays == [], "and never recommended"
    assert "total: over" in night.unstakeable


def test_a_night_that_recommends_nothing_says_so_deliberately():
    """The expected outcome most nights. It has to read as the system working,
    not as a page that failed to load."""
    night = guide.build(
        _markets("ml CLE -135 DET +115"),
        {(types.MONEYLINE, "cle"): MEASURED, (types.MONEYLINE, "det"): COMPLEMENT},
        tau=0.03, z_threshold=2.5)
    assert night.quiet
    assert night.plays == []


def test_a_big_enough_disagreement_does_produce_a_play():
    """The threshold has to be reachable, or none of this is worth building."""
    strong = Belief(probability=0.72, sigma=0.0142, measured=True, basis="test")
    weak = Belief(probability=0.28, sigma=0.0142, measured=True, basis="test")
    night = guide.build(
        _markets("ml CLE -135 DET +115"),
        {(types.MONEYLINE, "cle"): strong, (types.MONEYLINE, "det"): weak},
        tau=0.03, z_threshold=2.5)
    assert len(night.plays) == 1
    assert night.plays[0].selection == "CLE"
    assert night.plays[0].stake > 0
    assert not night.quiet


def test_plays_are_ranked_by_confidence_not_by_the_size_of_the_gap():
    """Ranking on the raw gap sorts on our own error. The whole selector exists
    to avoid that."""
    night = guide.build(
        _markets("ml CLE -200 DET +170"),
        {(types.MONEYLINE, "cle"): Belief(0.80, 0.0142, True),
         (types.MONEYLINE, "det"): Belief(0.20, 0.0142, True)},
        tau=0.03, z_threshold=1.0)
    zs = [p.z for p in night.considered]
    assert zs == sorted(zs, reverse=True)


def test_a_market_with_no_margin_is_flagged_loudly():
    """A feed that arrives already de-vigged makes every play look profitable."""
    night = guide.build(
        _markets("ml CLE +100 DET +100"),
        {(types.MONEYLINE, "cle"): MEASURED, (types.MONEYLINE, "det"): COMPLEMENT},
        tau=0.03, z_threshold=2.5)
    assert any("no margin" in w for w in night.warnings)


# ---------------------------------------------------------------------------
# Where the probabilities come from
# ---------------------------------------------------------------------------

def test_the_two_sides_of_a_moneyline_are_one_number_and_its_complement():
    class _Model:
        def win_probability(self, features): return 0.62
        win_cov = []
        win_calibration = []
        win_block_coef = []
        def win_probability_blocks(self, f): return []
        def win_probability_folds(self, f): return []

    beliefs = sources.moneyline(_Model(), {}, home="CLE", away="DET")
    assert beliefs[(types.MONEYLINE, "cle")].probability == pytest.approx(0.62)
    assert beliefs[(types.MONEYLINE, "det")].probability == pytest.approx(0.38)
    assert not beliefs[(types.MONEYLINE, "cle")].measured, (
        "a model with no calibration must not claim a measured sigma")


# The shape the simulator actually emits. An earlier fixture here used a plain
# mapping, which agreed with a wrong assumption in the code and passed while the
# real build raised on a list.
ROWS = [{"total": 8, "p": 0.3}, {"total": 9, "p": 0.4}, {"total": 10, "p": 0.3}]
MAPPING = {8: 0.3, 9: 0.4, 10: 0.3}


@pytest.mark.parametrize("distribution", [ROWS, MAPPING])
def test_a_total_landing_on_the_number_is_a_push_not_a_loss(distribution):
    """On a whole-number total the mass sitting exactly on the line is neither
    side. Counting it as a loss understates the over by all of it."""
    beliefs = sources.total({"total_distribution": distribution}, 9.0)
    # 0.3 over, 0.3 under, 0.4 pushed and removed from the denominator.
    assert beliefs[(types.TOTAL, "over")].probability == pytest.approx(0.5)


@pytest.mark.parametrize("distribution", [ROWS, MAPPING])
def test_a_half_point_total_has_nothing_to_push_on(distribution):
    beliefs = sources.total({"total_distribution": distribution}, 8.5)
    assert beliefs[(types.TOTAL, "over")].probability == pytest.approx(0.7)


def test_clearing_a_strikeout_line_means_the_next_whole_number():
    """5.5 is cleared by 6, not by 5. One step wrong here moves the probability
    by an entire bar of the distribution."""
    class _Prop:
        name = "Bibee"
        distribution = {4: 0.2, 5: 0.3, 6: 0.3, 7: 0.2}
        def at_least(self, k):
            return sum(p for n, p in self.distribution.items() if n >= k)

    beliefs = sources.strikeouts(_Prop(), 5.5)
    assert beliefs[(types.STRIKEOUTS, "bibee over")].probability == pytest.approx(0.5)


def test_a_first_five_tie_leaves_the_denominator():
    class _F5:
        home_leads, tied, away_leads = 0.45, 0.20, 0.35

    beliefs = sources.first_five(_F5(), home="CLE", away="DET")
    assert beliefs[(types.F5_MONEYLINE, "cle")].probability == pytest.approx(
        0.45 / 0.80)


def test_a_source_with_nothing_behind_it_returns_nothing():
    assert sources.total({}, 8.5) == {}
    assert sources.strikeouts(None, 5.5) == {}
    assert sources.first_five(None, home="CLE", away="DET") == {}
