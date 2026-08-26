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
        {(types.MONEYLINE, "cle", None): MEASURED, (types.MONEYLINE, "det", None): COMPLEMENT},
        tau=0.03, z_threshold=2.5)
    assert len(night.considered) == 2
    assert not night.warnings, night.warnings


def test_a_selection_we_cannot_match_is_reported_rather_than_dropped():
    """A mistyped team code and a night with no edge must not look the same."""
    night = guide.build(
        _markets("ml CLW -135 DET +115"),
        {(types.MONEYLINE, "cle", None): MEASURED, (types.MONEYLINE, "det", None): COMPLEMENT},
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
        {(types.TOTAL, "over", 8.5): guess, (types.TOTAL, "under", 8.5): other},
        tau=0.03, z_threshold=2.5)

    assert night.considered, "it must still be priced and shown"
    assert night.plays == [], "and never recommended"
    assert "total: over" in night.unstakeable


def test_a_night_that_recommends_nothing_says_so_deliberately():
    """The expected outcome most nights. It has to read as the system working,
    not as a page that failed to load."""
    night = guide.build(
        _markets("ml CLE -135 DET +115"),
        {(types.MONEYLINE, "cle", None): MEASURED, (types.MONEYLINE, "det", None): COMPLEMENT},
        tau=0.03, z_threshold=2.5)
    assert night.quiet
    assert night.plays == []


def test_a_big_enough_disagreement_does_produce_a_play():
    """The threshold has to be reachable, or none of this is worth building."""
    strong = Belief(probability=0.72, sigma=0.0142, measured=True, basis="test")
    weak = Belief(probability=0.28, sigma=0.0142, measured=True, basis="test")
    night = guide.build(
        _markets("ml CLE -135 DET +115"),
        {(types.MONEYLINE, "cle", None): strong, (types.MONEYLINE, "det", None): weak},
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
        {(types.MONEYLINE, "cle", None): Belief(0.80, 0.0142, True),
         (types.MONEYLINE, "det", None): Belief(0.20, 0.0142, True)},
        tau=0.03, z_threshold=1.0)
    zs = [p.z for p in night.considered]
    assert zs == sorted(zs, reverse=True)


def test_a_market_with_no_margin_is_flagged_loudly():
    """A feed that arrives already de-vigged makes every play look profitable."""
    night = guide.build(
        _markets("ml CLE +100 DET +100"),
        {(types.MONEYLINE, "cle", None): MEASURED, (types.MONEYLINE, "det", None): COMPLEMENT},
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
    assert beliefs[(types.MONEYLINE, "cle", None)].probability == pytest.approx(0.62)
    assert beliefs[(types.MONEYLINE, "det", None)].probability == pytest.approx(0.38)
    assert not beliefs[(types.MONEYLINE, "cle", None)].measured, (
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
    assert beliefs[(types.TOTAL, "over", 9.0)].probability == pytest.approx(0.5)


@pytest.mark.parametrize("distribution", [ROWS, MAPPING])
def test_a_half_point_total_has_nothing_to_push_on(distribution):
    beliefs = sources.total({"total_distribution": distribution}, 8.5)
    assert beliefs[(types.TOTAL, "over", 8.5)].probability == pytest.approx(0.7)


def test_clearing_a_strikeout_line_means_the_next_whole_number():
    """5.5 is cleared by 6, not by 5. One step wrong here moves the probability
    by an entire bar of the distribution."""
    class _Prop:
        name = "Bibee"
        distribution = {4: 0.2, 5: 0.3, 6: 0.3, 7: 0.2}
        def at_least(self, k):
            return sum(p for n, p in self.distribution.items() if n >= k)

    beliefs = sources.strikeouts(_Prop(), 5.5)
    assert beliefs[(types.STRIKEOUTS, "bibee over", 5.5)].probability == pytest.approx(0.5)


def test_a_first_five_tie_leaves_the_denominator():
    class _F5:
        home_leads, tied, away_leads = 0.45, 0.20, 0.35

    beliefs = sources.first_five(_F5(), home="CLE", away="DET")
    assert beliefs[(types.F5_MONEYLINE, "cle", None)].probability == pytest.approx(
        0.45 / 0.80)


def test_a_source_with_nothing_behind_it_returns_nothing():
    assert sources.total({}, 8.5) == {}
    assert sources.strikeouts(None, 5.5) == {}
    assert sources.first_five(None, home="CLE", away="DET") == {}


# ---------------------------------------------------------------------------
# Prices that are wrong rather than generous
# ---------------------------------------------------------------------------

def test_a_wildly_different_price_is_flagged_not_staked():
    """The failure this exists for.

    A game already underway drifted the trailing side to +1600 while our
    pregame projection sat unchanged at 41.9%. The gap cleared every threshold
    and the page recommended staking almost eight percent of bankroll on a
    price that had moved because the game was being lost.

    Past thirty points the explanation is a bad price -- in-play odds, a
    mis-mapped market, a stale line -- and every one of them looks like free
    money.
    """
    confident = Belief(probability=0.42, sigma=0.0142, measured=True, basis="t")
    other = Belief(probability=0.58, sigma=0.0142, measured=True, basis="t")
    night = guide.build(
        _markets("ml LAA +1600 CLE -2000"),
        {(types.MONEYLINE, "laa", None): confident, (types.MONEYLINE, "cle", None): other},
        tau=0.03, z_threshold=2.5)

    assert night.plays == [], "a 30-point gap must never be staked"
    assert any("bad price" in w for w in night.warnings), night.warnings
    assert night.considered, "it is still shown, so the reader can see why"


def test_an_ordinary_disagreement_is_untouched():
    """The guard must not swallow the edges the feature exists to find."""
    strong = Belief(probability=0.72, sigma=0.0142, measured=True, basis="t")
    weak = Belief(probability=0.28, sigma=0.0142, measured=True, basis="t")
    night = guide.build(
        _markets("ml CLE -135 DET +115"),
        {(types.MONEYLINE, "cle", None): strong, (types.MONEYLINE, "det", None): weak},
        tau=0.03, z_threshold=2.5)
    assert len(night.plays) == 1
    assert not night.warnings


def test_the_threshold_sits_above_any_real_edge():
    """Wide enough that nothing genuine trips it, narrow enough to catch a
    price that has come from a different game state."""
    assert 0.20 <= guide.IMPLAUSIBLE_DISAGREEMENT <= 0.40


def test_two_lines_on_the_same_bet_do_not_answer_for_each_other():
    """Books post alternate numbers on the same market.

    Mike Trout's hits were quoted at both 0.5 and 1.5 on one night. Keyed
    without the line the second wrote over the first, and his chance of
    clearing 0.5 hits was reported as his chance of clearing 1.5 -- 79% against
    a truth of 37%, printed beside the 0.5 line.
    """
    class _Prop:
        name = "Mike Trout"
        hits = {"distribution": {0: 0.38, 1: 0.41, 2: 0.17, 3: 0.04}}

    low = sources.batter_prop(_Prop(), types.HITS, 0.5)
    high = sources.batter_prop(_Prop(), types.HITS, 1.5)

    assert (types.HITS, "mike trout over", 0.5) in low
    assert (types.HITS, "mike trout over", 1.5) in high
    assert not set(low) & set(high), "the two lines must not share a key"

    # over 0.5 is at least one hit; over 1.5 is at least two.
    assert low[(types.HITS, "mike trout over", 0.5)].probability == pytest.approx(0.62)
    assert high[(types.HITS, "mike trout over", 1.5)].probability == pytest.approx(0.21)


# ---------------------------------------------------------------------------
# Choosing which book to price against
# ---------------------------------------------------------------------------

def _book(book: str, over: float, under: float, *, line: float = 0.5):
    return types.Market(name=types.HITS, quotes=[
        types.Quote(TODAY, types.HITS, "vaughn grissom over", over,
                    book=book, line=line),
        types.Quote(TODAY, types.HITS, "vaughn grissom under", under,
                    book=book, line=line),
    ])


def test_a_book_disagreeing_with_every_other_one_is_dropped():
    """Two books priced the same hitter to record a hit at -189 and +340 --
    65% against 23%, so one of them is not the bet it claims to be. Both had a
    margin near six percent, so choosing on tightness could not tell them
    apart and the odd one won."""
    from guards_report.odds import client

    kept = client._one_book([
        _book("BetOnline.ag", -189, 143),
        _book("BetMGM", -175, 140),
        _book("DraftKings", 340, -500),
    ])
    assert len(kept) == 1
    assert kept[0].book != "DraftKings"


def test_a_single_book_is_kept_rather_than_discarded():
    """One book is not a consensus, but it is the only price there is."""
    from guards_report.odds import client

    kept = client._one_book([_book("BetOnline.ag", -189, 143)])
    assert len(kept) == 1


def test_different_lines_are_never_collapsed_into_one_choice():
    """0.5 and 1.5 on the same hitter are different bets, and each needs its
    own book."""
    from guards_report.odds import client

    kept = client._one_book([
        _book("BetOnline.ag", -189, 143, line=0.5),
        _book("DraftKings", 380, -550, line=1.5),
    ])
    assert {m.line for m in kept} == {0.5, 1.5}


# ---------------------------------------------------------------------------
# The run line
# ---------------------------------------------------------------------------

# Home wins by 2+ thirty percent of the time and wins at all forty-five.
MARGINS = {"margin_distribution": [
    {"margin": 3, "p": 0.20}, {"margin": 2, "p": 0.10}, {"margin": 1, "p": 0.15},
    {"margin": -1, "p": 0.15}, {"margin": -2, "p": 0.15}, {"margin": -3, "p": 0.25},
]}


def test_covering_a_favorite_line_is_harder_than_winning():
    """The sign, which reads plausibly in both directions.

    A home favorite is posted -1.5 and has to win by two. With the comparison
    the wrong way round it meant "wins by more than minus one and a half" --
    every win plus half the losses -- and the page had Cleveland covering -1.5
    at 73.5% while the same model won them the game 57.9%.
    """
    beliefs = sources.runline(MARGINS, -1.5, home="HOME", away="AWAY")
    covers = beliefs[(types.RUNLINE, "home", -1.5)].probability
    assert covers == pytest.approx(0.30)
    assert covers < 0.45, "a team cannot cover -1.5 more often than it wins"


def test_taking_the_points_is_easier_than_winning():
    beliefs = sources.runline(MARGINS, 1.5, home="HOME", away="AWAY")
    covers = beliefs[(types.RUNLINE, "home", 1.5)].probability
    assert covers == pytest.approx(0.60)
    assert covers > 0.45, "a team covers +1.5 more often than it wins outright"


def test_each_side_is_keyed_with_its_own_posted_number():
    """A home favorite is -1.5 and the away side +1.5. Keying both on the home
    figure left every away price unmatched."""
    beliefs = sources.runline(MARGINS, -1.5, home="HOME", away="AWAY")
    assert (types.RUNLINE, "home", -1.5) in beliefs
    assert (types.RUNLINE, "away", 1.5) in beliefs


def test_the_two_sides_of_a_run_line_sum_to_one():
    beliefs = sources.runline(MARGINS, -1.5, home="HOME", away="AWAY")
    home = beliefs[(types.RUNLINE, "home", -1.5)].probability
    away = beliefs[(types.RUNLINE, "away", 1.5)].probability
    assert home + away == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Names as rosters spell them and books do not
# ---------------------------------------------------------------------------

def test_a_name_matches_with_or_without_its_accents():
    """We carry "Walbert Urena" with a tilde and the board posts it without, so
    an exact match found nothing and his whole strikeout market went unpriced."""
    aliases = sources.name_aliases("walbert ureña")
    assert "walbert urena" in aliases
    assert "urena" in aliases
    assert "walbert ureña" in aliases


def test_stripping_accents_leaves_a_plain_name_alone():
    assert sources.strip_accents("gavin williams") == "gavin williams"
    assert sources.strip_accents("José Ramírez") == "Jose Ramirez"


# ---------------------------------------------------------------------------
# Markets books post one way only
# ---------------------------------------------------------------------------

def _one_sided(price: float = 540, line: float = 0.5):
    return types.Market(name=types.HOME_RUNS, quotes=[
        types.Quote(TODAY, types.HOME_RUNS, "mike trout over", price, line=line),
    ])


def test_a_home_run_market_survives_having_only_one_side():
    """Books offer "to hit a home run" and not its opposite. The bet is real
    and the price is real; what is missing is the counterpart that would let
    the margin be measured -- which is a reason to state an assumption, not to
    drop the market. Dropped, the whole home run board was silently absent."""
    assert _one_sided().complete
    assert _one_sided().one_way


def test_a_two_way_market_still_needs_both_sides():
    half = types.Market(name=types.HITS, quotes=[
        types.Quote(TODAY, types.HITS, "mike trout over", -170, line=0.5)])
    assert not half.complete


def test_a_one_way_market_is_priced_with_the_margin_declared():
    belief = Belief(probability=0.21, sigma=0.0067, measured=True, basis="t")
    night = guide.build(
        [_one_sided()],
        {(types.HOME_RUNS, "mike trout over", 0.5): belief},
        tau=0.03, z_threshold=2.5)

    assert night.considered, "the market must be priced"
    assert night.assumed_margin, "and the assumption must be reported"
    assert "mike trout over" in night.assumed_margin[0]


def test_the_assumed_margin_lowers_what_the_book_is_taken_to_believe():
    """An un-adjusted price reads as the market believing the outcome more than
    it does, which shrinks our number toward something too confident."""
    belief = Belief(probability=0.21, sigma=0.0067, measured=True, basis="t")
    night = guide.build(
        [_one_sided()],
        {(types.HOME_RUNS, "mike trout over", 0.5): belief},
        tau=0.03, z_threshold=2.5)
    play = night.considered[0]
    assert play.p_market < play.break_even
    assert play.p_market == pytest.approx(
        play.break_even / guide.ONE_WAY_OVERROUND)


def test_the_assumed_margin_is_a_plausible_one():
    """Fat, because the book knows nobody is taking the other side -- but not
    so fat that it manufactures an edge on every longshot."""
    assert 1.05 <= guide.ONE_WAY_OVERROUND <= 1.35


def test_a_one_way_market_is_not_reported_as_having_no_margin():
    """The zero-overround alarm exists for a feed that arrived de-vigged. A
    market that never had two sides is a different thing and must not trip it."""
    belief = Belief(probability=0.21, sigma=0.0067, measured=True, basis="t")
    night = guide.build(
        [_one_sided()],
        {(types.HOME_RUNS, "mike trout over", 0.5): belief},
        tau=0.03, z_threshold=2.5)
    assert not any("no margin" in w for w in night.warnings), night.warnings


# ---------------------------------------------------------------------------
# Two lines on the same player, offered at once
# ---------------------------------------------------------------------------

def test_only_the_calibrated_home_run_line_is_priced():
    """Books post "to hit a home run" at 0.5 and "two or more" at 1.5 side by
    side. Jose Ramirez was quoted +525 to homer and +7000 to homer twice, and
    the page showed +7000 -- the newest-quote rule keys on market and selection
    and not on the number, which is right for a line that moved and wrong for
    two lines offered at once.

    There is a second reason beyond picking the wrong bet: the calibration was
    measured at 0.5 and covers predictions down to 0.005. A two-homer bet sits
    at or below that, so its standard error would be borrowed from the nearest
    measured bin rather than measured -- which is exactly what the staking rule
    refuses to act on everywhere else.
    """
    from guards_report.betting import attach

    assert attach.CALIBRATED_LINE[types.HOME_RUNS] == 0.5
    assert attach.CALIBRATED_LINE[types.HITS] == 0.5


def test_the_calibrated_line_matches_what_was_measured():
    """A constant that drifts from the record it refers to reintroduces the
    borrowed standard error silently."""
    import json
    from pathlib import Path

    from guards_report.betting import attach

    record = (Path(__file__).resolve().parents[1]
              / "data" / "models" / "market_calibration.json")
    if not record.is_file():
        pytest.skip("markets not calibrated here")
    stored = json.loads(record.read_text(encoding="utf-8"))
    assert stored["home_run"]["line"] == attach.CALIBRATED_LINE[types.HOME_RUNS]
    assert stored["hit"]["line"] == attach.CALIBRATED_LINE[types.HITS]


def test_a_lineup_column_carries_the_number_it_is_about():
    """"HR over" with no line reads as one bet whichever line produced it."""
    from guards_report.betting import section as sec

    block = sec.Section(teams=("CLE",), rows=[
        _lineup_row(types.HITS, "jose ramirez over", 0.5),
        _lineup_row(types.HOME_RUNS, "jose ramirez over", 0.5),
    ])
    side = block.lineups[0]
    assert side["hit_line"] == 0.5
    assert side["hr_line"] == 0.5


def _lineup_row(market, selection, line):
    from guards_report.betting import section as sec, verdict as vd

    return sec.Row(
        market=market, market_label=market.title(), selection=selection,
        line=line, american="+525", p_model=0.12, p_market=0.14,
        break_even=0.16, disagreement=-0.02, edge=-0.04, sigma=0.007, z=-1.0,
        confidence=0.2, stake=0.0, basis="t",
        verdict=vd.Verdict(action=vd.PASS_PRICED_IN, label="No",
                           reason="r", detail="d"),
        subject=selection.rsplit(" ", 1)[0], team="CLE", slot=3)
