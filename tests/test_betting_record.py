"""Storing prices, judging them, and scoring against the close.

Three things that only matter over time, and whose failures are therefore
invisible on any single night:

* a quote captured and not kept is a closing-line comparison that can never be
  made, and the closing line is the only scoreboard here that converges inside a
  season;
* a verdict without its reason is worth nothing to a reader deciding whether to
  trust it;
* closing line value read with the sign inverted would report the market moving
  away from us as a win, forever, without ever looking wrong.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from guards_report.betting import clv, edge, verdict
from guards_report.odds import store, types

TODAY = date(2026, 8, 25)


# ---------------------------------------------------------------------------
# Keeping the prices
# ---------------------------------------------------------------------------

def test_a_price_survives_being_written(tmp_path):
    store.record(tmp_path, "ml CLE -135 DET +115", game_date=TODAY)
    quotes = store.for_game(tmp_path, TODAY)
    assert len(quotes) == 2
    assert {q.selection for q in quotes} == {"CLE", "DET"}


def test_an_empty_record_still_has_its_columns(tmp_path):
    """A column-less frame merges into nothing downstream and the failure lands
    a long way from here. This project has already shipped that once."""
    frame = store.load(tmp_path)
    assert list(frame.columns) == store.COLUMNS
    assert store.for_game(tmp_path, TODAY) == []
    assert store.latest_markets(tmp_path, TODAY) == []


def test_the_line_moving_is_kept_as_history_not_as_a_correction(tmp_path):
    """Both prices are the record. Overwriting the first would erase the only
    evidence that the market moved at all."""
    store.record(tmp_path, "ml CLE -135 DET +115", game_date=TODAY)
    store.record(tmp_path, "ml CLE -155 DET +130", game_date=TODAY)
    assert len(store.for_game(tmp_path, TODAY)) == 4


def test_pricing_uses_the_newest_quote_not_the_first(tmp_path):
    """Comparing our number against a price nobody can take any more is worse
    than not comparing it at all."""
    store.record(tmp_path, "ml CLE -135 DET +115", game_date=TODAY)
    store.record(tmp_path, "ml CLE -155 DET +130", game_date=TODAY)
    market, = store.latest_markets(tmp_path, TODAY)
    assert market.american() == [-155, 130]


def test_every_posted_line_survives_as_its_own_market(tmp_path):
    """This assertion has been both ways round, and the second is right.

    Dropping the line from the key collapsed a re-quote into one row, which was
    the intent -- but books also post alternate lines at the same moment, and
    against those the rule just kept whichever arrived last. Jose Ramirez was
    quoted +525 to homer and +7000 to homer twice; the second won, and once the
    calibrated-line filter refused it he had no home run market at all.

    A book offering 4.5 and 5.5 is offering two bets. Which of them can be
    priced is a question for the calibration record, answered downstream where
    the answer is known.
    """
    store.record(tmp_path, "k Bibee o5.5 -120 u5.5 +100", game_date=TODAY)
    store.record(tmp_path, "k Bibee o4.5 -130 u4.5 +110", game_date=TODAY)
    markets = store.latest_markets(tmp_path, TODAY)
    assert {m.line for m in markets} == {4.5, 5.5}


def test_a_re_quote_at_the_same_line_keeps_only_the_newer_price(tmp_path):
    """The half of the old rule that was right: a price that moved is one
    market, not two."""
    store.record(tmp_path, "k Bibee o5.5 -120 u5.5 +100", game_date=TODAY)
    store.record(tmp_path, "k Bibee o5.5 -145 u5.5 +120", game_date=TODAY)
    markets = store.latest_markets(tmp_path, TODAY)
    assert len(markets) == 1
    assert sorted(markets[0].american()) == [-145, 120]


def test_a_different_date_is_a_different_night(tmp_path):
    store.record(tmp_path, "ml CLE -135 DET +115", game_date=TODAY)
    store.record(tmp_path, "ml CLE -200 DET +170",
                 game_date=TODAY + timedelta(days=1))
    assert len(store.for_game(tmp_path, TODAY)) == 2


def test_a_typo_stores_nothing_at_all(tmp_path):
    """One side of a two-way market cannot be de-vigged, so it must not reach
    the record either."""
    with pytest.raises(ValueError):
        store.record(tmp_path, "ml CLE -135", game_date=TODAY)
    assert store.for_game(tmp_path, TODAY) == []


# ---------------------------------------------------------------------------
# Bet or don't, and why
# ---------------------------------------------------------------------------

def _play(**overrides) -> edge.Play:
    kwargs = dict(
        market="moneyline", selection="CLE", american=-110,
        p_model=0.62, sigma=0.0145, p_market=0.50, tau=0.03)
    kwargs.update(overrides)
    return edge.assess(**kwargs)


def test_an_unchecked_model_is_refused_whatever_the_numbers_say():
    """The strongest-looking row on the page is often this one."""
    call = verdict.decide(_play(p_model=0.95), measured=False, z_threshold=2.5,
                          basis="starter strikeout distribution")
    assert call.action == verdict.PASS_UNMEASURED
    assert not call.is_bet
    assert "how wrong it usually is" in call.reason


def test_the_reason_reads_as_a_sentence_not_a_fragment():
    """The basis strings sit mid-sentence elsewhere and start one here."""
    call = verdict.decide(_play(), measured=False, z_threshold=2.5,
                          basis="starter strikeout distribution")
    assert call.detail.startswith("Starter strikeout distribution")

    named = verdict.decide(_play(), measured=False, z_threshold=2.5,
                           basis="Model A, calibrated logistic")
    assert named.detail.startswith("Model A, calibrated logistic"), (
        "capitalize() would flatten this to 'Model a'")


def test_agreeing_with_the_book_is_refused_and_says_why():
    call = verdict.decide(_play(p_model=0.50), measured=True, z_threshold=2.5)
    assert call.action == verdict.PASS_PRICED_IN
    assert "margin" in call.detail


def test_an_edge_smaller_than_our_error_is_the_interesting_refusal():
    """A reader will want to argue with this one, so it has to show both the
    edge and the uncertainty it lost to."""
    call = verdict.decide(_play(p_model=0.56), measured=True, z_threshold=8.0)
    assert call.action == verdict.PASS_INSIDE_ERROR
    assert call.label == "Not yet"
    assert "z of" in call.detail


def test_a_clear_edge_becomes_a_bet_with_a_stake_in_the_sentence():
    call = verdict.decide(_play(p_model=0.75), measured=True, z_threshold=2.5)
    assert call.is_bet
    assert "% of bankroll" in call.detail


def test_a_quiet_night_reads_as_the_system_working():
    calls = [verdict.decide(_play(p_model=0.50), measured=True, z_threshold=2.5)]
    assert "Nothing to bet" in verdict.summarize(calls)
    assert verdict.summarize([]) == "No prices entered for tonight."


def test_the_summary_distinguishes_priced_in_from_unchecked():
    """Reporting four unchecked markets as 'already priced in' would claim the
    market agrees with us when we have not asked."""
    calls = [
        verdict.decide(_play(p_model=0.50), measured=True, z_threshold=2.5),
        verdict.decide(_play(p_model=0.80), measured=False, z_threshold=2.5),
    ]
    line = verdict.summarize(calls)
    assert "priced in" in line and "not checked" in line


def test_the_disclaimer_is_exactly_what_was_asked_for():
    assert verdict.DISCLAIMER == "Not Gambling Advice"


# ---------------------------------------------------------------------------
# Closing line value
# ---------------------------------------------------------------------------

def _movement(taken, closed, *, edge_value=0.02, action="bet") -> clv.Movement:
    return clv.Movement(
        game_date=TODAY, market="moneyline", selection="CLE", action=action,
        p_taken=taken, p_closed=closed, edge=edge_value)


def test_the_market_coming_toward_us_is_positive():
    """Reads correctly in both directions, so it has to be pinned. Inverted,
    every loss would report as a win and nothing would look wrong."""
    assert _movement(0.52, 0.55).points == pytest.approx(0.03)
    assert _movement(0.52, 0.55).beat_close
    assert not _movement(0.55, 0.52).beat_close


def test_only_the_side_we_leaned_counts_toward_the_record():
    """Both sides of every market are logged deliberately, but they are exact
    complements -- pooling both averages to precisely zero forever."""
    ours = _movement(0.52, 0.55, edge_value=0.02)
    theirs = _movement(0.48, 0.45, edge_value=-0.02)
    record = clv.summarize([ours, theirs])
    assert record.n == 1
    assert record.mean_points == pytest.approx(0.03)


def test_a_record_of_only_the_other_side_is_no_record():
    assert clv.summarize([_movement(0.48, 0.45, edge_value=-0.02)]) is None
    assert clv.summarize([]) is None


def test_a_short_record_admits_it_is_short():
    """Closing line value needs hundreds of observations before its mean
    separates from noise, and a page reporting it without saying so invites
    exactly the over-reading everything else here guards against."""
    assert not clv.summarize([_movement(0.52, 0.55)]).enough


def test_the_beat_rate_counts_what_it_says():
    record = clv.summarize([
        _movement(0.52, 0.55), _movement(0.52, 0.56), _movement(0.52, 0.50)])
    assert record.n == 3
    assert record.beat_rate == pytest.approx(2 / 3)


def test_a_play_survives_a_round_trip_through_the_log(tmp_path):
    play = _play(p_model=0.75)
    clv.log(tmp_path, [play], game_date=TODAY, actions={"moneyline:CLE": "bet"})
    frame = clv.load(tmp_path)
    assert len(frame) == 1
    assert frame.iloc[0]["action"] == "bet"
    assert float(frame.iloc[0]["p_market_taken"]) == pytest.approx(play.p_market)


def test_everything_considered_is_logged_not_only_what_was_bet(tmp_path):
    """A record holding bets alone cannot say whether the bar is set right,
    because it contains no examples of what was turned down."""
    clv.log(tmp_path,
            [_play(p_model=0.75), _play(selection="DET", p_model=0.51)],
            game_date=TODAY, actions={"moneyline:CLE": "bet"})
    frame = clv.load(tmp_path)
    assert len(frame) == 2
    assert set(frame["action"]) == {"bet", "considered"}


def test_the_same_selection_at_two_lines_is_two_records(tmp_path):
    """"Over" at 8.5 and "over" at 9 share a selection name and are different
    bets. Keyed without the line, the record keeps only one of them."""
    clv.log(tmp_path, [
        _play(market="total", selection="over", line=8.5, p_model=0.60),
        _play(market="total", selection="over", line=9.0, p_model=0.52),
    ], game_date=TODAY)
    frame = clv.load(tmp_path)
    assert len(frame) == 2
    assert set(frame["line"]) == {8.5, 9.0}


def test_closing_probabilities_skip_a_market_missing_a_side():
    """A one-sided closing price cannot be de-vigged, and pairing it against an
    opening price that could would compare two different quantities."""
    half = types.Market(name=types.MONEYLINE, quotes=[
        types.Quote(TODAY, types.MONEYLINE, "CLE", -135)])
    assert clv.closing_probabilities([half]) == {}


def test_the_close_is_read_from_a_complete_market():
    whole = types.Market(name=types.MONEYLINE, quotes=[
        types.Quote(TODAY, types.MONEYLINE, "CLE", -155),
        types.Quote(TODAY, types.MONEYLINE, "DET", 130)])
    closing = clv.closing_probabilities([whole])
    assert closing[(types.MONEYLINE, "cle")] > 0.5
    assert sum(closing.values()) == pytest.approx(1.0, abs=1e-9)
