"""Every market's record against outcomes, and the standard error it produces.

Before this existed the betting page could stake a moneyline and refused
everything else, correctly: without a record of how often a stated probability
comes true there is no honest standard error, and a stake is a claim about
exactly that.

The failures here are the quiet kind. A harness that measures the wrong
population still returns a tidy table; a lookup that misses still returns a
number; an as-of merge with the wrong flag grades the model on the games it is
predicting and reports excellent calibration for it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from guards_report.betting import uncertainty
from guards_report.projections import calibrate

RECORD = (Path(__file__).resolve().parents[1]
          / "data" / "models" / "market_calibration.json")


# ---------------------------------------------------------------------------
# Reading the record
# ---------------------------------------------------------------------------

def _record(**bins) -> dict:
    return {name: {"market": name, "line": 0.5, "n": 1000,
                   "bins": rows, "seasons": [2025], "note": ""}
            for name, rows in bins.items()}


BINS = [
    {"n": 5000, "p_low": 0.30, "p_high": 0.55, "p_mean": 0.45,
     "frequency": 0.45, "sigma_systematic": 0.0, "resolution": 0.0070},
    {"n": 5000, "p_low": 0.55, "p_high": 0.90, "p_mean": 0.65,
     "frequency": 0.62, "sigma_systematic": 0.030, "resolution": 0.0067},
]


def test_a_market_with_no_record_cannot_be_staked():
    """The state every market except the moneyline was in. It must read as
    "not measured", never as agreement."""
    assert uncertainty.market_sigma({}, "hits", 0.6) is None
    assert uncertainty.market_sigma(_record(), "hits", 0.6) is None
    assert uncertainty.market_sigma(
        _record(hit=[]), "hits", 0.6) is None


def test_a_measured_market_returns_a_standard_error():
    sigma = uncertainty.market_sigma(_record(hit=BINS), "hits", 0.6)
    assert sigma is not None and sigma > 0


def test_the_betting_names_map_to_the_recorded_ones():
    """`hits` on the page is `hit` in the record, and both first-five markets
    read the same measurement. A missing entry here silently un-stakes a whole
    market."""
    for market in ("total", "f5_moneyline", "f5_total",
                   "strikeouts", "hits", "home_runs"):
        assert market in uncertainty.MARKET_KEYS, market


def test_one_deviant_bin_does_not_set_the_standard_error():
    """Same correction the win model needed: a per-bin figure is a noisy
    estimate floored at zero, and flooring one biases it upward."""
    sigma = uncertainty.market_sigma(_record(hit=BINS), "hits", 0.65)
    assert sigma < 0.030


def test_the_resolution_floor_binds_when_nothing_is_detectable():
    clean = [dict(row, frequency=row["p_mean"]) for row in BINS]
    sigma = uncertainty.market_sigma(_record(hit=clean), "hits", 0.45)
    assert sigma == pytest.approx(max(0.0070, uncertainty.MINIMUM_SIGMA))


def test_a_probability_outside_every_measured_region_still_prices():
    """A number the model has never produced before is not one to be confident
    about, but it still needs something."""
    sigma = uncertainty.market_sigma(_record(hit=BINS), "hits", 0.99)
    assert sigma is not None and sigma > 0


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------

def test_only_the_starting_nine_are_measured():
    """The bug that made the model look broken.

    A substitute enters in the seventh and gets one turn while the model assumes
    his slot's usual four -- because on the night it is projecting, he is not in
    the lineup at all. Left in, those rows were a tenth of the sample predicted
    at four times their real chances and pulled every bin three to eight points
    low, which read as the model overstating hits.
    """
    import inspect

    source = inspect.getsource(calibrate.batter_counts)
    assert 'games["slot"].between(1, 9)' in source


def test_the_as_of_merge_cannot_see_the_game_it_predicts():
    """`allow_exact_matches=False` is the whole difference between a
    calibration figure and a self-portrait."""
    import inspect

    source = inspect.getsource(calibrate._merge_as_of)
    assert "allow_exact_matches=False" in source


def test_the_prior_is_refitted_per_season():
    """The shipped artifact is fitted on every season including the ones being
    tested, so using it directly lets the model grade itself on data it has
    already seen."""
    import inspect

    for function in (calibrate.batter_counts, calibrate.strikeouts):
        source = inspect.getsource(function)
        assert 'plate["season"] < season' in source, function.__name__
        assert "props.fit_rates(before" in source, function.__name__


def test_a_market_with_nothing_to_measure_says_so():
    empty = pd.DataFrame(columns=["season", "game_pk", "game_date", "batter",
                                  "pitcher", "batting_team", "at_bat_number",
                                  "events", "stand", "p_throws"])
    result = calibrate.batter_counts(empty, outcome="hit", seasons=[2025])
    assert not result.measured
    assert result.note


def test_a_perfectly_calibrated_market_leaves_no_systematic_error():
    rng = np.random.default_rng(4)
    p = rng.uniform(0.3, 0.7, 40_000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    from guards_report.projections import backtest

    bins = backtest.calibration_bins(y, p)
    assert uncertainty.pooled_systematic(bins) < 0.01


# ---------------------------------------------------------------------------
# The shipped record
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not RECORD.is_file(), reason="markets not calibrated here")
def test_every_market_the_page_prices_has_been_measured():
    """Some markets carry one record and some carry one per line. Both count
    as measured; neither being present does not."""
    stored = json.loads(RECORD.read_text(encoding="utf-8"))
    for key in ("hit", "home_run", "strikeout", "total", "first_five"):
        assert key in stored, key
        block = stored[key]
        if block.get("by_line"):
            assert block["by_line"], f"{key} has no lines"
            for line, record in block["by_line"].items():
                assert record["bins"], f"{key} at {line} has no bins"
        else:
            assert block["bins"], f"{key} has no bins"
        assert block["n"] > 1000, f"{key} measured on too little"


@pytest.mark.skipif(not RECORD.is_file(), reason="markets not calibrated here")
def test_the_record_is_held_out_rather_than_fitted():
    """Seasons the model was not trained through, or the figure is circular."""
    stored = json.loads(RECORD.read_text(encoding="utf-8"))
    for key, block in stored.items():
        assert block["seasons"], key
        assert min(block["seasons"]) >= 2022, key


@pytest.mark.skipif(not RECORD.is_file(), reason="markets not calibrated here")
def test_every_market_produces_a_usable_standard_error():
    """Each at a line books actually post, since a per-line market has no
    figure to give without one -- deliberately, because reaching for a
    neighbour's is the failure this replaced."""
    stored = json.loads(RECORD.read_text(encoding="utf-8"))
    for market, line in (("total", 8.5), ("f5_moneyline", None),
                         ("strikeouts", 5.5), ("hits", 0.5),
                         ("home_runs", 0.5)):
        sigma = uncertainty.market_sigma(stored, market, 0.55, line)
        assert sigma is not None, market
        # Wide enough to be honest, narrow enough that a real edge can clear it.
        assert 0.001 < sigma < 0.15, f"{market}: {sigma}"


@pytest.mark.skipif(not RECORD.is_file(), reason="markets not calibrated here")
def test_a_strikeout_line_nobody_measured_gives_nothing():
    """The guard rather than the happy path: an unmeasured line has to come
    back empty so the guide refuses to stake it."""
    stored = json.loads(RECORD.read_text(encoding="utf-8"))
    assert uncertainty.market_sigma(stored, "strikeouts", 0.55, 12.5) is None


# ---------------------------------------------------------------------------
# How the page is laid out
# ---------------------------------------------------------------------------

def _row(market, selection, *, team="CLE", slot=None, edge=0.02, bet=False,
         line=0.5):
    from guards_report.betting import section as sec, verdict as vd

    action = vd.BET if bet else vd.PASS_INSIDE_ERROR
    return sec.Row(
        market=market, market_label=market.title(), selection=selection,
        line=line, american="+120", p_model=0.5, p_market=0.48,
        break_even=0.48, disagreement=0.02, edge=edge, sigma=0.02, z=1.0,
        confidence=0.8, stake=0.01, basis="t",
        verdict=vd.Verdict(action=action, label="x", reason="r", detail="d"),
        subject=selection.rsplit(" ", 1)[0], team=team, slot=slot)


def test_hitters_are_laid_out_as_lineups_not_as_a_list():
    """A hitter is one row and his bets are columns, the way the projections
    page already shows a batting order. Flattened by edge, the same nine names
    appeared four times each in an order nobody reads a lineup in."""
    from guards_report.betting import section as sec
    from guards_report.odds import types as ot

    block = sec.Section(teams=("CLE", "LAA"), rows=[
        _row(ot.HITS, "steven kwan over", team="CLE", slot=1),
        _row(ot.HITS, "steven kwan under", team="CLE", slot=1),
        _row(ot.HOME_RUNS, "steven kwan over", team="CLE", slot=1, bet=True),
        _row(ot.HITS, "mike trout over", team="LAA", slot=2),
    ])
    lineups = block.lineups
    assert [s["team"] for s in lineups] == ["CLE", "LAA"]
    kwan = lineups[0]["players"][0]
    assert kwan["hits_over"] is not None and kwan["hits_under"] is not None
    assert kwan["hr_over"] is not None
    assert lineups[0]["bets"] == 1


def test_the_home_run_under_column_appears_only_when_it_has_something_in_it():
    """Books post home runs to happen and not to not happen, so the column is
    usually empty and an always-blank column is noise."""
    from guards_report.betting import section as sec
    from guards_report.odds import types as ot

    over_only = sec.Section(teams=("CLE",), rows=[
        _row(ot.HOME_RUNS, "jose ramirez over", team="CLE")])
    assert not over_only.lineups[0]["hr_under"]

    both = sec.Section(teams=("CLE",), rows=[
        _row(ot.HOME_RUNS, "jose ramirez over", team="CLE"),
        _row(ot.HOME_RUNS, "jose ramirez under", team="CLE")])
    assert both.lineups[0]["hr_under"]


def test_each_starter_gets_his_own_table():
    """Two pitchers stacked together read as one list of eight strikeout prices
    with no indication which four belong to whom."""
    from guards_report.betting import section as sec
    from guards_report.odds import types as ot

    block = sec.Section(rows=[
        _row(ot.STRIKEOUTS, "gavin williams over", line=8.5),
        _row(ot.STRIKEOUTS, "gavin williams under", line=8.5),
        _row(ot.STRIKEOUTS, "walbert urena over", line=4.5, bet=True),
        _row(ot.STRIKEOUTS, "walbert urena under", line=4.5),
    ])
    arms = block.pitchers
    assert len(arms) == 2
    assert {a["subject"] for a in arms} == {"gavin williams", "walbert urena"}
    assert arms[0]["bets"] == 1, "the one with a bet leads"


def test_a_batting_order_is_used_when_it_is_known():
    from guards_report.betting import section as sec
    from guards_report.odds import types as ot

    carded = sec.Section(teams=("CLE",), rows=[
        _row(ot.HITS, "b over", team="CLE", slot=2),
        _row(ot.HITS, "a over", team="CLE", slot=1)])
    assert [p["slot"] for p in carded.lineups[0]["players"]] == [1, 2]
    assert carded.lineups[0]["carded"]

    uncarded = sec.Section(teams=("CLE",), rows=[
        _row(ot.HITS, "b over", team="CLE"),
        _row(ot.HITS, "a over", team="CLE")])
    assert not uncarded.lineups[0]["carded"]
    assert [p["name"] for p in uncarded.lineups[0]["players"]] == ["a", "b"]


def test_only_a_bet_or_a_real_near_miss_is_coloured():
    """An edge that rounds to +0.0 is not close, whatever its sign. Marking it
    implies a near miss where there is only a rounding artifact, and on a full
    board that is most of the colour on the page."""
    from guards_report.odds import types as ot

    assert _row(ot.HITS, "x over", bet=True).tone == "bet"
    assert _row(ot.HITS, "x over", edge=0.02).tone == "near"
    assert _row(ot.HITS, "x over", edge=0.0001).tone == ""


def test_game_bets_are_kept_apart_from_player_bets():
    from guards_report.betting import section as sec
    from guards_report.odds import types as ot

    block = sec.Section(rows=[
        _row(ot.MONEYLINE, "CLE", line=None),
        _row(ot.HITS, "steven kwan over"),
    ])
    assert [r.market for r in block.game_rows] == [ot.MONEYLINE]


# ---------------------------------------------------------------------------
# One record per line
# ---------------------------------------------------------------------------
# The error is not the same at every line. Measured on strikeouts, the model
# overstates by 1.8 points at 4.5 and understates by 3.5 at 6.5 and 8.5 -- the
# opposite direction and twice the size -- while the standard error doubles. A
# record taken at one line and applied to another carries a bias pointing the
# wrong way, which is not extrapolation so much as the wrong answer.

def _per_line(**lines) -> dict:
    return {"strikeout": {
        "market": "strikeout",
        "lines": list(lines),
        "by_line": {
            key: {"market": "strikeout", "line": float(key), "n": 4000,
                  "bins": rows, "seasons": [2025], "note": ""}
            for key, rows in lines.items()
        },
    }}


def _bins(systematic: float) -> list[dict]:
    return [
        {"n": 2000, "p_low": 0.10, "p_high": 0.50, "p_mean": 0.30,
         "frequency": 0.30 + systematic, "sigma_systematic": abs(systematic),
         "resolution": 0.010},
        {"n": 2000, "p_low": 0.50, "p_high": 0.95, "p_mean": 0.70,
         "frequency": 0.70 + systematic, "sigma_systematic": abs(systematic),
         "resolution": 0.010},
    ]


def test_each_line_is_read_against_its_own_record():
    record = _per_line(**{"4.5": _bins(-0.018), "8.5": _bins(+0.035)})
    near = uncertainty.market_sigma(record, "strikeouts", 0.55, 4.5)
    far = uncertainty.market_sigma(record, "strikeouts", 0.55, 8.5)
    assert near is not None and far is not None
    assert far > near, "the line with the larger measured error must price wider"


def test_a_line_with_no_record_is_refused_rather_than_borrowing_a_neighbour():
    """The whole failure this replaces. Reaching for the nearest measured line
    is what applied a 4.5 record, and its bias, to an 8.5 bet."""
    record = _per_line(**{"4.5": _bins(-0.018)})
    assert uncertainty.market_sigma(record, "strikeouts", 0.55, 8.5) is None
    assert uncertainty.market_sigma(record, "strikeouts", 0.55, 4.5) is not None


def test_a_per_line_record_needs_the_line_to_be_named():
    """Called without one there is no way to choose, and guessing is the bug."""
    record = _per_line(**{"4.5": _bins(-0.018)})
    assert uncertainty.market_sigma(record, "strikeouts", 0.55, None) is None


def test_the_line_is_matched_however_it_is_spelled():
    record = _per_line(**{"4.5": _bins(-0.018)})
    assert uncertainty.market_sigma(record, "strikeouts", 0.55, 4.50) is not None
    assert uncertainty._line_key(4.50) == uncertainty._line_key(4.5)


def test_a_single_record_market_still_works_without_lines():
    """Hits and home runs are posted at one number that matters, so they keep a
    single record and must not be broken by the per-line path."""
    single = {"hit": {"market": "hit", "line": 0.5, "n": 1000,
                      "bins": _bins(-0.02), "seasons": [2025], "note": ""}}
    assert uncertainty.market_sigma(single, "hits", 0.55, 0.5) is not None
    assert uncertainty.market_sigma(single, "hits", 0.55, None) is not None


@pytest.mark.skipif(not RECORD.is_file(), reason="markets not calibrated here")
def test_the_shipped_strikeout_record_covers_the_lines_books_post():
    stored = json.loads(RECORD.read_text(encoding="utf-8"))
    by_line = stored["strikeout"].get("by_line") or {}
    assert by_line, "strikeouts must be measured per line"
    for line in ("4.5", "5.5", "6.5", "7.5", "8.5"):
        assert line in by_line, f"no record at {line}"
        assert by_line[line]["bins"], f"{line} has no bins"


@pytest.mark.skipif(not RECORD.is_file(), reason="markets not calibrated here")
def test_the_measured_error_really_does_differ_by_line():
    """The finding this whole change rests on. If it ever stops being true, one
    record would do and this complexity is not paying for itself."""
    stored = json.loads(RECORD.read_text(encoding="utf-8"))
    by_line = stored["strikeout"]["by_line"]

    def gap(line: str) -> float:
        rows = by_line[line]["bins"]
        return sum(b["frequency"] - b["p_mean"] for b in rows) / len(rows)

    assert gap("4.5") < 0 < gap("8.5"), (
        f"expected the bias to flip sign: 4.5 {gap('4.5'):+.4f}, "
        f"8.5 {gap('8.5'):+.4f}")
