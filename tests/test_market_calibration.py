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
    stored = json.loads(RECORD.read_text(encoding="utf-8"))
    for key in ("hit", "home_run", "strikeout", "total", "first_five"):
        assert key in stored, key
        assert stored[key]["bins"], f"{key} has no bins"
        assert stored[key]["n"] > 1000, f"{key} measured on too little"


@pytest.mark.skipif(not RECORD.is_file(), reason="markets not calibrated here")
def test_the_record_is_held_out_rather_than_fitted():
    """Seasons the model was not trained through, or the figure is circular."""
    stored = json.loads(RECORD.read_text(encoding="utf-8"))
    for key, block in stored.items():
        assert block["seasons"], key
        assert min(block["seasons"]) >= 2022, key


@pytest.mark.skipif(not RECORD.is_file(), reason="markets not calibrated here")
def test_every_market_produces_a_usable_standard_error():
    stored = json.loads(RECORD.read_text(encoding="utf-8"))
    for market in ("total", "f5_moneyline", "strikeouts", "hits", "home_runs"):
        sigma = uncertainty.market_sigma(stored, market, 0.55)
        assert sigma is not None, market
        # Wide enough to be honest, narrow enough that a real edge can clear it.
        assert 0.001 < sigma < 0.15, f"{market}: {sigma}"
