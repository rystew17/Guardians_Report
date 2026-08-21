"""Guarantees for the player props and the first-five model.

The props are where the pitch corpus finally paid, and they are also the easiest
place to publish a confident-looking number that means nothing. A .400 average
over twenty at-bats, a home-run rate from a September call-up, a batting order
ignored -- each would read plausibly and each is wrong.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from guards_report.projections import props


def _frame(n_batters: int = 6, per: int = 300) -> pd.DataFrame:
    """Synthetic plate appearances with known rates per batter."""
    rows = []
    rng = np.random.default_rng(4)
    for index in range(n_batters):
        rate = 0.10 + 0.06 * index          # 0.10 .. 0.40
        for i in range(per):
            hit = rng.random() < rate
            rows.append({
                "game_date": date(2024, 5, 1 + i % 28),
                "game_pk": 1000 + i,
                "at_bat_number": i,
                "batter": 100 + index,
                "pitcher": 900 + (i % 3),
                "stand": "L" if index % 2 else "R",
                "p_throws": "R",
                "events": "single" if hit else "field_out",
                "batting_team": "AAA",
                "home_team": "AAA" if i % 2 else "BBB",
                "away_team": "BBB" if i % 2 else "AAA",
                "season": 2024,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# log5
# --------------------------------------------------------------------------

def test_log5_returns_the_batter_rate_against_an_average_pitcher():
    assert props.log5(0.300, 0.250, 0.250) == pytest.approx(0.300)
    assert props.log5(0.100, 0.250, 0.250) == pytest.approx(0.100)


def test_log5_is_not_the_average_of_the_two_rates():
    """The whole reason for the odds form.

    Averaging would count the league baseline twice and flatten every extreme
    matchup, in a report whose purpose is to surface them.
    """
    combined = props.log5(0.300, 0.200, 0.250)
    assert combined < 0.250, "a good pitcher must pull a good hitter down"
    assert combined != pytest.approx(0.250)


def test_log5_stays_inside_the_unit_interval():
    for b in (0.001, 0.5, 0.999):
        for p in (0.001, 0.5, 0.999):
            value = props.log5(b, p, 0.25)
            assert 0.0 < value < 1.0


# --------------------------------------------------------------------------
# Shrinkage
# --------------------------------------------------------------------------

def test_a_tiny_sample_is_pulled_to_the_league_rate():
    """Extremes live in the smallest samples; ranking on raw rates is the trap."""
    frame = _frame()
    tiny = frame.head(20).copy()
    tiny["batter"] = 555
    tiny["events"] = "single"                  # a perfect 20-for-20
    model = props.fit_rates(pd.concat([frame, tiny]), "hit")

    assert model.batter_rate(555) < model.league * 1.15, "20-for-20 survived shrinkage"
    assert model.evidence(555, side="batter") == 20


def test_a_large_sample_keeps_its_own_rate():
    model = props.fit_rates(_frame(), "hit")
    # Batter 105 was generated at 0.40 over 300 chances; shrinkage should leave
    # him clearly above league without preserving him exactly.
    assert model.batter_rate(105) > model.league
    assert model.batter_rate(100) < model.league


def test_stabilisation_differs_by_outcome():
    """Strikeouts settle fast, batting average slowest -- measured, not assumed."""
    assert props.DEFAULT_STABILISATION["strikeout"] < props.DEFAULT_STABILISATION["home_run"]
    assert props.DEFAULT_STABILISATION["home_run"] < props.DEFAULT_STABILISATION["hit"]


# --------------------------------------------------------------------------
# Counts and the batting order
# --------------------------------------------------------------------------

def test_count_distribution_sums_to_one_and_matches_its_mean():
    d = props.count_distribution(0.25, props.SLOT_PA_DISTRIBUTION[1])
    assert sum(d.distribution.values()) == pytest.approx(1.0, abs=1e-6)
    assert d.expected == pytest.approx(
        sum(k * p for k, p in d.distribution.items()), abs=1e-9
    )
    assert d.at_least(1) == pytest.approx(1.0 - d.none)


def test_batting_order_changes_the_total_for_identical_hitters():
    """A leadoff spot is worth about a third more chances than the ninth."""
    lead = props.count_distribution(0.25, props.SLOT_PA_DISTRIBUTION[1])
    last = props.count_distribution(0.25, props.SLOT_PA_DISTRIBUTION[9])
    assert lead.expected > last.expected
    assert lead.expected - last.expected > 0.20
    assert lead.at_least(2) > last.at_least(2)


def test_at_least_two_is_never_above_at_least_one():
    for rate in (0.02, 0.25, 0.45):
        for slot in range(1, 10):
            d = props.count_distribution(rate, props.SLOT_PA_DISTRIBUTION[slot])
            assert d.at_least(2) <= d.at_least(1) <= 1.0


def test_an_unknown_slot_falls_between_the_extremes():
    unknown = props.count_distribution(0.25, props.UNKNOWN_SLOT_PA)
    lead = props.count_distribution(0.25, props.SLOT_PA_DISTRIBUTION[1])
    last = props.count_distribution(0.25, props.SLOT_PA_DISTRIBUTION[9])
    assert last.expected < unknown.expected < lead.expected


# --------------------------------------------------------------------------
# Adjustments
# --------------------------------------------------------------------------

def test_adjust_cannot_push_a_probability_past_one():
    assert props.adjust(0.95, 10.0) < 1.0
    assert props.adjust(0.001, 0.01) > 0.0


def test_adjust_is_neutral_at_a_factor_of_one():
    assert props.adjust(0.31, 1.0) == pytest.approx(0.31)
    assert props.adjust(0.31, 1.0, 1.0) == pytest.approx(0.31)


def test_platoon_factors_centre_on_one():
    frame = _frame()
    factors = props.platoon_factors(frame, "hit")
    assert factors
    assert 0.5 < min(factors.values()) and max(factors.values()) < 1.6


# --------------------------------------------------------------------------
# Strikeout totals
# --------------------------------------------------------------------------

def test_starter_strikeouts_scales_with_batters_faced():
    model = props.fit_rates(_frame(), "hit")
    card = list(range(100, 106))
    short = props.starter_strikeouts(model, pitcher_id=900, lineup_ids=card, expected_bf=12)
    long = props.starter_strikeouts(model, pitcher_id=900, lineup_ids=card, expected_bf=26)
    assert long.expected > short.expected
    assert sum(long.distribution.values()) == pytest.approx(1.0, abs=1e-4)


def test_strikeout_distribution_is_not_a_point_estimate():
    model = props.fit_rates(_frame(), "hit")
    projection = props.starter_strikeouts(
        model, pitcher_id=900, lineup_ids=list(range(100, 106)), expected_bf=24
    )
    assert len(projection.distribution) > 4, "a total with no spread is a guess"
    assert projection.at_least(1) > projection.at_least(5)
