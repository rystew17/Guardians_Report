"""Savant rate columns are normalized to percentages exactly once.

The trap: Savant's percentile and arsenal boards send 32.4 for 32.4%, while the
batted-ball and bat-tracking boards send 0.324 for the same figure. Player rows
and league benchmarks are read by different code paths, so the conversion has to
live somewhere both share -- otherwise a rate renders as "0.3%" and its
benchmark delta comes out around +32.
"""

from __future__ import annotations

from guards_report.ingest.preview import BATTED_BALL_FIELDS, _pick
from guards_report.metrics.league_averages import leaderboard_means
from guards_report.sources import savant as sv


def test_fraction_columns_become_percentages():
    row = {"pull_rate": "0.3269230769", "gb_rate": "0.3653846", "bbe": "52"}
    picked = _pick(row, ("pull_rate", "gb_rate", "bbe"))

    assert round(picked["pull_rate"], 1) == 32.7
    assert round(picked["gb_rate"], 1) == 36.5


def test_counts_are_not_scaled():
    # bbe is a count of batted balls, not a rate.
    assert _pick({"bbe": "52"}, ("bbe",))["bbe"] == 52.0


def test_already_percent_columns_are_left_alone():
    # Arsenal and percentile boards already send percentages.
    assert sv.scale_rate("whiff_percent", 32.4) == 32.4
    assert sv.scale_rate("k_percent", 22.1) == 22.1


def test_player_and_benchmark_share_one_convention():
    """The property that actually matters: a value and its benchmark compare."""
    rows = [
        {"bbe": "100", "pull_rate": "0.32"},
        {"bbe": "100", "pull_rate": "0.40"},
    ]
    player = _pick(rows[0], ("pull_rate",))["pull_rate"]
    league = leaderboard_means(rows, ("pull_rate",), weight_field="bbe")["pull_rate"]

    assert round(player, 1) == 32.0
    assert round(league, 1) == 36.0
    # The delta is a few points, not tens of points.
    assert abs(player - league) < 10


def test_directional_rates_sum_to_one_hundred():
    row = {"pull_rate": "0.3269230769", "straight_rate": "0.5", "oppo_rate": "0.1730769"}
    picked = _pick(row, ("pull_rate", "straight_rate", "oppo_rate"))
    assert round(sum(picked.values())) == 100


def test_every_batted_ball_rate_is_registered_as_a_fraction():
    """A new *_rate column must not quietly skip the conversion."""
    missed = [
        f for f in BATTED_BALL_FIELDS
        if f.endswith("_rate") and f not in sv.FRACTION_FIELDS
    ]
    assert not missed, f"rate columns not normalized: {missed}"
