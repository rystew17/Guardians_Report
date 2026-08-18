"""Tests for pitch-level Statcast aggregation.

The zone and spray builders are new hard-coded math over raw event rows, so
they get the same treatment as the rest of metrics/: hand-computable cases,
structural invariants, and explicit checks on the conventions that would
otherwise be silent assumptions.
"""

from __future__ import annotations

import pytest

from guards_report.metrics import statcast as sc


def pitch(**kwargs) -> dict[str, str]:
    """One Statcast row, with the fields the aggregator reads."""
    row = {
        "zone": "5", "stand": "R", "p_throws": "R",
        "description": "called_strike", "events": "",
        "estimated_woba_using_speedangle": "", "hc_x": "", "hc_y": "",
        "bb_type": "", "launch_speed": "",
    }
    row.update({k: str(v) for k, v in kwargs.items()})
    return row


# ---------------------------------------------------------------------------
# Zone aggregation
# ---------------------------------------------------------------------------


def test_slash_line_in_a_zone_is_hand_computable():
    rows = [
        pitch(zone="5", events="single", description="hit_into_play"),
        pitch(zone="5", events="home_run", description="hit_into_play"),
        pitch(zone="5", events="strikeout", description="swinging_strike"),
        pitch(zone="5", events="field_out", description="hit_into_play"),
        pitch(zone="5", events="walk", description="ball"),
    ]
    charts = sc.build_zone_charts(rows, perspective="batter")
    cell = charts["R"]["ops"].cells["5"]

    # 4 at-bats (walk excluded), 2 hits, 5 total bases, 5 plate appearances.
    assert cell.at_bats == 4
    assert cell.hits == 2
    assert cell.total_bases == 5
    assert cell.plate_appearances == 5
    assert cell.walks == 1
    assert cell.strikeouts == 1

    assert cell.avg == pytest.approx(0.5)
    assert cell.slg == pytest.approx(1.25)
    assert cell.obp == pytest.approx(3 / 5)
    assert cell.k_pct == pytest.approx(0.2)


def test_swing_and_whiff_rates_use_the_right_denominators():
    rows = [
        pitch(description="called_strike"),   # take
        pitch(description="ball"),            # take
        pitch(description="foul"),            # swing, contact
        pitch(description="swinging_strike"), # swing, whiff
        pitch(description="hit_into_play"),   # swing, contact
    ]
    chart = sc.build_zone_charts(rows, perspective="batter")["R"]["swing"]
    cell = chart.cells["5"]

    assert cell.pitches == 5
    assert cell.swings == 3
    assert cell.whiffs == 1
    # Swing rate is per pitch; whiff rate is per swing, not per pitch.
    assert cell.swing_pct == pytest.approx(3 / 5)
    assert cell.whiff_pct == pytest.approx(1 / 3)


def test_handedness_split_separates_and_all_combines():
    rows = [
        pitch(p_throws="L", events="single", description="hit_into_play"),
        pitch(p_throws="R", events="field_out", description="hit_into_play"),
    ]
    charts = sc.build_zone_charts(rows, perspective="batter")

    assert charts["L"]["avg"].cells["5"].hits == 1
    assert charts["R"]["avg"].cells["5"].hits == 0
    # "all" is the union, not a third independent bucket.
    assert charts["all"]["avg"].cells["5"].at_bats == 2
    assert charts["all"]["avg"].cells["5"].hits == 1


def test_pitcher_perspective_splits_on_batter_side():
    rows = [
        pitch(stand="L", p_throws="R", events="single", description="hit_into_play"),
        pitch(stand="R", p_throws="R", events="field_out", description="hit_into_play"),
    ]
    charts = sc.build_zone_charts(rows, perspective="pitcher")
    assert charts["L"]["avg"].cells["5"].hits == 1
    assert charts["R"]["avg"].cells["5"].hits == 0


def test_rows_outside_the_thirteen_zones_are_ignored():
    rows = [pitch(zone=""), pitch(zone="99"), pitch(zone="5")]
    chart = sc.build_zone_charts(rows, perspective="batter")["R"]["ops"]
    assert chart.pitches == 1


def test_unknown_handedness_is_dropped_rather_than_guessed():
    rows = [pitch(p_throws=""), pitch(p_throws="R")]
    charts = sc.build_zone_charts(rows, perspective="batter")
    assert charts["all"]["ops"].pitches == 1


def test_empty_cells_return_none_not_zero():
    chart = sc.build_zone_charts([], perspective="batter")["all"]["ops"]
    assert chart.value("5") is None
    assert chart.sample("5") == 0
    assert not chart.has_data


# ---------------------------------------------------------------------------
# Spray chart
# ---------------------------------------------------------------------------


def spray_row(hc_x: float, hc_y: float, events: str = "single") -> dict[str, str]:
    return pitch(
        hc_x=hc_x, hc_y=hc_y, events=events, bb_type="line_drive",
        description="hit_into_play", launch_speed="95.0",
    )


def test_right_handed_pull_is_negative_x():
    """A ball to left field from a right-handed hitter is a pull."""
    chart = sc.build_spray([spray_row(60, 100)], bats="R")
    assert chart.points[0].x < 0


def test_left_handed_hitters_are_mirrored_so_pull_matches():
    """The same physical direction should classify oppositely by handedness,
    and after mirroring both hitters' pulls land on the same side."""
    to_left_field = spray_row(60, 100)
    right = sc.build_spray([to_left_field], bats="R")
    left = sc.build_spray([to_left_field], bats="L")

    # Same batted ball, mirrored for the left-handed hitter.
    assert right.points[0].x == pytest.approx(-left.points[0].x)
    # For the right-hander that is a pull; for the left-hander, opposite field.
    assert right.pull_pct == 100.0
    assert left.oppo_pct == 100.0


def test_balls_behind_home_plate_are_discarded():
    """hc_y beyond home plate means a tracking artefact, not a batted ball."""
    chart = sc.build_spray([spray_row(125, 250)], bats="R")
    assert chart.points == []


def test_outcome_is_classified_from_events():
    rows = [
        spray_row(125, 100, "home_run"),
        spray_row(125, 150, "field_out"),
        spray_row(125, 120, "double"),
    ]
    chart = sc.build_spray(rows, bats="R")
    assert [p.outcome for p in chart.points] == ["home_run", "out", "double"]


def test_direction_shares_sum_to_one_hundred():
    rows = [
        spray_row(60, 100), spray_row(125, 100), spray_row(190, 100),
        spray_row(70, 120), spray_row(180, 130),
    ]
    chart = sc.build_spray(rows, bats="R")
    total = chart.pull_pct + chart.straight_pct + chart.oppo_pct
    assert total == pytest.approx(100.0)
