"""The six batter evaluators.

Pitchers were the easy half: an arsenal is a small set of named things, each
with an obvious comparison. A hitter has no equivalent, so this file covers four
different frames and lets selection decide which is worth saying.

Two properties carry everything and neither announces itself when broken.

**Sign.** Every value is stored so that positive means better, which is what
lets selection rank a strength against a weakness and lets the contrast rule
find a pair pointing opposite ways. Chase rate and whiff rate are negated for
exactly that reason, and an un-negated one produces a finding that reads as a
compliment about a flaw.

**Handedness.** Pull is signed by which side the hitter bats from, so "pull"
means the same thing for a left-hander as a right-hander rather than meaning
"toward left field". Get that wrong and every left-handed hitter's spray chart
inverts, silently, while every number stays inside its plausible range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from guards_report.insight import batters


# ---------------------------------------------------------------------------
# Plate-appearance fixtures
# ---------------------------------------------------------------------------

def _pitches(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _pitch(**over) -> dict:
    row = {
        "batter": 1, "pitcher": 99, "stand": "R", "p_throws": "R",
        "events": None, "description": "ball", "zone": 5,
        "launch_speed": np.nan, "launch_angle": np.nan,
        "launch_speed_angle": np.nan, "bb_type": None,
        "hc_x": np.nan, "hc_y": np.nan,
        "woba_value": np.nan, "estimated_woba_using_speedangle": np.nan,
        "pitch_name": "4-Seam Fastball",
    }
    row.update(over)
    return row


def _contact(n: int, *, batter=1, bb_type="line_drive", speed=92.0,
             angle_class=np.nan, stand="R", hc_x=125.0, hc_y=100.0,
             woba=0.4, xwoba=0.4, throws="R") -> list[dict]:
    return [_pitch(batter=batter, events="single", description="hit_into_play",
                   bb_type=bb_type, launch_speed=speed,
                   launch_speed_angle=angle_class, stand=stand,
                   hc_x=hc_x, hc_y=hc_y, woba_value=woba,
                   estimated_woba_using_speedangle=xwoba, p_throws=throws)
            for _ in range(n)]


def _swings(n: int, *, batter=1, missed=False, zone=5) -> list[dict]:
    kind = "swinging_strike" if missed else "foul"
    return [_pitch(batter=batter, description=kind, zone=zone) for _ in range(n)]


# ---------------------------------------------------------------------------
# The league reference
# ---------------------------------------------------------------------------

def test_the_reference_is_built_only_from_players_with_a_real_sample():
    """A comparison is only as good as the population behind it.

    A reference that admitted twenty-plate-appearance callups would have its
    tails set by noise, and every percentile drawn from it would inherit that.
    """
    rows = _contact(200, batter=1) + _contact(5, batter=2)
    profile = batters.league_profile(_pitches(rows), minimum=150)
    assert 1 in profile.index
    assert 2 not in profile.index


def test_the_reference_carries_the_measures_the_evaluators_read():
    rows = []
    for batter in (1, 2, 3):
        rows += _contact(200, batter=batter, speed=88.0 + batter)
        rows += _swings(60, batter=batter, missed=batter == 1)
    profile = batters.league_profile(_pitches(rows), minimum=150)
    for column in ("barrel_rate", "exit_velocity", "whiff", "chase"):
        assert column in profile.columns, column


def test_batted_ball_shares_sum_to_one():
    """They are a decomposition, and one that does not close is a lost bucket."""
    rows = (_contact(80, batter=1, bb_type="line_drive")
            + _contact(80, batter=1, bb_type="ground_ball")
            + _contact(80, batter=1, bb_type="fly_ball"))
    profile = batters.league_profile(_pitches(rows), minimum=150)
    shares = [c for c in profile.columns if c.endswith("_share")]
    assert profile.loc[1, shares].sum() == pytest.approx(1.0)


def test_an_empty_frame_yields_an_empty_reference_rather_than_raising():
    empty = _pitches([_pitch()]).iloc[0:0]
    profile = batters.league_profile(empty, minimum=150)
    assert len(profile) == 0


# ---------------------------------------------------------------------------
# Spray — where handedness has to be handled
# ---------------------------------------------------------------------------

def test_pull_means_the_same_thing_for_both_batting_sides():
    """The bug this signing exists to prevent.

    A right-hander pulls toward left field and a left-hander toward right. Left
    unsigned, "pull rate" would mean "hit it to left field", every left-handed
    hitter's chart would invert, and every number would stay inside its
    plausible range while doing so.
    """
    league = pd.Series([0.40] * 30)
    # Toward left field: hc_x below the plate's x origin.
    to_left = _contact(120, stand="R", hc_x=60.0, hc_y=90.0)
    righty = batters.spray_tendency(
        batter_id=1, plate=_pitches(to_left), league_pull=league)

    to_right = _contact(120, stand="L", hc_x=190.0, hc_y=90.0)
    lefty = batters.spray_tendency(
        batter_id=1, plate=_pitches(to_right), league_pull=league)

    assert righty and lefty, "both sides should produce a finding"
    assert righty[0].value == pytest.approx(lefty[0].value, abs=0.05), (
        "a pulled ball is a pulled ball whichever side he bats from")


def test_a_thin_spray_sample_produces_nothing():
    league = pd.Series([0.40] * 30)
    assert batters.spray_tendency(
        batter_id=1, plate=_pitches(_contact(10)), league_pull=league) == []


def test_a_batted_ball_with_no_landing_point_is_not_counted():
    """Statcast leaves coordinates blank on some balls; those are unknown, not
    hit to center."""
    league = pd.Series([0.40] * 30)
    rows = _contact(120, hc_x=60.0, hc_y=90.0) + _contact(200, hc_x=np.nan)
    found = batters.spray_tendency(
        batter_id=1, plate=_pitches(rows), league_pull=league)
    assert found, "the 120 located balls should still qualify"


# ---------------------------------------------------------------------------
# Plate discipline — where the signs live
# ---------------------------------------------------------------------------

def _discipline(rows, **kw):
    return batters.plate_discipline(
        batter_id=1, plate=_pitches(rows),
        league_chase=pd.Series([0.28] * 40),
        league_whiff=pd.Series([0.22] * 40), **kw)


def test_chasing_is_stored_negated_so_positive_still_means_better():
    """Selection ranks on the value and pairs opposite signs into a contrast.

    An un-negated chase rate makes a free swinger look like a strength and
    hides the contrast that makes a scouting note read as one.
    """
    outside = [_pitch(zone=13, description="swinging_strike") for _ in range(150)]
    inside = [_pitch(zone=5, description="foul") for _ in range(100)]
    found = _discipline(outside + inside)
    chase = next((f for f in found if f.code == "bat.approach.chase"), None)
    assert chase is not None
    assert chase.value < 0, "a chase rate must be stored negative"


def test_a_disciplined_hitter_grades_above_a_free_swinger():
    patient = [_pitch(zone=13, description="ball") for _ in range(150)] + \
              [_pitch(zone=5, description="foul") for _ in range(100)]
    hacker = [_pitch(zone=13, description="swinging_strike") for _ in range(150)] + \
             [_pitch(zone=5, description="foul") for _ in range(100)]
    good = next(f for f in _discipline(patient) if f.code == "bat.approach.chase")
    poor = next(f for f in _discipline(hacker) if f.code == "bat.approach.chase")
    assert good.value > poor.value


def test_a_thin_sample_produces_no_discipline_finding():
    assert _discipline([_pitch(zone=13) for _ in range(20)]) == []


def test_too_few_pitches_outside_the_zone_withhold_the_chase_rate():
    """Two hundred pitches is enough to evaluate a hitter; two hundred pitches
    of which four are outside the zone is not enough to evaluate his chase."""
    rows = [_pitch(zone=5, description="foul") for _ in range(250)] + \
           [_pitch(zone=13, description="swinging_strike") for _ in range(4)]
    found = _discipline(rows)
    assert not any(f.code == "bat.approach.chase" for f in found)


# ---------------------------------------------------------------------------
# Platoon — the one inferential evaluator here
# ---------------------------------------------------------------------------

def test_a_platoon_finding_is_marked_inferential():
    """Most of this file is descriptive. Claiming a hitter has a platoon
    problem is a claim about a difference, and differences over small samples
    are where false findings come from -- so this one is gated."""
    rows = (_contact(120, throws="R", xwoba=0.42, woba=0.42)
            + _contact(120, throws="L", xwoba=0.24, woba=0.24))
    found = batters.platoon_split(
        batter_id=1, plate=_pitches(rows), league_gap=pd.Series([0.02] * 40))
    if found:
        assert found[0].inferential is True


def test_a_hitter_with_no_opposite_hand_sample_gets_no_split():
    """A split needs both sides. One side is a season line, not a split."""
    rows = _contact(300, throws="R")
    assert batters.platoon_split(
        batter_id=1, plate=_pitches(rows),
        league_gap=pd.Series([0.02] * 40)) == []


# ---------------------------------------------------------------------------
# Every evaluator, on the same contract
# ---------------------------------------------------------------------------

EVALUATORS = (
    ("batted_ball_profile", dict(league_profile=pd.DataFrame(
        {"barrel_rate": [0.07] * 30, "exit_velocity": [88.0] * 30,
         "line_drive_share": [0.25] * 30, "ground_ball_share": [0.43] * 30,
         "fly_ball_share": [0.25] * 30, "popup_share": [0.07] * 30}))),
    ("spray_tendency", dict(league_pull=pd.Series([0.40] * 30))),
    ("plate_discipline", dict(league_chase=pd.Series([0.28] * 30),
                              league_whiff=pd.Series([0.22] * 30))),
    ("platoon_split", dict(league_gap=pd.Series([0.02] * 30))),
    ("results_versus_contact", dict(league_gap=pd.Series([0.0] * 30))),
)


@pytest.mark.parametrize("name,kwargs", EVALUATORS)
def test_an_unknown_batter_yields_nothing_rather_than_raising(name, kwargs):
    """A bench player with no record is the normal case, not an error."""
    evaluator = getattr(batters, name)
    rows = _contact(300, batter=1) + _swings(300, batter=1)
    assert evaluator(batter_id=999_999, plate=_pitches(rows), **kwargs) == []


@pytest.mark.parametrize("name,kwargs", EVALUATORS)
def test_an_empty_frame_yields_nothing_rather_than_raising(name, kwargs):
    evaluator = getattr(batters, name)
    empty = _pitches([_pitch()]).iloc[0:0]
    assert evaluator(batter_id=1, plate=empty, **kwargs) == []


def test_pitch_type_weakness_needs_a_real_sample_of_that_pitch():
    """The minimum was raised from 40 to 90 after a note credited a hitter with
    a .053 expected wOBA on forty pitches -- a real number that said nothing."""
    rows = _contact(50, batter=1)
    for row in rows:
        row["pitch_name"] = "Slider"
    assert batters.pitch_type_weakness(
        batter_id=1, plate=_pitches(rows), league=_pitches(rows)) == []


def test_every_finding_carries_the_population_it_was_measured_against():
    """A value with no reference is a number, not a finding.

    Fixed thresholds have gone stale twice in this project, which is why the
    reference travels with the finding rather than living in a constant.
    """
    rows = (_contact(200, batter=1, hc_x=60.0, hc_y=90.0)
            + _swings(200, batter=1))
    found = batters.spray_tendency(
        batter_id=1, plate=_pitches(rows), league_pull=pd.Series([0.40] * 30))
    found += _discipline(
        [_pitch(zone=13, description="swinging_strike") for _ in range(150)]
        + [_pitch(zone=5, description="foul") for _ in range(100)])
    assert found
    for finding in found:
        assert finding.reference is not None
        assert finding.reference.population
        assert finding.evidence > 0
