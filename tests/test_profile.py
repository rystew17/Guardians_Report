"""Tool grades, player value, and the archetypes built on them.

The largest module in the package and the one whose failures are hardest to see.
Every number it produces is plausible by construction -- a percentile is always
between 0 and 100, a run value is always a small number of runs -- so a sign
error, a wrong population or a mis-weighted composite produces output that looks
exactly like correct output.

Four bugs were found here by reading the rendered page rather than by any test:
stolen bases filed most of the league at the 0th percentile, a composite quoted
a measure that had not driven it, an archetype asserted nothing played above
average beside a plus grade, and a rate scaled from fifty plate appearances put
a part-timer in the 98th percentile. Each is pinned below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest

from guards_report.insight import profile as prof


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _population(n: int = 200, seed: int = 4) -> pd.DataFrame:
    """A league-shaped reference, with every measure the tools read."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "batter": np.arange(1, n + 1),
        "season": 2026,
        "pa": rng.integers(120, 650, n),
        "k_rate": rng.normal(0.214, 0.060, n).clip(0.03, 0.45),
        "bb_rate": rng.normal(0.085, 0.030, n).clip(0.01, 0.24),
        "hr_rate": rng.normal(0.031, 0.015, n).clip(0.0, 0.09),
        "barrel_rate": rng.normal(0.077, 0.042, n).clip(0.0, 0.28),
        "ev": rng.normal(88.5, 2.45, n),
        "la": rng.normal(12.0, 5.0, n),
        "gb_rate": rng.normal(0.434, 0.067, n).clip(0.15, 0.70),
        "ld_rate": rng.normal(0.248, 0.033, n).clip(0.10, 0.42),
        "fb_rate": rng.normal(0.250, 0.055, n).clip(0.08, 0.50),
        "chase_rate": rng.normal(0.286, 0.059, n).clip(0.08, 0.55),
        "whiff_per_swing": rng.normal(0.223, 0.059, n).clip(0.03, 0.45),
        "xwoba": rng.normal(0.317, 0.040, n),
        "max_ev": rng.normal(109.0, 4.0, n),
        "hard_rate": rng.normal(0.38, 0.08, n).clip(0.05, 0.75),
        "bat_rv": rng.normal(0.0, 9.0, n),
    })


@dataclass
class _Box:
    player_id: int = 1
    position: str = "LF"
    season: dict = field(default_factory=lambda: {
        "gamesPlayed": 150, "singles": 100, "baseOnBalls": 50,
        "hitByPitch": 5, "stolenBases": 10, "caughtStealing": 3,
    })
    fielding: dict = field(default_factory=lambda: {
        "outs_above_average": 0.0, "fielding_runs_prevented": 0.0})
    running: dict = field(default_factory=lambda: {"sprint_speed": 27.3})
    baserunning_runs: float | None = None
    batting_runs: float | None = None
    batting_order: int | None = None


def _row(reference: pd.DataFrame, **over) -> dict:
    row = reference.iloc[0].to_dict()
    row.update(over)
    return row


# ---------------------------------------------------------------------------
# Tool grading — signs are the whole content
# ---------------------------------------------------------------------------

def test_a_low_strikeout_rate_grades_as_good_bat_to_ball():
    """`k_rate` enters negated. Unsigned, an undisciplined slugger reads balanced."""
    pop = _population()
    good = prof.grade_tools(
        _row(pop, k_rate=0.08, whiff_per_swing=0.10), pop, prof.BATTER_TOOLS)
    poor = prof.grade_tools(
        _row(pop, k_rate=0.36, whiff_per_swing=0.38), pop, prof.BATTER_TOOLS)
    assert good["contact"].grade > 80 > poor["contact"].grade


def test_a_high_chase_rate_grades_as_poor_discipline():
    pop = _population()
    patient = prof.grade_tools(
        _row(pop, bb_rate=0.17, chase_rate=0.16), pop, prof.BATTER_TOOLS)
    hacker = prof.grade_tools(
        _row(pop, bb_rate=0.03, chase_rate=0.44), pop, prof.BATTER_TOOLS)
    assert patient["discipline"].grade > 80 > hacker["discipline"].grade


def test_a_high_walk_rate_for_a_pitcher_grades_as_poor_command():
    """The same measure, opposite sign, on the other side of the plate."""
    pop = pd.DataFrame({
        "pitcher": range(1, 101), "season": 2026, "bf": 400,
        "k_rate": np.linspace(0.12, 0.36, 100),
        "whiff_per_swing": np.linspace(0.15, 0.38, 100),
        "bb_rate": np.linspace(0.03, 0.15, 100),
        "zone_rate": np.linspace(0.38, 0.58, 100),
        "gb_rate": np.linspace(0.30, 0.60, 100),
        "fb_rate": np.linspace(0.15, 0.40, 100),
        "barrel_allowed": np.linspace(0.02, 0.14, 100),
        "ev_allowed": np.linspace(85.0, 93.0, 100),
        "mix": np.linspace(2, 6, 100),
        "primary_share": np.linspace(0.25, 0.75, 100),
    })
    wild = prof.grade_tools(
        {**pop.iloc[0].to_dict(), "bb_rate": 0.15, "zone_rate": 0.38},
        pop, prof.PITCHER_TOOLS)
    sharp = prof.grade_tools(
        {**pop.iloc[0].to_dict(), "bb_rate": 0.03, "zone_rate": 0.58},
        pop, prof.PITCHER_TOOLS)
    assert sharp["command"].grade > 80 > wild["command"].grade


def test_a_tool_records_what_each_measure_contributed():
    """Without this the note quotes a fixed measure and can contradict itself.

    A composite of exit velocity and line drives that always quoted exit
    velocity produced "elite contact quality, 89.1 mph" for one player and
    "poor, 87.2 mph" for another -- both true, jointly unbelievable.
    """
    pop = _population()
    tools = prof.grade_tools(_row(pop, ev=88.5, ld_rate=0.40), pop, prof.BATTER_TOOLS)
    hard = tools["hard"]
    assert set(hard.contributions) == {"ev", "ld_rate"}
    assert hard.driver == "ld_rate", "the measure that earned it must be quoted"


def test_a_measure_the_reference_lacks_is_skipped_rather_than_guessed():
    pop = _population().drop(columns=["ld_rate"])
    tools = prof.grade_tools(_row(pop, ev=95.0), pop, prof.BATTER_TOOLS)
    assert "ld_rate" not in tools["hard"].contributions
    assert np.isfinite(tools["hard"].grade)


def test_a_player_missing_every_measure_of_a_tool_gets_no_tool():
    pop = _population()
    row = {k: None for k in pop.columns}
    tools = prof.grade_tools(row, pop, prof.BATTER_TOOLS)
    assert "power" not in tools


def test_grades_are_percentiles_of_the_composite_not_of_a_normal_curve():
    """Every player graded against the same population must span it."""
    pop = _population()
    grades = [
        prof.grade_tools(pop.iloc[i].to_dict(), pop, prof.BATTER_TOOLS)["power"].grade
        for i in range(len(pop))
    ]
    assert min(grades) < 5 and max(grades) > 95
    assert 45 < float(np.median(grades)) < 55


# ---------------------------------------------------------------------------
# Value — is he a good player
# ---------------------------------------------------------------------------

def test_batting_runs_prefer_the_measured_figure_over_the_estimate():
    """The two are not the same quantity and must not be blended silently.

    Savant's run value credits decisions -- laying off a shadow pitch -- that
    expected wOBA cannot see. On one card they correlated 0.56 with 9.6 runs of
    spread, which is the difference between a good season and an ordinary one.
    """
    assert prof.batting_runs(0.400, 600, measured=12.0) == 12.0
    estimate = prof.batting_runs(0.400, 600, measured=None)
    assert estimate == pytest.approx((0.400 - prof.LEAGUE_XWOBA) / prof.WOBA_SCALE * 600)


def test_an_average_hitter_is_worth_nothing_above_average():
    assert prof.batting_runs(prof.LEAGUE_XWOBA, 600) == pytest.approx(0.0)


def test_a_missing_expected_woba_contributes_zero_rather_than_a_penalty():
    assert prof.batting_runs(None, 600) == 0.0
    assert prof.batting_runs(float("nan"), 600) == 0.0


def test_baserunning_prefers_the_measured_run_value():
    """The estimate cannot see the half of baserunning without a throw.

    A runner who never steals but goes first to third all year reads as neutral
    on stolen bases and is not neutral.
    """
    season = _Box().season
    assert prof.baserunning_runs(season, measured=3.4) == 3.4


def test_being_caught_costs_more_than_stealing_gains():
    """Otherwise a low-percentage thief grades above a man who never runs."""
    assert abs(prof.RUN_CS) > prof.RUN_SB
    reckless = prof.baserunning_runs(
        {"singles": 100, "baseOnBalls": 50, "hitByPitch": 0,
         "stolenBases": 10, "caughtStealing": 10})
    still = prof.baserunning_runs(
        {"singles": 100, "baseOnBalls": 50, "hitByPitch": 0,
         "stolenBases": 0, "caughtStealing": 0})
    assert reckless < still


def test_a_player_who_never_reaches_first_is_not_penalised():
    assert prof.baserunning_runs(
        {"singles": 0, "baseOnBalls": 0, "hitByPitch": 0,
         "stolenBases": 0, "caughtStealing": 0}) == 0.0


def test_the_positional_adjustment_separates_a_catcher_from_a_first_baseman():
    """Outs above average is already position-relative.

    Without the adjustment a good defensive catcher and a good defensive first
    baseman carry the same figure and are not worth the same, and a glove-first
    shortstop reads as a bad player.
    """
    catcher = prof.fielding_runs(_Box(position="C"), games=150)
    first = prof.fielding_runs(_Box(position="1B"), games=150)
    assert catcher - first == pytest.approx(
        prof.POSITION_ADJUSTMENT["C"] - prof.POSITION_ADJUSTMENT["1B"])
    assert catcher > 0 > first


def test_the_positional_adjustment_is_prorated_by_playing_time():
    """A half-season catcher earns half the credit for catching."""
    full = prof.fielding_runs(_Box(position="C"), games=150)
    half = prof.fielding_runs(_Box(position="C"), games=75)
    assert half == pytest.approx(full / 2)


def test_an_unknown_position_carries_no_adjustment_rather_than_a_default():
    box = _Box(position="")
    assert prof.fielding_runs(box, games=150) == 0.0


def test_the_tiers_are_ordered_and_a_boundary_takes_the_better_one():
    labels = [prof.tier_for(v) for v in (40, 20, 5, -5, -30)]
    assert len(set(labels)) == 5
    for cut, label in prof.TIERS:
        assert prof.tier_for(cut) == label


def test_a_missing_value_yields_no_tier_rather_than_replacement_level():
    """Absent evidence is not evidence of a bad player."""
    assert prof.tier_for(None) == ""
    assert prof.tier_for(float("nan")) == ""


def test_value_is_the_three_terms_added():
    pop = _population()
    box = _Box(batting_runs=10.0, baserunning_runs=2.0, position="LF")
    box.fielding = {"fielding_runs_prevented": 4.0}
    total, per150 = prof.value_runs(box, xwoba=0.320, plate_appearances=600)
    expected = 10.0 + 2.0 + 4.0 + prof.POSITION_ADJUSTMENT["LF"]
    assert total == pytest.approx(expected)
    assert per150 == pytest.approx(expected)   # 150 games


def test_value_is_a_rate_so_a_short_season_is_not_penalised():
    """Six outs above average in forty games is elite, not ordinary."""
    box = _Box(batting_runs=5.0, position="2B")
    box.season = {**box.season, "gamesPlayed": 40}
    _, per150 = prof.value_runs(box, xwoba=0.320, plate_appearances=160)
    assert per150 > 5.0


def test_the_breakdown_splits_where_value_came_from():
    """A single total cannot separate a glove-first shortstop from a bat.

    Two players can arrive at the same number by opposite routes, and the whole
    point of the profile is to say which.
    """
    box = _Box(batting_runs=-8.0, baserunning_runs=1.0, position="SS")
    box.fielding = {"fielding_runs_prevented": 12.0}
    parts = prof.value_breakdown(box, xwoba=0.290, plate_appearances=500)
    assert parts["batting"]["runs"] == pytest.approx(-8.0)
    assert parts["fielding"]["runs"] > 0
    assert parts["total"]["runs"] == pytest.approx(
        sum(parts[k]["runs"] for k in ("batting", "baserunning", "fielding")))


# ---------------------------------------------------------------------------
# Archetypes
# ---------------------------------------------------------------------------

def _tools(**grades) -> dict:
    return {
        name: prof.Tool(name=name, label=name, grade=float(g), z=(g - 50) / 25)
        for name, g in grades.items()
    }


def test_a_player_may_match_several_profiles():
    """Players are several things, and forcing one label throws away the rest."""
    both = prof.match_profiles(
        _tools(power=92, discipline=90, contact=30, loft=70, hard=70),
        {}, kind="batter", limit=5)
    assert len(both) >= 2


def test_a_player_may_match_none():
    """Mid-table everywhere has no archetype, and inventing one manufactures it."""
    assert prof.match_profiles(
        _tools(power=50, discipline=50, contact=50, loft=50, hard=50),
        {}, kind="batter") == []


def test_matches_are_ranked_by_how_squarely_the_player_sits_in_them():
    found = prof.match_profiles(
        _tools(power=99, discipline=99, contact=40, loft=60, hard=60),
        {}, kind="batter", limit=5)
    assert found == sorted(found, key=lambda m: -m.strength)


def test_a_profile_asserting_an_absence_requires_one():
    """"Nothing carries him" must mean nothing carries him.

    Fringe Bat was firing beside a plus grade, so the note credited a player
    with plus baserunning in the sentence that said nothing played above
    average.
    """
    with_a_tool = prof.match_profiles(
        _tools(power=30, discipline=30, contact=30),
        {"speed": 95, "defense": 95, "baserunning": 95}, kind="batter", limit=6)
    assert not any(m.code == "fringe" for m in with_a_tool)

    without = prof.match_profiles(
        _tools(power=30, discipline=30, contact=30), {}, kind="batter", limit=6)
    assert any(m.code == "fringe" for m in without)


def test_a_glove_first_player_is_recognised_without_a_bat():
    found = prof.match_profiles(
        _tools(power=20, discipline=40, contact=45),
        {"defense": 95, "speed": 40}, kind="batter", limit=6)
    assert any(m.code == "glove_first" for m in found)


def test_five_tool_requires_all_of_them():
    """The rarest label in the taxonomy has to stay rare."""
    complete = prof.match_profiles(
        _tools(power=75, contact=75, discipline=70, loft=60, hard=70),
        {"speed": 75, "defense": 75}, kind="batter", limit=6)
    assert any(m.code == "five_tool" for m in complete)

    no_glove = prof.match_profiles(
        _tools(power=75, contact=75, discipline=70, loft=60, hard=70),
        {"speed": 75, "defense": 20}, kind="batter", limit=6)
    assert not any(m.code == "five_tool" for m in no_glove)


def test_an_unknown_tool_never_matches_rather_than_matching_on_a_default():
    """A missing grade is not a mid-table grade."""
    assert prof.match_profiles({}, {}, kind="batter") == []


def test_every_profile_carries_several_phrasings():
    for definition in prof.BATTER_PROFILES + prof.PITCHER_PROFILES:
        assert isinstance(definition.blurb, tuple), definition.code
        assert len(definition.blurb) >= 2, definition.code


# ---------------------------------------------------------------------------
# Derived percentiles
# ---------------------------------------------------------------------------

def test_an_unqualified_player_is_ranked_on_the_qualified_scale():
    """It answers "if he qualified, where would he rank".

    Ranking against everyone inflates a chip by up to 27 points, which is the
    distance between a bad hitter and an average one -- and the chip beside it
    is Savant's published figure, which is the qualified scale.
    """
    pop = _population(n=400)
    thin = _row(pop, pa=60, k_rate=0.30, bb_rate=0.04)
    got = prof.derived_percentiles(thin, pop)
    qualified = pop[pop["pa"] >= prof.QUALIFIED_PA]
    expected = (qualified["k_rate"] < 0.30).mean() * 100
    assert got["k_percent"] == pytest.approx(round(100 - expected), abs=1)


def test_max_exit_velocity_is_withheld_from_a_thin_sample():
    """A best-of-N statistic climbs with N, so thinness biases rather than blurs.

    It runs 106.6 mph in the 25-100 plate appearance band against 111.5 above
    400. A reader cannot tell a number depressed by playing time from one that
    is merely uncertain, so it is withheld rather than shown with a caveat.
    """
    pop = _population(n=400)
    assert "max_ev" not in prof.derived_percentiles(_row(pop, pa=60), pop)
    assert "max_ev" in prof.derived_percentiles(_row(pop, pa=600), pop)


def test_a_sample_too_thin_for_any_rate_yields_nothing():
    """Twenty plate appearances is an anecdote, not a rate."""
    pop = _population()
    assert prof.derived_percentiles(_row(pop, pa=12), pop) == {}


def test_a_lower_strikeout_rate_earns_a_higher_percentile():
    """Higher is better on every chip, whatever direction the measure runs."""
    pop = _population(n=400)
    # Each chip has to be moved by its own measure; varying one and asserting on
    # three passes for the wrong reason when two of them do not move at all.
    for chip, measure in (("k_percent", "k_rate"),
                          ("whiff_percent", "whiff_per_swing"),
                          ("chase_percent", "chase_rate")):
        low = prof.derived_percentiles(_row(pop, pa=600, **{measure: 0.09}), pop)
        high = prof.derived_percentiles(_row(pop, pa=600, **{measure: 0.40}), pop)
        assert low[chip] > high[chip], chip


def test_every_derived_percentile_is_in_range():
    pop = _population(n=300)
    for i in range(0, len(pop), 25):
        for chip, value in prof.derived_percentiles(pop.iloc[i].to_dict(), pop).items():
            assert 0 <= value <= 100, f"{chip} out of range: {value}"


# ---------------------------------------------------------------------------
# The lineup a pitcher faces
# ---------------------------------------------------------------------------

def test_a_lineup_aggregates_into_one_opponent():
    pop = _population(n=200)
    boxes = [_Box(player_id=int(pop.iloc[i]["batter"]), batting_order=i + 1)
             for i in range(9)]
    lineup = prof.lineup_profile(boxes, pop, {}, hand="R")
    assert lineup.hitters == 9
    assert set(lineup.tools) & {"power", "contact", "discipline"}
    assert lineup.hand == "R"


def test_the_leadoff_spot_counts_for_more_than_the_ninth():
    """He comes up 4.65 times to the ninth hitter's 3.79.

    A flat mean overstates the bottom of the order by roughly a fifth of a turn,
    which is the same magnitude as the batting-order effect the props model
    spends real machinery on.
    """
    assert prof.SLOT_WEIGHT[1] > prof.SLOT_WEIGHT[9]
    pop = _population(n=200)
    strong = int(pop.nlargest(1, "barrel_rate").iloc[0]["batter"])
    weak = int(pop.nsmallest(1, "barrel_rate").iloc[0]["batter"])

    top = prof.lineup_profile(
        [_Box(player_id=strong, batting_order=1), _Box(player_id=weak, batting_order=9)],
        pop, {})
    flipped = prof.lineup_profile(
        [_Box(player_id=weak, batting_order=1), _Box(player_id=strong, batting_order=9)],
        pop, {})
    assert top.grade("power") > flipped.grade("power")


def test_a_bench_bat_with_no_record_does_not_move_the_aggregate():
    """The aggregate describes the lineup, not its noisiest member."""
    pop = _population(n=200)
    real = [_Box(player_id=int(pop.iloc[i]["batter"]), batting_order=i + 1)
            for i in range(9)]
    with_ghost = real + [_Box(player_id=999_999, batting_order=None)]
    assert prof.lineup_profile(real, pop, {}).hitters == \
        prof.lineup_profile(with_ghost, pop, {}).hitters


def test_the_count_of_dangerous_bats_is_kept_beside_the_average():
    """An average hides a lineup split between four good bats and five outs."""
    pop = _population(n=200)
    boxes = [_Box(player_id=int(b), batting_order=i + 1)
             for i, b in enumerate(pop.nlargest(9, "barrel_rate")["batter"])]
    lineup = prof.lineup_profile(boxes, pop, {})
    assert lineup.counts.get("power", 0) >= 5


def test_an_empty_reference_yields_an_empty_lineup_rather_than_raising():
    assert prof.lineup_profile([_Box()], pd.DataFrame(), {}).hitters == 0
    assert prof.lineup_profile([], None, {}).tools == {}
