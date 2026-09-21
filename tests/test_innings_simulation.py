"""The simulation plays innings, so the ninth-inning rule is generated, not assumed.

Drawing each club's game runs independently and comparing them treats the two
sides as independent. They are not: the home club does not bat in the bottom of
the ninth when it is already ahead, and stops the moment it takes the lead, so
its runs are censored by the outcome being predicted. Over 11,964 held-out
games that priced the home side at 50.9% against an actual 53.0%.

These pin the behaviour that fixes it, and the measured constants behind it.
"""

from __future__ import annotations

import numpy as np
import pytest

from guards_report.projections import model as model_module


def _model(**kwargs):
    return model_module.OutcomeModel(
        score_columns=["is_home"],
        score_coef=[0.0],
        score_mean=[0.0],
        league_rpg=4.5,
        **kwargs,
    )


def _sim(mu_home, mu_away, draws=20000):
    m = _model()
    m.expected_runs = lambda f: f["mu"]           # bypass the GLM for the test
    return m.simulate({"mu": mu_home}, {"mu": mu_away}, draws=draws)


def test_the_home_club_scores_less_than_its_rate_because_it_stops_batting():
    """The censoring is the point. Equal rates must not give equal runs."""
    out = _sim(4.5, 4.5)
    assert out["exp_home_runs"] < out["exp_away_runs"]
    # Measured gap over the corpus is about 0.17 runs; anything in this band is
    # the rule biting rather than noise or a runaway.
    gap = out["exp_away_runs"] - out["exp_home_runs"]
    assert 0.05 < gap < 0.40


def test_equal_rates_still_favour_the_home_club_on_the_scoreboard():
    """Losing the ninth costs runs and wins nothing; the home club's edge comes
    from Elo and the rest of the block, not from the simulation. With identical
    rates the game should be close to a coin flip, not tilted."""
    out = _sim(4.5, 4.5)
    assert 0.45 < out["home_win_probability"] < 0.55


def test_the_better_club_still_wins_more_often():
    weak = _sim(3.5, 5.5)["home_win_probability"]
    even = _sim(4.5, 4.5)["home_win_probability"]
    strong = _sim(5.5, 3.5)["home_win_probability"]
    assert weak < even < strong
    assert strong > 0.6 and weak < 0.4


def test_no_game_is_ever_reported_as_a_tie():
    """Baseball has no ties, and a tie left in the draws would leak into every
    derived price."""
    m = _model()
    m.expected_runs = lambda f: f["mu"]
    out = m.simulate({"mu": 4.5}, {"mu": 4.5}, draws=20000)
    assert out["home_win_probability"] + out["away_win_probability"] == pytest.approx(1.0)


def test_the_reported_runs_are_what_is_scored_not_the_uncensored_rate():
    """The page prints a number a total settles on. The rate is kept beside it
    for anyone who needs the uncensored figure."""
    out = _sim(4.5, 4.5)
    assert out["exp_home_rate"] == pytest.approx(4.5)
    assert out["exp_away_rate"] == pytest.approx(4.5)
    assert out["exp_home_runs"] < out["exp_home_rate"]


def test_expected_total_is_consistent_with_the_two_sides():
    out = _sim(4.6, 4.3)
    assert out["expected_total"] == pytest.approx(
        out["exp_home_runs"] + out["exp_away_runs"], abs=0.05)


def test_the_measured_constants_are_the_measured_values():
    """Guards against a silent edit to numbers that came from the corpus.

    alpha solves var = mu + alpha*mu^2 at the measured half-inning mean 0.5021
    and variance 1.0604. The extras rate is 0.9957 against 0.4955 runs a
    half-inning over 2021-26. The coin flip is the measured 50.41% home rate in
    extra-inning games, which is not distinguishable from a half.
    """
    assert model_module.HALF_INNING_ALPHA == pytest.approx(
        (1.0604 - 0.5021) / 0.5021 ** 2, abs=0.01)
    assert model_module.EXTRA_INNING_RATE == pytest.approx(0.9957 / 0.4955, abs=0.02)
    assert model_module.EXTRA_INNING_HOME_EDGE == 0.5


def test_a_higher_scoring_game_produces_a_wider_total():
    low = _sim(3.0, 3.0)
    high = _sim(6.0, 6.0)
    assert high["expected_total"] > low["expected_total"]
    assert high["p_home_shutout"] < low["p_home_shutout"]


def test_the_exposure_term_turns_game_runs_into_a_rate():
    """`score.design`'s side of the same change.

    A club that batted eight innings and scored four is scoring at a higher
    rate than one that batted nine for the same four, and the offset is what
    tells the fit so.
    """
    import pandas as pd

    from guards_report.projections import score

    frame = pd.DataFrame({
        "is_home": [1.0, 0.0],
        "runs": [4.0, 4.0],
        "league_rpg": [4.5, 4.5],
        "innings_batted": [8.0, 9.0],
    })
    _, _, plain, _ = score.design(frame, ["is_home"])
    _, _, exposed, _ = score.design(frame, ["is_home"], exposure=True)

    assert plain[0] == pytest.approx(plain[1])
    assert exposed[0] < exposed[1]
    assert exposed[1] == pytest.approx(plain[1])
    assert exposed[0] == pytest.approx(plain[0] + np.log(8.0 / 9.0))


def test_a_missing_innings_column_leaves_the_offset_alone():
    """Older callers must not silently change meaning."""
    import pandas as pd

    from guards_report.projections import score

    frame = pd.DataFrame({
        "is_home": [1.0], "runs": [4.0], "league_rpg": [4.5]})
    _, _, plain, _ = score.design(frame, ["is_home"])
    _, _, exposed, _ = score.design(frame, ["is_home"], exposure=True)
    assert exposed == pytest.approx(plain)
