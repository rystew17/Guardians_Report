"""Model B is graded out of sample, and the grading cannot see the future.

The score model went to production for months with no held-out metric at all.
These pin the one that now exists: each season is scored by a fit that never saw
it, against baselines it has to beat to be worth its inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from guards_report.projections import train


def _data(seasons=(2020, 2021, 2022, 2023), per_season=800, seed=7):
    """Runs driven by one real predictor, so a fitted model should beat both
    baselines by construction."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        signal = rng.normal(0.0, 1.0, per_season)
        mu = 4.5 * np.exp(0.25 * signal)
        runs = rng.poisson(mu)
        rows.append(pd.DataFrame({
            "season": season,
            "signal": signal,
            "runs": runs.astype(float),
            "league_rpg": 4.5,
            # Own form that knows nothing: a constant, so it cannot beat a
            # model with access to the signal.
            "off_rpg": 4.5,
        }))
    return pd.concat(rows, ignore_index=True)


def test_every_held_out_season_is_scored_and_none_before_the_first():
    result = train._score_backtest(_data(), ["signal"], first_test=2022)
    assert set(result["per_season"]) == {"2022", "2023"}


def test_a_model_with_a_real_predictor_beats_both_baselines():
    result = train._score_backtest(_data(), ["signal"], first_test=2022)
    assert result["mae"] < result["mae_league"]
    assert result["mae"] < result["mae_own_form"]
    assert abs(result["bias_runs"]) < 0.2


def test_a_held_out_season_is_scored_by_a_fit_that_never_saw_it():
    """Poison the test season's relationship. A fit that leaked the season it
    scores would learn the poisoned slope and score it well; one that did not
    is fitted on the clean seasons and scores it badly."""
    clean = _data(seasons=(2020, 2021, 2022))
    poisoned = _data(seasons=(2023,), seed=11)
    poisoned["runs"] = np.round(4.5 * np.exp(-0.9 * poisoned["signal"]))
    data = pd.concat([clean, poisoned], ignore_index=True)

    result = train._score_backtest(data, ["signal"], first_test=2022)
    assert (result["per_season"]["2023"]["mae"]
            > result["per_season"]["2022"]["mae"] + 0.5)


def test_nothing_to_score_returns_an_empty_record_rather_than_raising():
    assert train._score_backtest(_data(seasons=(2020,)), ["signal"], 2022) == {}
