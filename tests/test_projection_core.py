"""Elo, walk-forward evaluation, and the first-five substrate.

These three carry the properties everything downstream assumes and none of them
can check. Elo is the incumbent every model has to beat, so a sign error in its
update makes every comparison meaningless while the numbers stay plausible.
`walk_forward` is the only thing standing between this project and training on
the future. And `starter_history` is a shifted expanding mean, which is the
single most common place a leak gets written by accident.

The tests are built around what a failure would look like from the outside,
which in every case here is: fine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from guards_report.projections import backtest, elo, first5


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------

def _games(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["home_win"] = (frame["home_runs"] > frame["away_runs"]).astype(int)
    return frame


def _game(home, away, hr, ar, season=2024) -> dict:
    return {
        "home_team_id": home, "away_team_id": away,
        "home_runs": hr, "away_runs": ar, "season": season,
    }


# ---------------------------------------------------------------------------
# Elo — the incumbent
# ---------------------------------------------------------------------------

def test_the_first_game_is_home_field_alone():
    """Two unrated clubs differ only by who is batting last.

    With both at the mean, the probability is entirely the HFA term, so this
    pins the sign and scale of `hfa` -- 24 rating points, not 24 percent.
    """
    frame = _games([_game(1, 2, 5, 3)])
    probability = elo.run(frame, elo.EloParams(hfa=24.0))[0]
    assert probability == pytest.approx(1 / (1 + 10 ** (-24 / 400)), abs=1e-9)
    assert 0.5 < probability < 0.55


def test_no_home_advantage_makes_two_fresh_clubs_a_coin_flip():
    frame = _games([_game(1, 2, 5, 3)])
    assert elo.run(frame, elo.EloParams(hfa=0.0))[0] == pytest.approx(0.5)


def test_winning_raises_your_rating_and_lowers_theirs_by_the_same_amount():
    """Elo is zero-sum. If it is not, the league's mean rating drifts.

    A drifting mean would show up as nothing in particular for a long time and
    then as a model that thinks every team is above average.
    """
    frame = _games([_game(1, 2, 9, 1), _game(3, 4, 1, 9)])
    state = elo.fit_state(frame, elo.EloParams())
    assert state.ratings[1] > elo.MEAN_RATING > state.ratings[2]
    assert state.ratings[4] > elo.MEAN_RATING > state.ratings[3]
    assert sum(state.ratings.values()) == pytest.approx(4 * elo.MEAN_RATING, abs=1e-9)


def test_a_bigger_margin_moves_the_rating_further():
    params = elo.EloParams(mov=True)
    narrow = elo.fit_state(_games([_game(1, 2, 4, 3)]), params).ratings[1]
    blowout = elo.fit_state(_games([_game(1, 2, 12, 0)]), params).ratings[1]
    assert blowout > narrow > elo.MEAN_RATING


def test_margin_of_victory_is_damped_rather_than_proportional():
    """A 12-run win is not four times the evidence of a 3-run win.

    Baseball scoring is heavy-tailed enough that an undamped margin lets one
    blowout outweigh a fortnight of results.
    """
    params = elo.EloParams(mov=True)
    gain = lambda margin: (  # noqa: E731
        elo.fit_state(_games([_game(1, 2, margin, 0)]), params).ratings[1]
        - elo.MEAN_RATING
    )
    assert gain(12) < 4 * gain(3)
    assert gain(12) > gain(3)


def test_the_margin_correction_discounts_a_favourite_beating_a_weak_side():
    """Without this, good teams inflate without bound.

    The same six-run margin says less when the winner was already expected to
    win, and the rating-difference denominator is what encodes that.
    """
    strong_over_weak = elo._mov_multiplier(6, rating_diff=400.0)
    between_equals = elo._mov_multiplier(6, rating_diff=0.0)
    assert strong_over_weak < between_equals


def test_a_new_season_regresses_ratings_toward_the_mean():
    """Carry is applied at the boundary, and it must pull toward 1500.

    A carry that pushed away from the mean would compound every year, and the
    first place it would show is a preseason favorite rated 1900.
    """
    history = _games([_game(1, 2, 9, 1, season=2023)] * 20)
    earned = elo.fit_state(history, elo.EloParams(carry=0.75)).ratings[1]

    with_next_season = pd.concat(
        [history, _games([_game(1, 2, 5, 4, season=2024)])], ignore_index=True
    )
    # The probability of that first 2024 game is computed from the carried
    # rating, so it is the observable that carry actually moves.
    full = elo.run(with_next_season, elo.EloParams(carry=0.75))[-1]
    none = elo.run(with_next_season, elo.EloParams(carry=0.0))[-1]
    assert earned > elo.MEAN_RATING
    assert full > none
    assert none == pytest.approx(
        1 / (1 + 10 ** (-24 / 400)), abs=1e-9
    ), "carry=0 must reset both clubs to the mean"


def test_every_probability_precedes_the_game_it_describes():
    """The property that makes Elo leak-free without any as-of bookkeeping.

    The first game of a corpus can only ever be rated from the prior, whatever
    happens in it. If a future result reached backwards, this is where it would
    show first.
    """
    rout = elo.run(_games([_game(1, 2, 20, 0), _game(1, 2, 5, 4)]), elo.EloParams())
    loss = elo.run(_games([_game(1, 2, 0, 20), _game(1, 2, 5, 4)]), elo.EloParams())
    assert rout[0] == pytest.approx(loss[0]), "game one saw its own result"
    assert rout[1] > loss[1], "game two ignored game one"


def test_an_unseen_club_is_predicted_at_the_prior_rather_than_refused():
    """September call-ups aside, this is the expansion and relocation case."""
    state = elo.fit_state(_games([_game(1, 2, 5, 3)]), elo.EloParams())
    assert state.probability(999, 998) == pytest.approx(
        1 / (1 + 10 ** (-24 / 400)), abs=1e-9
    )


def test_ratings_per_game_agrees_with_the_probability_the_same_corpus_produces():
    """Two functions walking the same recursion must not drift apart.

    `run` gives the probability, `ratings_per_game` gives the two ratings behind
    it, and the run model uses the second while the win model uses the first. If
    they disagreed the two halves of the report would describe different leagues.
    """
    frame = _games(
        [_game(1, 2, 5, 3), _game(2, 3, 1, 7), _game(3, 1, 4, 4 + 1)] * 4
    )
    params = elo.EloParams()
    probability = elo.run(frame, params)
    home, away = elo.ratings_per_game(frame, params)
    implied = 1 / (1 + 10 ** (-(home + params.hfa - away) / 400))
    assert np.allclose(probability, implied, atol=1e-9)


# ---------------------------------------------------------------------------
# Walk-forward — the leakage guard
# ---------------------------------------------------------------------------

def _seasons(*years: int) -> pd.DataFrame:
    return pd.DataFrame(
        [{"season": year, "i": i} for year in years for i in range(4)]
    )


def test_no_training_row_is_ever_from_the_test_season_or_later():
    """The whole point. Random k-fold trains on the future to predict the past.

    Asserted on every fold rather than the first, because an off-by-one that
    only bites on the last fold is exactly the kind that survives a spot check.
    """
    frame = _seasons(2018, 2019, 2020, 2021, 2022, 2023)
    folds = list(backtest.walk_forward(frame, min_train_seasons=3))
    assert folds, "no folds produced"
    for train, test, season in folds:
        assert train["season"].max() < season
        assert set(test["season"]) == {season}


def test_every_season_after_the_warm_up_is_tested_exactly_once():
    frame = _seasons(2018, 2019, 2020, 2021, 2022, 2023)
    tested = [season for _, _, season in backtest.walk_forward(frame, min_train_seasons=3)]
    assert tested == [2021, 2022, 2023]


def test_the_training_window_expands_rather_than_sliding():
    frame = _seasons(2018, 2019, 2020, 2021, 2022, 2023)
    sizes = [len(train) for train, _, _ in backtest.walk_forward(frame, min_train_seasons=3)]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)


def test_too_short_a_corpus_yields_no_folds_rather_than_a_thin_one():
    """Better to report nothing than a fold trained on one season."""
    assert list(backtest.walk_forward(_seasons(2018, 2019), min_train_seasons=3)) == []


def test_a_perfect_prediction_scores_zero_loss_and_a_coin_flip_scores_ln_two():
    truth = np.array([1, 0, 1, 0])
    assert backtest.evaluate(truth, np.array([1.0, 0.0, 1.0, 0.0])).log_loss < 1e-6
    assert backtest.evaluate(
        truth, np.full(4, 0.5)
    ).log_loss == pytest.approx(np.log(2), abs=1e-9)


def test_a_confident_wrong_prediction_does_not_produce_an_infinite_loss():
    """Clipping is the difference between a bad fold and an unusable summary.

    One over-confident miss would otherwise take the mean to infinity and
    destroy a whole season's evaluation.
    """
    loss = backtest.evaluate(np.array([1, 1]), np.array([0.0, 1.0])).log_loss
    assert np.isfinite(loss) and loss > 0


def test_the_summary_reports_spread_and_not_only_the_centre():
    """A mean that swings year to year has not established anything.

    Two model histories can share a mean log loss while one is stable and the
    other had a single good season, and only the spread separates them.
    """
    steady = {
        year: backtest.evaluate(np.array([1, 0]), np.array([0.6, 0.4]))
        for year in (2021, 2022, 2023)
    }
    swingy = dict(steady)
    swingy[2023] = backtest.evaluate(np.array([1, 0]), np.array([0.95, 0.05]))

    assert backtest.summarize(steady)["log_loss_sd"] == pytest.approx(0.0, abs=1e-9)
    assert backtest.summarize(swingy)["log_loss_sd"] > 0


# ---------------------------------------------------------------------------
# First five — the shifted expanding mean
# ---------------------------------------------------------------------------

def _pitches(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    return frame


def _start(game_pk, date, pitcher, runs, batters=18, season=2026,
           batting_team="OPP") -> list[dict]:
    """One starter's five innings: `batters` plate appearances, `runs` allowed."""
    return [
        {
            "game_pk": game_pk, "game_date": date, "season": season,
            "inning": 1 + (i % 5), "pitcher": pitcher, "batting_team": batting_team,
            "events": "strikeout", "bat_score": 0, "post_bat_score": runs,
        }
        for i in range(batters)
    ]


def test_a_starters_first_start_carries_no_history():
    """He has no prior starts, so there is nothing true to say about him.

    The model reads this absence through `f5_line_known` rather than through a
    league average dressed up as his number.
    """
    history = first5.starter_history(_pitches(_start(1, "2026-04-01", 7, runs=3)))
    assert len(history) == 1
    assert pd.isna(history.iloc[0]["sp_f5_ra"])
    assert history.iloc[0]["sp_f5_starts"] == 0


def test_each_row_carries_only_the_starts_before_it():
    """Row i must be built from starts 0..i-1 and never include start i.

    This is the leak that reads as a good model: including the current start
    makes the feature partly the answer, and the backtest rewards it. Six
    scoreless starts then one where he is hit hard -- if the seventh row moves,
    it saw its own result.
    """
    pitch = _pitches(
        sum(
            (_start(i, f"2026-04-{i:02d}", 7, runs=0) for i in range(1, 7)), []
        )
        + _start(7, "2026-04-07", 7, runs=6)
        + _start(8, "2026-04-08", 7, runs=0)
    )
    history = first5.starter_history(pitch).sort_values("game_pk")
    assert list(history["sp_f5_starts"]) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert history.iloc[6]["sp_f5_ra"] == pytest.approx(0.0), "start seven saw its own six runs"
    # Six scoreless then a six-run start: 6/7 going into the eighth.
    assert history.iloc[7]["sp_f5_ra"] == pytest.approx(6 / 7)


def test_a_thin_record_is_withheld_rather_than_reported_noisily():
    """One prior start is a number, not a rate."""
    pitch = _pitches(sum(
        (_start(i, f"2026-04-{i:02d}", 7, runs=3) for i in range(1, 9)), []
    ))
    history = first5.starter_history(pitch).sort_values("game_pk")
    thin = history[history["sp_f5_starts"] < first5.MIN_PRIOR_STARTS]
    settled = history[history["sp_f5_starts"] >= first5.MIN_PRIOR_STARTS]
    assert len(thin) and thin["sp_f5_ra"].isna().all()
    assert len(settled) and settled["sp_f5_ra"].notna().all()


def test_history_is_kept_per_starter_and_not_pooled():
    """Two starters in the same games, facing opposite sides, must not blend."""
    rows = []
    for game in range(1, 8):
        date = f"2026-04-{game:02d}"
        rows += _start(game, date, 7, runs=0, batting_team="AWAY")
        rows += _start(game, date, 9, runs=8, batting_team="HOME")
    history = first5.starter_history(_pitches(rows))

    good = history[(history["starter"] == 7) & (history["game_pk"] == 7)].iloc[0]
    poor = history[(history["starter"] == 9) & (history["game_pk"] == 7)].iloc[0]
    assert good["sp_f5_ra"] == pytest.approx(0.0)
    assert poor["sp_f5_ra"] == pytest.approx(8.0)


def test_the_table_carries_the_date_each_row_belongs_to():
    """Without it the live lookup cannot ask for a line as of a given day, and
    the refresh cannot tell whether the table has fallen behind its corpus."""
    history = first5.starter_history(_pitches(_start(1, "2026-04-01", 7, runs=3)))
    assert "game_date" in history.columns
    assert pd.to_datetime(history["game_date"]).max().date().isoformat() == "2026-04-01"


def test_attaching_the_line_does_not_collide_with_the_frames_own_date():
    """A merge that produced `game_date_x` would break every later reader."""
    history = first5.starter_history(_pitches(_start(1, "2026-04-01", 7, runs=3)))
    data = pd.DataFrame({
        "game_pk": [1], "game_date": pd.to_datetime(["2026-04-01"]),
        "opp_starter_id": [7],
    })
    joined = first5.add_features(data, history)
    assert "game_date" in joined.columns
    assert not any(column.endswith(("_x", "_y")) for column in joined.columns)


def test_an_unknown_starter_is_flagged_rather_than_filled():
    history = first5.starter_history(_pitches(_start(1, "2026-04-01", 7, runs=3)))
    data = pd.DataFrame({
        "game_pk": [1], "game_date": pd.to_datetime(["2026-04-01"]),
        "opp_starter_id": [404],
    })
    joined = first5.add_features(data, history)
    assert joined.iloc[0]["f5_line_known"] == 0
    assert pd.isna(joined.iloc[0]["opp_f5_ra"])


def test_a_first_five_result_can_be_level():
    """Collapsing ties into one side or the other misstates a real outcome.

    A full game has none -- extra innings are played until somebody leads -- but
    through five roughly one game in six is level, and a three-way model that
    reports two ways is wrong about all of them.
    """
    home = np.full(4000, 2.3)
    away = np.full(4000, 2.3)
    result = first5.outcome_probabilities(home, away, draws=4000, seed=3)
    total = result["home_leads"] + result["away_leads"] + result["tied"]
    assert np.allclose(total, 1.0, atol=1e-9)
    assert (result["tied"] > 0.05).all(), "ties were resolved away"


def test_the_stronger_side_is_favoured_through_five():
    result = first5.outcome_probabilities(
        np.array([3.4]), np.array([1.6]), draws=8000, seed=5
    )
    assert result["home_leads"][0] > result["away_leads"][0]
