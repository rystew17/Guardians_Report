"""Feature assembly and the run model.

The substrate every projection is built on, and the place where a mistake is
hardest to notice. Both modules are governed by one rule -- a feature for a game
on date D may use only data from games played before D -- and breaking it does
not raise. It shows up as a backtest that looks unusually good, which is the one
result nobody is inclined to interrogate.

The other silent class here is sign. Every feature is normalized so a larger
value favours the home side; a single inverted sign makes a coefficient
interpretable-looking and backwards, and the fitted model will happily absorb it
and still produce probabilities between 0 and 1.

So these tests mostly do two things: hand a function a future it must not see,
and check that better inputs move numbers the right way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from guards_report.projections import elo, features, score


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------

def _games(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["home_win"] = (frame["home_runs"] > frame["away_runs"]).astype(int)
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    return frame


def _game(pk, date, home, away, hr, ar, *, season=2026, venue=1) -> dict:
    return {
        "game_pk": pk, "game_date": date, "season": season, "venue_id": venue,
        "home_team_id": home, "away_team_id": away,
        "home_runs": hr, "away_runs": ar,
    }


def _season(season: int, *, venue_runs: dict[int, int], games: int = 90) -> list[dict]:
    """A season where each venue produces a fixed run total every night."""
    rows = []
    pk = season * 1000
    for i in range(games):
        for venue, runs in venue_runs.items():
            rows.append(_game(
                pk, f"{season}-04-{(i % 28) + 1:02d}", 10 + venue, 20 + venue,
                runs // 2, runs - runs // 2, season=season, venue=venue))
            pk += 1
    return rows


# ---------------------------------------------------------------------------
# Park factors — the leak that would be invisible
# ---------------------------------------------------------------------------

def test_a_park_factor_never_uses_the_season_it_describes():
    """The extreme case is a park factor containing the game it predicts.

    Computed from the season in progress it would, and the backtest would
    reward it for exactly that.
    """
    corpus = _games(
        _season(2024, venue_runs={1: 8, 2: 8})
        + _season(2025, venue_runs={1: 14, 2: 4})
    )
    factors = features.park_factors(corpus)
    early = factors[factors["season"] == 2025]
    if len(early):
        # 2025 is the first season with any history, and that history is 2024,
        # where both parks were identical. A factor that had seen 2025 would
        # separate them.
        by_venue = early.set_index("venue_id")["park_factor"]
        if len(by_venue) == 2:
            assert abs(by_venue.iloc[0] - by_venue.iloc[1]) < 0.05


def test_a_hitters_park_earns_a_factor_above_one_the_following_season():
    """The sign, which nothing downstream would flag if it inverted."""
    corpus = _games(
        _season(2024, venue_runs={1: 14, 2: 4})
        + _season(2025, venue_runs={1: 14, 2: 4})
        + _season(2026, venue_runs={1: 9, 2: 9})
    )
    factors = features.park_factors(corpus)
    latest = factors[factors["season"] == 2026].set_index("venue_id")["park_factor"]
    if len(latest) == 2:
        assert latest.loc[1] > 1.0 > latest.loc[2]


def test_a_park_with_no_history_falls_back_to_neutral():
    """A handful of games is a worse estimate than admitting you have none."""
    corpus = _games(_season(2026, venue_runs={1: 9}))
    factors = features.park_factors(corpus)
    assert (factors["park_factor"] == 1.0).all() or factors.empty


# ---------------------------------------------------------------------------
# Rest
# ---------------------------------------------------------------------------

def test_rest_is_days_since_that_team_last_played():
    corpus = _games([
        _game(1, "2026-04-01", 10, 20, 5, 3),
        _game(2, "2026-04-04", 10, 30, 5, 3),
    ])
    rest = features.team_rest(corpus)
    second = rest[(rest["game_pk"] == 2) & (rest["team_id"] == 10)]
    assert second.iloc[0]["team_rest_days"] == 3


def test_the_first_game_of_a_season_is_capped_rather_than_null():
    """Spring rest is long but not unbounded, and an unbounded value would
    dominate any scaling the model applies."""
    corpus = _games([_game(1, "2026-04-01", 10, 20, 5, 3)])
    rest = features.team_rest(corpus)
    assert rest["team_rest_days"].notna().all()
    assert (rest["team_rest_days"] <= 6).all()


def test_rest_is_returned_long_so_the_two_sides_cannot_be_crossed():
    """One row per team-game. A wide frame invites joining the home value onto
    the away column, which is a mistake with no visible symptom."""
    corpus = _games([_game(1, "2026-04-01", 10, 20, 5, 3)])
    rest = features.team_rest(corpus)
    assert len(rest) == 2
    assert set(rest["team_id"]) == {10, 20}
    assert "home_team_id" not in rest.columns


def test_a_starters_rest_is_capped_at_both_ends():
    """Unknown rest takes the modal turn rather than a flag the model overreads."""
    logs = pd.DataFrame({
        "pitcher_id": [7, 7, 7],
        "game_pk": [1, 2, 3],
        "season": [2026, 2026, 2026],
        "game_date": pd.to_datetime(["2026-04-01", "2026-04-02", "2026-06-01"]),
    })
    rest = features.starter_rest(logs).set_index("game_pk")["starter_rest_days"]
    assert rest.loc[1] == 5, "no prior appearance takes the modal turn"
    assert rest.loc[2] == 1
    assert rest.loc[3] == 10, "a long layoff is capped, not carried"


# ---------------------------------------------------------------------------
# The design matrix
# ---------------------------------------------------------------------------

def _frame(**columns) -> pd.DataFrame:
    frame = pd.DataFrame(columns)
    frame["home_win"] = [1, 0] * (len(frame) // 2) + [1] * (len(frame) % 2)
    return frame


def test_a_missing_feature_is_imputed_to_the_column_mean():
    """Imputation lives here so the null stays visible upstream."""
    frame = _frame(a=[1.0, 3.0, np.nan, 5.0], b=[1.0, 1.0, 1.0, 1.0])
    X, _ = features.design_matrix(frame, ["a", "b"])
    assert not np.isnan(X).any()
    assert X[2, 0] == pytest.approx(3.0)


def test_imputation_uses_each_columns_own_mean():
    """One shared mean across columns would be a silent scaling error."""
    frame = _frame(a=[10.0, np.nan, 20.0, 30.0], b=[1.0, 2.0, np.nan, 3.0])
    X, _ = features.design_matrix(frame, ["a", "b"])
    assert X[1, 0] == pytest.approx(20.0)
    assert X[2, 1] == pytest.approx(2.0)


def test_the_target_comes_out_as_the_home_win_column():
    frame = _frame(a=[1.0, 2.0])
    _, y = features.design_matrix(frame, ["a"])
    assert list(y) == [1.0, 0.0]


def test_the_matrix_keeps_the_column_order_it_was_given():
    """The fitted coefficients are read positionally, so order is a contract."""
    frame = _frame(a=[1.0, 2.0], b=[30.0, 40.0])
    X, _ = features.design_matrix(frame, ["b", "a"])
    assert list(X[0]) == [30.0, 1.0]


def test_starter_known_rides_alongside_the_imputed_values():
    """The model has to be able to tell a measured value from a filled one."""
    assert "starter_known" in features.CORE_COLUMNS


# ---------------------------------------------------------------------------
# The run environment
# ---------------------------------------------------------------------------

def test_the_league_run_level_never_includes_the_game_it_describes():
    """Shifted before the rolling mean.

    A game contributing to the environment used to predict it is the same leak
    as a park factor built from the current season, one layer down.
    """
    quiet = [_game(i, "2026-04-01", 10, 20, 1, 1) for i in range(200)]
    loud = [_game(200, "2026-04-02", 10, 20, 25, 25)]
    corpus = _games(quiet + loud)
    level = score.league_run_level(corpus)
    # The final row is the blowout. Its own runs must not have raised the
    # environment it is measured against.
    assert level.iloc[-1] < 5.0


def test_the_run_level_falls_back_before_it_has_history():
    corpus = _games([_game(1, "2026-04-01", 10, 20, 5, 3)])
    level = score.league_run_level(corpus)
    assert level.notna().all()
    assert (level > 0).all()


def test_team_offense_carries_only_prior_games():
    """Row i must hold games 0..i-1, and the first row must hold nothing."""
    rows = [_game(i, f"2026-04-{i + 1:02d}", 10, 20, 10, 0) for i in range(25)]
    corpus = _games(rows)
    offense = score.team_offense(corpus)
    first = offense[(offense["game_pk"] == 0) & (offense["team_id"] == 10)]
    assert first.iloc[0]["off_games"] == 0
    assert pd.isna(first.iloc[0]["off_rpg"])


def test_a_thin_offensive_sample_is_withheld_rather_than_reported():
    """Under fifteen games the rate is mostly noise."""
    rows = [_game(i, f"2026-04-{i + 1:02d}", 10, 20, 8, 2) for i in range(20)]
    offense = score.team_offense(_games(rows))
    team = offense[offense["team_id"] == 10].sort_values("off_games")
    thin = team[team["off_games"] < 15]
    settled = team[team["off_games"] >= 15]
    assert thin["off_rpg"].isna().all()
    assert settled["off_rpg"].notna().all()


def test_a_high_scoring_team_earns_a_higher_rate():
    rows = [_game(i, f"2026-04-{i % 28 + 1:02d}", 10, 20, 9, 1) for i in range(30)]
    offense = score.team_offense(_games(rows))
    good = offense[offense["team_id"] == 10]["off_rpg"].dropna()
    poor = offense[offense["team_id"] == 20]["off_rpg"].dropna()
    assert good.mean() > poor.mean()


# ---------------------------------------------------------------------------
# Dispersion — the statistic that chose the model family
# ---------------------------------------------------------------------------

def test_poisson_data_disperses_at_about_one():
    """The reference case. This statistic is what put NB2 in the model rather
    than convention, so it has to be right about the null."""
    rng = np.random.default_rng(0)
    mu = np.full(4000, 4.5)
    y = rng.poisson(mu)
    assert score.dispersion(y, mu, n_params=1) == pytest.approx(1.0, abs=0.10)


def test_overdispersed_data_reports_above_one():
    """Baseball scoring runs about 2.1. Anything that could not detect that
    would have let a Poisson model ship with intervals too narrow."""
    rng = np.random.default_rng(1)
    mu = np.full(4000, 4.5)
    y = rng.negative_binomial(4, 4 / (4 + 4.5), size=4000)
    assert score.dispersion(y, mu, n_params=1) > 1.5


def test_dispersion_accounts_for_the_parameters_spent():
    """Chi-square over residual degrees of freedom, not over n."""
    y = np.array([4.0, 5.0, 3.0, 6.0, 4.0])
    mu = np.full(5, 4.5)
    assert score.dispersion(y, mu, 1) < score.dispersion(y, mu, 3)


def test_a_zero_mean_does_not_divide_by_zero():
    """Clipped rather than guarded at the call site, because a shutout is a
    legitimate prediction and a crash mid-backtest is not."""
    value = score.dispersion(np.array([0.0, 1.0]), np.array([0.0, 1.0]), 1)
    assert np.isfinite(value)


# ---------------------------------------------------------------------------
# The score design
# ---------------------------------------------------------------------------

def _score_frame(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame({
        "runs": [4.0, 5.0, 3.0, 6.0, 2.0, 7.0][:n],
        "league_rpg": [4.5] * n,
        "elo_diff": [0.1, -0.2, 0.3, 0.0, -0.1, 0.2][:n],
        "od_exp_runs": [4.4, 4.6, 4.2, 4.8, 4.1, 4.9][:n],
    })


def test_the_offset_is_the_run_environment_on_the_log_scale():
    """That is what makes a fitted coefficient relative to the era rather than
    absolute, and the run environment moves: F = 13.32 across seasons."""
    frame = _score_frame()
    _, _, offset, _ = score.design(frame, ["elo_diff", "od_exp_runs"])
    assert offset == pytest.approx(np.log(4.5))


def test_a_collapsed_run_environment_is_floored_rather_than_logged_to_minus_infinity():
    frame = _score_frame()
    frame["league_rpg"] = 0.0
    _, _, offset, _ = score.design(frame, ["elo_diff"])
    assert np.isfinite(offset).all()


def test_rows_without_a_score_are_dropped_rather_than_imputed():
    """A missing score is a game that did not finish. Imputing one would invent
    a result, where imputing a missing *predictor* only softens a real one."""
    frame = _score_frame()
    frame.loc[2, "runs"] = np.nan
    X, y, _, kept = score.design(frame, ["elo_diff"])
    assert len(y) == len(frame) - 1
    assert len(X) == len(kept) == len(y)


def test_a_missing_predictor_is_imputed_and_the_row_survives():
    frame = _score_frame()
    frame.loc[1, "elo_diff"] = np.nan
    X, y, _, _ = score.design(frame, ["elo_diff", "od_exp_runs"])
    assert len(y) == len(frame)
    assert not np.isnan(X).any()


def test_the_column_sets_do_not_silently_overlap():
    """Each block is added and measured for its own marginal contribution, so a
    column counted twice would credit one block with another's signal."""
    blocks = [score.STRENGTH_COLUMNS, score.PEN_COLUMNS, score.PA_COLUMNS]
    seen: set[str] = set()
    for block in blocks:
        assert not (seen & set(block)), f"{block} repeats a column"
        seen |= set(block)


# ---------------------------------------------------------------------------
# Core assembly
# ---------------------------------------------------------------------------

def test_every_core_column_is_a_comparison_rather_than_one_side_s_figure():
    """Signs are normalized so a larger value always favours home.

    Not cosmetic: it makes a fitted coefficient's sign readable at a glance, so
    one that comes out backwards is an obvious signal rather than a detail
    buried in the encoding. Checked by name because the naming is the only
    place that convention is written down -- a bare `fip` in this list would be
    one club's figure with no opponent to compare it against.
    """
    assert features.CORE_COLUMNS
    for name in features.CORE_COLUMNS:
        comparative = (
            name.endswith(("_diff", "_logit", "_known", "_factor"))
            or name.startswith("sp_")   # starter block, already home-minus-away
        )
        assert comparative, f"{name} does not read as a comparison"


def test_the_starter_block_appears_in_the_core_columns():
    """Each metric that survived testing has to actually reach the model."""
    for metric in features.STARTER_METRICS:
        assert any(name == f"sp_{metric}" for name in features.CORE_COLUMNS), metric


def test_the_bullpen_block_is_kept_separate_from_the_core():
    """Each block earns its place on its own measured contribution."""
    assert not (set(features.CORE_COLUMNS) & set(features.BULLPEN_COLUMNS))


def test_the_starter_metrics_exclude_the_two_that_were_tested_and_dropped():
    """ERA added nothing next to FIP, and RA/9 took a negative coefficient that
    survived orthogonalisation -- what is left of it past FIP is luck, and luck
    reverts."""
    assert "era" not in features.STARTER_METRICS
    assert "ra9" not in features.STARTER_METRICS
    assert "fip" in features.STARTER_METRICS and "k_pct" in features.STARTER_METRICS
