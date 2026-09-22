"""Fit both outcome models on the full corpus and persist the artifact.

Run occasionally -- after a season ends, or when the feature set changes -- not
as part of building a report. A report loads the stored fit; it does not
re-estimate anything, so two reports of the same game always agree.

The metrics recorded here come from the held-out seasons, not from the final
full-corpus fit. A model's reported accuracy should be the one it earned on data
it had not seen, even though the coefficients it ships with were fitted on
everything available.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from guards_report.projections import (
    backtest, corpus, elo, features, lineup, model, pa, pitchers, ratings,
    score, talent, weather,
)

ELO_PARAMS = elo.EloParams(k=4, hfa=24, carry=0.70, mov=True)
OFF_DEF_PARAMS = ratings.OffDefParams(0.010, 0.010, 0.16, 0.70, 0.40)

# Regularization chosen by sweep on 2018-2021; the curve is flat above 0.03.
LOGISTIC_C = 3.0

# Width of the disjoint-era fits used to measure how uncertain this model is
# about itself. Four years is a compromise: short enough that three fit inside
# the corpus and the eras genuinely differ, long enough that each one has the
# thousands of games a nine-feature logistic needs to be stable.
BLOCK_YEARS = 4
MIN_BLOCK_GAMES = 2000

# Ridge penalty for the plate-appearance talent model, chosen forward in time
# on 2015-2021 validating against 2022 and confirmed independently on 2023,
# which selects the same value.
TALENT_ALPHA = 800

FIRST_TEST_SEASON = 2022


def _add_pa_block(data, games, plate, prior):
    """Attach opposing-starter talent and own-lineup value to each team-game.

    Built per season from a prior fitted on the seasons before it, then carried
    forward through that season's plate appearances. `as_of_for_games` joins on
    state standing strictly before first pitch, so a start can never contribute
    to the feature predicting it.

    Historical lineups are the real starting nine, reconstructed from the corpus.
    Only the first nine distinct batters count -- a pinch-hitter appears as a
    consequence of how the game went and would not be on a card three hours
    before.
    """
    import numpy as np
    import pandas as pd

    slots_by_season = lineup.slot_weights_by_season(plate)
    sp_rows, lu_rows, proj_rows = [], [], []

    def value_lineups(starters, batters, weights):
        """Slot-weighted mean batter effect per team-game.

        Shared by the posted card and the projected one so the two columns
        cannot drift apart in how they are computed -- only in which nine they
        are computed over, which is the whole point of having both.
        """
        merged = starters.merge(
            batters[["batter", "_key", "score"]], on="batter", how="left"
        )
        merged = merged[pd.to_datetime(merged["game_date"]) > merged["_key"]]
        merged = merged.sort_values("_key").groupby(
            ["game_pk", "batting_team", "batter", "slot"], as_index=False
        ).last()
        merged["w"] = merged["slot"].map(
            lambda n: weights[n - 1] if 1 <= n <= 9 else weights[-1]
        )
        merged["score"] = merged["score"].fillna(0.0)
        return (
            merged.groupby(["game_pk", "batting_team"], as_index=False)
            .apply(lambda b: pd.Series({
                "lineup_value": float((b.score * b.w).sum() / b.w.sum())
                if b.w.sum() else 0.0,
            }), include_groups=False)
        )

    for season in sorted(plate["season"].unique()):
        before = plate[plate["season"] < season]
        if not len(before):
            continue
        season_prior = talent.fit(before, alpha=TALENT_ALPHA)
        current = plate[plate["season"] == season]
        season_games = games[games["season"] == season]
        if not len(season_games):
            continue

        # -- opposing starter -------------------------------------------------
        running = talent.running_scores(season_prior, current, side="pitcher")
        block = season_games[["game_pk"]].copy()
        for side in ("home", "away"):
            joined = talent.as_of_for_games(
                season_games[["game_pk", "game_date", f"{side}_starter_id"]]
                .rename(columns={f"{side}_starter_id": "pitcher"}),
                running, side="pitcher",
            ).set_index("game_pk")
            block[f"{side}_sp_talent"] = joined["score"].reindex(block.game_pk).to_numpy()
        sp_rows.append(block)

        # -- own lineup -------------------------------------------------------
        batters = talent.running_scores(season_prior, current, side="batter")
        batters["_key"] = pd.to_datetime(batters["game_date"])
        starters = lineup.starting_lineups(current)
        weights = slots_by_season.get(int(season), lineup.DEFAULT_SLOT_PA)

        lu_rows.append(value_lineups(starters, batters, weights))

        # The same valuation over the nine a morning build would have GUESSED,
        # which is what the win model is fitted on. The score model keeps the
        # posted card it was built and validated against; changing that is a
        # separate question with its own evidence, not a side effect of this.
        proj_rows.append(value_lineups(
            lineup.projected_lineups(current), batters, weights))

    if not sp_rows:
        for column in score.PA_COLUMNS:
            data[column] = np.nan
        return data, pd.DataFrame(columns=["game_pk", "home_lineup", "away_lineup"])

    sp = pd.concat(sp_rows, ignore_index=True)
    lu = pd.concat(lu_rows, ignore_index=True)
    proj = pd.concat(proj_rows, ignore_index=True) if proj_rows else lu.iloc[0:0]

    data = data.merge(sp, on="game_pk", how="left")
    data["opp_sp_talent"] = np.where(
        data["is_home"] == 1, data["away_sp_talent"], data["home_sp_talent"]
    )
    # The score dataset keys on numeric team id while lineups key on the club
    # abbreviation. Pivoting to home and away sidesteps the mapping entirely and
    # matches how the starter column above is handled.
    sides = games[["game_pk", "home_team", "away_team"]]

    def pivot(frame):
        home = frame.rename(
            columns={"batting_team": "home_team", "lineup_value": "home_lineup"})
        away = frame.rename(
            columns={"batting_team": "away_team", "lineup_value": "away_lineup"})
        return (
            sides.merge(home, on=["game_pk", "home_team"], how="left")
                 .merge(away, on=["game_pk", "away_team"], how="left")
        )[["game_pk", "home_lineup", "away_lineup"]]

    wide = pivot(lu)

    data = data.merge(wide, on="game_pk", how="left")
    data["own_lineup"] = np.where(
        data["is_home"] == 1, data["home_lineup"], data["away_lineup"]
    )
    # Per game rather than per team-game, which is the shape the win model's
    # frame is in. Returned rather than recomputed because the talent fit behind
    # it is the most expensive step in a refit. Built from the PROJECTED nine,
    # so the coefficient is fitted at the noise level the build serves at.
    return data, pivot(proj) if len(proj) else wide.iloc[0:0]


def _win_covariance(
    x_scaled: np.ndarray, coef: np.ndarray, intercept: float,
) -> list[list[float]]:
    """Sandwich covariance of the fitted logistic coefficients.

    A penalized fit is biased, so the textbook `(X'WX)^-1` is the wrong matrix:
    it describes an estimator we did not use. The sandwich accounts for the
    ridge by putting the penalized Hessian on the outside and the unpenalized
    information in the middle:

        H     = X'WX + P          P = diag(0, 1/C, ..., 1/C)
        Cov   = H^-1 (X'WX) H^-1

    The leading zero in P is because scikit-learn does not penalize the
    intercept, so the intercept column is prepended and left alone. W is
    diag(p(1-p)) at the fitted probabilities.

    Returns a plain nested list; this goes straight into the JSON artifact.
    """
    n, k = x_scaled.shape
    augmented = np.column_stack([np.ones(n), x_scaled])
    eta = intercept + x_scaled @ coef
    p = 1.0 / (1.0 + np.exp(-eta))
    w = p * (1.0 - p)

    information = augmented.T @ (augmented * w[:, None])
    penalty = np.zeros((k + 1, k + 1))
    penalty[1:, 1:] = np.eye(k) / LOGISTIC_C

    try:
        bread = np.linalg.inv(information + penalty)
    except np.linalg.LinAlgError:
        # Singular only if a feature is constant or perfectly collinear, which
        # would be a data problem worth seeing rather than papering over -- but
        # it must not take the whole training run down with it.
        return []
    cov = bread @ information @ bread
    return [[float(v) for v in row] for row in cov]


def _innings_batted(root: Path, data):
    """How many innings each club actually batted, per team-game.

    The exposure the score model is fitted against. Read from the pitch corpus
    rather than assumed to be nine, because the home club's last turn is
    skipped whenever it is already ahead -- which is half of all nine-inning
    games.
    """
    import pandas as pd

    columns = ["game_pk", "season", "inning", "inning_topbot", "events"]
    pitch = pa.cached_pitch_frame(
        root, columns, "innings_batted_cache.parquet", ended_only=True)
    if pitch is None or not len(pitch):
        data["innings_batted"] = 9.0
        return data

    last = (pitch.groupby(["game_pk", "inning_topbot"])["inning"]
            .max().unstack(fill_value=0))
    last.columns = [str(c) for c in last.columns]
    if "Top" not in last.columns or "Bot" not in last.columns:
        data["innings_batted"] = 9.0
        return data

    sides = []
    for is_home, column in ((1, "Bot"), (0, "Top")):
        block = last[[column]].rename(
            columns={column: "innings_batted"}).reset_index()
        block["is_home"] = is_home
        sides.append(block)

    data = data.merge(pd.concat(sides, ignore_index=True),
                      on=["game_pk", "is_home"], how="left")
    # A zero means the corpus has no pitches for that side, not a side that
    # never batted; nine is the neutral exposure and leaves the row unchanged.
    data["innings_batted"] = (
        data["innings_batted"].replace(0, np.nan).fillna(9.0))
    return data


def _score_backtest(data, columns, first_test: int) -> dict:
    """Walk-forward scores for Model B, one season held out at a time.

    Until this existed the artifact carried three numbers for the score model
    and none of them compared a prediction with an outcome. Totals, run lines
    and first-five prices all come from this model.

    Per team-game, against two baselines it has to beat to be worth anything:
    the run environment alone (the offset with no predictors) and the club's
    own season-to-date scoring. Log-likelihood is the NB2 density at the
    observed runs, so it grades the whole distribution rather than the centre.
    """
    from scipy import stats as sp_stats

    per_season: dict[str, dict] = {}
    for season in range(first_test, int(data["season"].max()) + 1):
        train_rows = data[data["season"] < season]
        test_rows = data[data["season"] == season]
        if not len(train_rows) or not len(test_rows):
            continue
        x_tr, y_tr, off_tr, _ = score.design(train_rows, columns, exposure=True)
        fitted = sm.GLM(
            y_tr, x_tr,
            family=sm.families.NegativeBinomial(alpha=model.NB_ALPHA),
            offset=off_tr,
        ).fit()
        # Scored WITH the same exposure the fit used. The model now predicts a
        # rate per nine innings batted, so grading it against runs scored in a
        # game the home club often leaves early would charge it for the rule
        # rather than for its own error. Innings batted are known for a game
        # already played, so the honest question here is: given that this club
        # batted these innings, were its runs predicted well?
        #
        # How the rate turns into a price -- where the censoring has to be
        # generated rather than conditioned on -- is what `model.simulate`
        # answers, and it is judged on the derived markets instead.
        x_te, y_te, off_te, kept = score.design(test_rows, columns, exposure=True)
        mu = np.asarray(fitted.predict(x_te, offset=off_te), dtype=float)

        # Both baselines carry the same exposure as the model, or the contest
        # is decided by which side is quoted per game and which per inning.
        # `off_te` already includes it; the club's own form is per game, so it
        # is scaled the same way.
        league = np.exp(off_te)
        share = np.clip(
            kept["innings_batted"].to_numpy(dtype=float) / 9.0, 0.2, None
        ) if "innings_batted" in kept.columns else 1.0
        own = kept["off_rpg"].to_numpy(dtype=float) * share
        own = np.where(np.isnan(own), league, own)

        n = 1.0 / model.NB_ALPHA
        loglik = sp_stats.nbinom.logpmf(y_te.astype(int), n, n / (n + mu))

        per_season[str(season)] = {
            "n_team_games": int(len(y_te)),
            "mae": float(np.mean(np.abs(y_te - mu))),
            # Root mean squared miss: how far one club's runs in one game land
            # from any projection of them. The report quotes it so the size of
            # the irreducible scatter is a measurement, not a figure of speech.
            "rmse": float(np.sqrt(np.mean((y_te - mu) ** 2))),
            "mae_league": float(np.mean(np.abs(y_te - league))),
            "mae_own_form": float(np.mean(np.abs(y_te - own))),
            "loglik": float(np.mean(loglik)),
            "mean_predicted": float(mu.mean()),
            "mean_actual": float(y_te.mean()),
        }

    if not per_season:
        return {}
    weights = np.array([m["n_team_games"] for m in per_season.values()], float)

    def pooled(key):
        return float(np.average(
            [m[key] for m in per_season.values()], weights=weights))

    return {
        "per_season": per_season,
        "mae": pooled("mae"),
        "rmse": float(np.sqrt(np.average(
            [m["rmse"] ** 2 for m in per_season.values()], weights=weights))),
        "mae_league": pooled("mae_league"),
        "mae_own_form": pooled("mae_own_form"),
        "loglik": pooled("loglik"),
        "bias_runs": pooled("mean_predicted") - pooled("mean_actual"),
    }


def fit(
    *, corpus_dir: Path, pitcher_dir: Path, pa_dir: Path, seasons: range,
    verbose: bool = True,
) -> model.OutcomeModel:
    games = corpus.build(seasons, cache_dir=corpus_dir).query("game_type == 'R'")
    games = games.reset_index(drop=True)
    starts = pitchers.build(games, cache_dir=pitcher_dir)
    asof = pitchers.as_of_table(starts)

    frame = features.build_core(games, starts, asof, elo_params=ELO_PARAMS)
    elo_ratings = elo.ratings_per_game(games, ELO_PARAMS)
    od = ratings.off_def(games, OFF_DEF_PARAMS)

    data = score.build_dataset(
        games, frame, asof, elo_ratings=elo_ratings, od_expected=od
    )
    data["sp_known"] = (
        data["opp_sp_prior"].notna() & (data["opp_sp_prior"] >= 10)
    ).astype(int)

    # -- plate-appearance block, score model only ---------------------------
    # These do nothing for the win model: nine blocks have now failed there
    # because Elo already summarises the outcomes they cause. Runs are a
    # different target with no such incumbent, and the gain rises with how far
    # tonight's lineup departs from the club's own average, which is the
    # signature of a feature that works by knowing who is playing.
    plate = pa.load(pa_dir)
    talent_prior = talent.fit(plate, alpha=TALENT_ALPHA)
    data, lineup_by_game = _add_pa_block(data, games, plate, talent_prior)
    score_pa = [c for c in score.PA_COLUMNS if data[c].notna().mean() > 0.5]

    # -- lineup block, win model -------------------------------------------
    # The one plate-appearance block the win model keeps. The others failed
    # because Elo already summarises the outcomes they describe; this one says
    # who is playing TONIGHT, which no rating built from past results can know.
    #
    # It is carried on that reasoning rather than on a measured edge: held out
    # over 2022-26 it was worth +0.31 points with the posted card, +0.20 with
    # the projected one the morning build actually has, and neither clears its
    # own standard error. The sign was positive in every variant tried and in
    # four seasons of five. `significant` records that plainly so the report can
    # say so too.
    win_cols = list(features.CORE_COLUMNS)
    lineup_fitted = False
    if len(lineup_by_game):
        frame = frame.merge(lineup_by_game, on="game_pk", how="left")
        covered = frame[features.LINEUP_COLUMNS].notna().all(axis=1).mean()
        # Below half the games the column is mostly the imputed mean, which is
        # not a feature -- it is noise wearing one's name.
        if covered > 0.5:
            win_cols = win_cols + list(features.LINEUP_COLUMNS)
            lineup_fitted = True
        if verbose:
            print(f"  lineup value on {covered * 100:.1f}% of games"
                  f"{'' if lineup_fitted else ' -- too thin, left out'}")
    score_cols = (
        list(score.SCORE_COLUMNS) + list(score.STRENGTH_COLUMNS)
        + ["sp_known"] + score_pa
    )

    # -- weather, runs model ------------------------------------------------
    # The one game-day input that cleared the bar when seven were tested
    # together: z = +3.53 on held-out log-likelihood, better in all five
    # held-out seasons. See `weather.py` for the evidence and the sources.
    # Read from the cache without reaching the network here -- the refit's
    # refresh step is what tops the cache up -- so a fit never stalls on an
    # outside service.
    data = weather.attach(data, Path(pa_dir).parent, games, fetch=False)
    weather_cols = [c for c in weather.WEATHER_COLUMNS
                    if data[c].notna().mean() > 0.5]
    if verbose:
        print(f"  weather on {data['temp_f'].notna().mean() * 100:.1f}% of team-games"
              f"{'' if weather_cols else ' -- too thin, left out'}")
    score_cols = score_cols + weather_cols

    # -- held-out metrics, season by season, before the final fit -----------
    # Each fold's coefficients are kept as well as its score. Predicting one
    # game with all of them and taking the spread is the only estimate of our
    # own uncertainty that reaches past the coefficients to our inputs and to
    # misspecification, and the fits happen here anyway.
    per_season = {}
    folds: list[dict] = []
    held_out_y: list[np.ndarray] = []
    held_out_p: list[np.ndarray] = []
    for test_season in range(FIRST_TEST_SEASON, int(games["season"].max()) + 1):
        train = frame[frame["season"] < test_season]
        test = frame[frame["season"] == test_season]
        if not len(test):
            continue
        x_train, y_train = features.design_matrix(train, win_cols)
        x_test, y_test = features.design_matrix(test, win_cols)
        scaler = StandardScaler().fit(x_train)
        fitted = LogisticRegression(C=LOGISTIC_C, max_iter=3000).fit(
            scaler.transform(x_train), y_train
        )
        probability = fitted.predict_proba(scaler.transform(x_test))[:, 1]
        per_season[test_season] = backtest.evaluate(y_test, probability)
        folds.append({
            "season": int(test_season),
            "coef": [float(c) for c in fitted.coef_[0]],
            "intercept": float(fitted.intercept_[0]),
            "mean": [float(v) for v in scaler.mean_],
            "scale": [float(v) for v in scaler.scale_],
        })
        # Kept so calibration can be measured against outcomes rather than
        # against other versions of this same model.
        held_out_y.append(np.asarray(y_test, dtype=float))
        held_out_p.append(np.asarray(probability, dtype=float))

    calibration = backtest.calibration_bins(
        np.concatenate(held_out_y), np.concatenate(held_out_p),
    ) if held_out_y else []

    # -- disjoint-era fits, purely to measure our own uncertainty ------------
    # The walk-forward folds above are nested: the 2026 fit trains on 2015-2025
    # and the 2025 fit on 2015-2024, sharing over ninety percent of their rows.
    # Their spread therefore measures almost nothing, which is exactly how a
    # first attempt at this returned a smaller figure than the delta method.
    # These blocks share no games at all, so where they disagree the
    # disagreement is real.
    blocks: list[dict] = []
    span = list(seasons)
    for start in range(span[0], span[-1] + 1, BLOCK_YEARS):
        window = frame[
            (frame["season"] >= start) & (frame["season"] < start + BLOCK_YEARS)
        ]
        if len(window) < MIN_BLOCK_GAMES:
            continue
        x_block, y_block = features.design_matrix(window, win_cols)
        block_scaler = StandardScaler().fit(x_block)
        block_fit = LogisticRegression(C=LOGISTIC_C, max_iter=3000).fit(
            block_scaler.transform(x_block), y_block
        )
        blocks.append({
            "label": f"{start}-{min(start + BLOCK_YEARS - 1, span[-1])}",
            "coef": [float(c) for c in block_fit.coef_[0]],
            "intercept": float(block_fit.intercept_[0]),
            "mean": [float(v) for v in block_scaler.mean_],
            "scale": [float(v) for v in block_scaler.scale_],
        })
    if verbose:
        print(f"  {len(blocks)} disjoint-era fits for the uncertainty spread")

    summary = backtest.summarize(per_season)
    if verbose:
        for season, metrics in per_season.items():
            print("  " + metrics.line(str(season)))
        print(
            f"  pooled log loss {summary['log_loss']:.5f} "
            f"(sd {summary['log_loss_sd']:.5f}), lift {summary['lift_pt']:+.2f}pt"
        )

    # -- final fits on everything -------------------------------------------
    x_all, y_all = features.design_matrix(frame, win_cols)
    scaler = StandardScaler().fit(x_all)
    logistic = LogisticRegression(C=LOGISTIC_C, max_iter=3000).fit(
        scaler.transform(x_all), y_all
    )
    win_cov = _win_covariance(
        scaler.transform(x_all),
        logistic.coef_[0],
        float(logistic.intercept_[0]),
    )

    data = _innings_batted(Path(pa_dir).parent, data)
    x_score, y_score, offset, _ = score.design(data, score_cols, exposure=True)
    negbin = sm.GLM(
        y_score,
        x_score,
        family=sm.families.NegativeBinomial(alpha=model.NB_ALPHA),
        offset=offset,
    ).fit()
    score_held_out = _score_backtest(data, score_cols, FIRST_TEST_SEASON)
    # Pearson dispersion of the fitted mean under a Poisson variance. This was
    # stored as the literal 2.19; it is now computed from the fit it describes.
    score_dispersion = score.dispersion(
        y_score, np.asarray(negbin.fittedvalues, dtype=float),
        x_score.shape[1])
    if verbose and score_held_out:
        print(f"  score model held out: MAE {score_held_out['mae']:.3f} "
              f"(league {score_held_out['mae_league']:.3f}, "
              f"own form {score_held_out['mae_own_form']:.3f}), "
              f"bias {score_held_out['bias_runs']:+.3f} runs")

    # -- reference distributions --------------------------------------------
    # A projection means nothing on its own. 60% is a strong call or a routine
    # one depending on what this model usually says, and 9.8 expected runs is
    # high or low depending on the league. Both comparisons need the model's own
    # history, so it is measured here and carried in the artifact rather than
    # recomputed -- or worse, guessed at -- when a report is built.
    z_all = logistic.decision_function(scaler.transform(x_all))
    p_all = 1.0 / (1.0 + np.exp(-z_all))
    hist, edges = np.histogram(p_all, bins=40, range=(0.15, 0.85))

    totals = (games["home_runs"] + games["away_runs"]).dropna().astype(int)
    total_hist = totals.value_counts(normalize=True).sort_index()

    reference = {
        # What this model typically says, so tonight can be placed against it.
        "win_prob_hist": [int(v) for v in hist],
        "win_prob_edges": [float(v) for v in edges],
        "win_prob_sorted": [
            float(v) for v in np.percentile(p_all, np.arange(0, 101))
        ],
        "win_prob_sd": float(p_all.std()),
        # What a normal night's scoring looks like, measured not modelled.
        "total_runs": {
            str(int(k)): float(v) for k, v in total_hist.items() if k <= 24
        },
        "total_runs_mean": float(totals.mean()),
        "n_games": int(len(games)),
    }

    final_ratings = elo.fit_state(games, ELO_PARAMS)
    final_od = ratings.final_off_def(games, OFF_DEF_PARAMS)
    parks = features.park_factors(games)
    latest = parks[parks["season"] == parks["season"].max()]

    artifact = model.OutcomeModel(
        win_columns=win_cols,
        win_coef=[float(c) for c in logistic.coef_[0]],
        win_intercept=float(logistic.intercept_[0]),
        win_mean=[float(v) for v in scaler.mean_],
        win_scale=[float(v) for v in scaler.scale_],
        win_cov=win_cov,
        win_fold_seasons=[f["season"] for f in folds],
        win_fold_coef=[f["coef"] for f in folds],
        win_fold_intercept=[f["intercept"] for f in folds],
        win_fold_mean=[f["mean"] for f in folds],
        win_fold_scale=[f["scale"] for f in folds],
        win_block_labels=[b["label"] for b in blocks],
        win_block_coef=[b["coef"] for b in blocks],
        win_block_intercept=[b["intercept"] for b in blocks],
        win_block_mean=[b["mean"] for b in blocks],
        win_block_scale=[b["scale"] for b in blocks],
        win_calibration=calibration,
        score_columns=score_cols,
        score_coef=[float(c) for c in negbin.params],
        score_mean=[float(v) for v in np.nanmean(x_score, axis=0)],
        alpha=model.NB_ALPHA,
        elo_ratings={str(k): float(v) for k, v in final_ratings.ratings.items()},
        elo_params=ELO_PARAMS.as_dict(),
        off_def={str(k): [float(o), float(d)] for k, (o, d) in final_od.items()},
        park_factors={
            str(int(r.venue_id)): float(r.park_factor) for r in latest.itertuples()
        },
        league_rpg=float(score.league_run_level(games).iloc[-1]),
        talent_batter={str(k): float(v) for k, v in talent_prior.batter.items()},
        talent_pitcher={str(k): float(v) for k, v in talent_prior.pitcher.items()},
        talent_platoon={k: float(v) for k, v in talent_prior.platoon.items()},
        talent_intercept=float(talent_prior.intercept),
        talent_alpha=float(TALENT_ALPHA),
        talent_batter_pa={str(k): int(v) for k, v in talent_prior.batter_pa.items()},
        talent_pitcher_pa={str(k): int(v) for k, v in talent_prior.pitcher_pa.items()},
        slot_weights=[float(w) for w in lineup.slot_weights(
            plate[plate['season'] == plate['season'].max()]
        )],
        metrics={
            "win_model": {
                "log_loss": summary["log_loss"],
                "log_loss_sd": summary["log_loss_sd"],
                "accuracy": summary["accuracy"],
                "baseline_accuracy": summary["baseline_accuracy"],
                "lift_points": summary["lift_pt"],
                "seasons_beating_baseline": summary["seasons_beating_baseline"],
                "seasons": summary["seasons"],
                "n_games": summary["n"],
            },
            "score_model": {
                "alpha": model.NB_ALPHA,
                "dispersion_measured": round(score_dispersion, 3),
                "n_team_games": int(len(y_score)),
                "held_out": score_held_out,
            },
            # Measured over the corpus, not simulated for any one game. The
            # report quotes these when it makes a claim about baseball rather
            # than about tonight -- the two are easy to conflate, and a
            # simulated figure presented as a league fact would be wrong.
            "league_rates": {
                "one_run_game": float(
                    ((games["home_runs"] - games["away_runs"]).abs() == 1).mean()
                ),
                "blowout": float(
                    ((games["home_runs"] - games["away_runs"]).abs() >= 5).mean()
                ),
                "n_games": int(len(games)),
            },
            "elo_benchmark_log_loss": 0.67836,
            # Each season carries the baseline it actually had. Without it the
            # report has nothing to subtract but the pooled figure, which is
            # wrong for every season except the one that happens to match.
            "per_season": {
                str(s): {
                    "log_loss": m.log_loss,
                    "accuracy": m.accuracy,
                    "baseline_accuracy": m.baseline_accuracy,
                    "lift_points": m.lift,
                    "n_games": m.n,
                }
                for s, m in per_season.items()
            },
        },
        reference=reference,
        fitted_at=datetime.now(timezone.utc).isoformat(),
        corpus_through=str(games["game_date"].max()),
    )
    return artifact


def main() -> int:
    from guards_report.config import load_settings

    settings = load_settings()
    root = settings.raw_archive_dir.parent

    print("Fitting outcome models ...")
    artifact = fit(
        corpus_dir=root / "corpus",
        pitcher_dir=root / "pitchers",
        pa_dir=root / "pitches",
        # Through the season in progress, not up to it.
        #
        # This was hardcoded to stop at the current season, and that was a
        # category error rather than a precaution: the walk-forward validation
        # trains on seasons before N and tests on N, which is the right way to
        # *estimate* performance, and the deployed fit inherited the same
        # "complete seasons only" framing. Those answer different questions.
        # Predicting a game that has not been played, from games that have, is
        # not leakage -- it is just using the data you have. Every feature was
        # already as-of; only the coefficients were being denied the season.
        #
        # It also buys the fold that matters most: a held-out score for the
        # current season, testing a model trained through last year, which is
        # exactly the model in production. Without it the report could only say
        # how the model did historically.
        seasons=range(corpus.FIRST_SEASON, date.today().year + 1),
    )
    path = model.save(artifact, root / "models" / "game_outcome.json")
    print(f"\nsaved {path}")
    print(f"  corpus through {artifact.corpus_through}")
    print(f"  teams rated: {len(artifact.elo_ratings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
