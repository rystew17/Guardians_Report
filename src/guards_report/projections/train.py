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

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from guards_report.projections import (
    backtest, corpus, elo, features, model, pitchers, ratings, score,
)

ELO_PARAMS = elo.EloParams(k=4, hfa=24, carry=0.70, mov=True)
OFF_DEF_PARAMS = ratings.OffDefParams(0.010, 0.010, 0.16, 0.70, 0.40)

# Regularization chosen by sweep on 2018-2021; the curve is flat above 0.03.
LOGISTIC_C = 3.0

FIRST_TEST_SEASON = 2022


def fit(
    *, corpus_dir: Path, pitcher_dir: Path, seasons: range, verbose: bool = True
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

    win_cols = list(features.CORE_COLUMNS)
    score_cols = list(score.SCORE_COLUMNS) + list(score.STRENGTH_COLUMNS) + ["sp_known"]

    # -- held-out metrics, season by season, before the final fit -----------
    per_season = {}
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

    x_score, y_score, offset, _ = score.design(data, score_cols)
    negbin = sm.GLM(
        y_score,
        x_score,
        family=sm.families.NegativeBinomial(alpha=model.NB_ALPHA),
        offset=offset,
    ).fit()

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
        metrics={
            "win_model": {
                "log_loss": summary["log_loss"],
                "log_loss_sd": summary["log_loss_sd"],
                "accuracy": summary["accuracy"],
                "baseline_accuracy": 0.5332,
                "lift_points": summary["lift_pt"],
                "seasons_beating_baseline": summary["seasons_beating_baseline"],
                "seasons": summary["seasons"],
                "n_games": summary["n"],
            },
            "score_model": {
                "alpha": model.NB_ALPHA,
                "dispersion_measured": 2.19,
                "n_team_games": int(len(y_score)),
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
            "per_season": {
                str(s): {"log_loss": m.log_loss, "accuracy": m.accuracy}
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
        seasons=range(corpus.FIRST_SEASON, 2026),
    )
    path = model.save(artifact, root / "models" / "game_outcome.json")
    print(f"\nsaved {path}")
    print(f"  corpus through {artifact.corpus_through}")
    print(f"  teams rated: {len(artifact.elo_ratings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
