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
    score, talent,
)

ELO_PARAMS = elo.EloParams(k=4, hfa=24, carry=0.70, mov=True)
OFF_DEF_PARAMS = ratings.OffDefParams(0.010, 0.010, 0.16, 0.70, 0.40)

# Regularization chosen by sweep on 2018-2021; the curve is flat above 0.03.
LOGISTIC_C = 3.0

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
    sp_rows, lu_rows = [], []

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
        lu_rows.append(
            merged.groupby(["game_pk", "batting_team"], as_index=False)
            .apply(lambda b: pd.Series({
                "lineup_value": float((b.score * b.w).sum() / b.w.sum())
                if b.w.sum() else 0.0,
            }), include_groups=False)
        )

    if not sp_rows:
        for column in score.PA_COLUMNS:
            data[column] = np.nan
        return data

    sp = pd.concat(sp_rows, ignore_index=True)
    lu = pd.concat(lu_rows, ignore_index=True)

    data = data.merge(sp, on="game_pk", how="left")
    data["opp_sp_talent"] = np.where(
        data["is_home"] == 1, data["away_sp_talent"], data["home_sp_talent"]
    )
    # The score dataset keys on numeric team id while lineups key on the club
    # abbreviation. Pivoting to home and away sidesteps the mapping entirely and
    # matches how the starter column above is handled.
    sides = games[["game_pk", "home_team", "away_team"]]
    home = lu.rename(columns={"batting_team": "home_team", "lineup_value": "home_lineup"})
    away = lu.rename(columns={"batting_team": "away_team", "lineup_value": "away_lineup"})
    wide = (
        sides.merge(home, on=["game_pk", "home_team"], how="left")
             .merge(away, on=["game_pk", "away_team"], how="left")
    )[["game_pk", "home_lineup", "away_lineup"]]

    data = data.merge(wide, on="game_pk", how="left")
    data["own_lineup"] = np.where(
        data["is_home"] == 1, data["home_lineup"], data["away_lineup"]
    )
    return data


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
    data = _add_pa_block(data, games, plate, talent_prior)
    score_pa = [c for c in score.PA_COLUMNS if data[c].notna().mean() > 0.5]

    win_cols = list(features.CORE_COLUMNS)
    score_cols = (
        list(score.SCORE_COLUMNS) + list(score.STRENGTH_COLUMNS)
        + ["sp_known"] + score_pa
    )

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
