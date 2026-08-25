"""Measure every market against outcomes, and store the record.

Run occasionally, not per report. The output is what lets the betting page put
a stake on anything: without a record of how often a stated probability comes
true there is no honest standard error, and a stake is a claim about exactly
that. Before this existed the page could price a moneyline and refused
everything else, correctly.

The expensive part is building the game features, which happens once. The
walk-forward fits on top of them are seconds apiece, so the whole run is
dominated by that single build rather than by the five refits.

Written into the artifacts the served models already load, so nothing new has
to be shipped alongside them.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guards_report.projections import (  # noqa: E402
    atomic, calibrate, corpus, elo, features, first5 as f5_module, model,
    pa as pa_module, pitchers, props, ratings, score, train_props,
)

FIRST_TEST_SEASON = 2022


def _say(message: str) -> None:
    print(message, flush=True)


def _walk_forward_scores(data: pd.DataFrame, columns: list[str], seasons: list[int]):
    """Per-season expected runs for both sides, fitted only on prior seasons."""
    out: dict[int, dict] = {}
    for season in seasons:
        train = data[data["season"] < season]
        test = data[data["season"] == season]
        if not len(train) or not len(test):
            continue
        x_train, y_train, off_train, _ = score.design(train, columns)
        x_test, _, off_test, frame = score.design(test, columns)
        fitted = sm.GLM(
            y_train, x_train,
            family=sm.families.NegativeBinomial(alpha=model.NB_ALPHA),
            offset=off_train,
        ).fit()
        frame = frame.copy()
        frame["mu"] = np.asarray(
            fitted.predict(x_test, offset=off_test), dtype=float)

        home = frame[frame["is_home"] == 1][["game_pk", "mu", "runs"]].rename(
            columns={"mu": "mu_home", "runs": "runs_home"})
        away = frame[frame["is_home"] == 0][["game_pk", "mu", "runs"]].rename(
            columns={"mu": "mu_away", "runs": "runs_away"})
        both = home.merge(away, on="game_pk", how="inner")
        out[season] = {
            "game_pk": both["game_pk"].to_numpy(),
            "mu_home": both["mu_home"].to_numpy(),
            "mu_away": both["mu_away"].to_numpy(),
            "total": (both["runs_home"] + both["runs_away"]).to_numpy(),
        }
    return out


def main() -> int:
    from guards_report.config import load_settings

    settings = load_settings()
    root = settings.raw_archive_dir.parent
    seasons = list(range(FIRST_TEST_SEASON, 2027))

    started = time.time()
    _say("Loading the plate corpus ...")
    plate = pa_module.load(root / "pitches", columns=pa_module.PROP_COLUMNS)
    _say(f"  {len(plate):,} plate appearances")

    starts = pd.concat(
        [pd.read_parquet(f) for f in sorted((root / "pitchers").glob("starts-*.parquet"))],
        ignore_index=True)
    props_artifact = train_props.load_props(root / "models" / "props.json")

    records: dict[str, dict] = {}

    # -- batter counts ----------------------------------------------------
    for outcome in ("hit", "home_run"):
        _say(f"Calibrating {outcome} ...")
        platoon = props.platoon_factors(
            plate[plate["season"] < FIRST_TEST_SEASON], outcome)
        result = calibrate.batter_counts(
            plate, outcome=outcome, seasons=seasons, line=0.5,
            platoon=platoon,
            calibration_factor=props.CALIBRATION.get(outcome, 1.0))
        records[outcome] = result.as_dict()
        _say(f"  n={result.n:,}  seasons {result.seasons}")

    # -- starter strikeouts ------------------------------------------------
    _say("Calibrating strikeouts ...")
    result = calibrate.strikeouts(
        plate, starts, seasons=seasons,
        starter_bf=props_artifact.starter_bf_mean if props_artifact else 21.9)
    records["strikeout"] = result.as_dict()
    _say(f"  n={result.n:,}  seasons {result.seasons}")

    # -- the game features, built once -------------------------------------
    _say("Building game features (the slow part) ...")
    games = corpus.build(
        range(corpus.FIRST_SEASON, 2027),
        cache_dir=root / "corpus").query("game_type == 'R'").reset_index(drop=True)
    logs = pitchers.build(games, cache_dir=root / "pitchers")
    asof = pitchers.as_of_table(logs)
    params = elo.EloParams(k=4, hfa=24, carry=0.70, mov=True)
    frame = features.build_core(games, logs, asof, elo_params=params)
    data = score.build_dataset(
        games, frame, asof,
        elo_ratings=elo.ratings_per_game(games, params),
        od_expected=ratings.off_def(
            games, ratings.OffDefParams(0.010, 0.010, 0.16, 0.70, 0.40)))
    data["sp_known"] = (
        data["opp_sp_prior"].notna() & (data["opp_sp_prior"] >= 10)).astype(int)
    columns = list(score.SCORE_COLUMNS) + list(score.STRENGTH_COLUMNS) + ["sp_known"]
    _say(f"  {len(data):,} team-games")

    # -- game totals -------------------------------------------------------
    _say("Calibrating totals ...")
    blocks = _walk_forward_scores(data, columns, seasons)
    result = calibrate.totals(blocks, alpha=model.NB_ALPHA)
    records["total"] = result.as_dict()
    _say(f"  n={result.n:,}  seasons {result.seasons}")

    # -- first five --------------------------------------------------------
    _say("Calibrating first five ...")
    f5 = pd.read_parquet(root / "models" / "first5.parquet")
    long = pd.concat([
        f5[["game_pk", "home_team", "home_f5"]]
        .rename(columns={"home_team": "team", "home_f5": "f5"}).assign(is_home=1),
        f5[["game_pk", "away_team", "away_f5"]]
        .rename(columns={"away_team": "team", "away_f5": "f5"}).assign(is_home=0),
    ])
    f5_data = data.merge(
        long[["game_pk", "is_home", "f5"]], on=["game_pk", "is_home"], how="inner")
    f5_data = f5_data.copy()
    f5_data["runs"] = f5_data["f5"]
    # The offset has to describe five innings, not nine, or every coefficient
    # absorbs the difference.
    f5_data["league_rpg"] = f5_data["league_rpg"] * train_props.FIRST5_SHARE

    f5_blocks = _walk_forward_scores(f5_data, columns, seasons)
    outcomes = f5.set_index("game_pk")[["f5_home_win", "f5_tie"]]
    for season, block in f5_blocks.items():
        joined = outcomes.reindex(block["game_pk"])
        block["home_win"] = joined["f5_home_win"].fillna(0).to_numpy()
        block["tie"] = joined["f5_tie"].fillna(0).to_numpy()
    result = calibrate.first_five(
        f5_blocks, alpha=train_props.FIRST5_ALPHA)
    records["first_five"] = result.as_dict()
    _say(f"  n={result.n:,}  seasons {result.seasons}")

    # -- persist -----------------------------------------------------------
    target = root / "models" / "market_calibration.json"
    atomic.write_text(target, json.dumps(records, indent=1))
    _say("")
    _say(f"Wrote {target} in {time.time() - started:.0f}s")
    for name, block in records.items():
        bins = block.get("bins") or []
        if not bins:
            _say(f"  {name:12} not measured -- {block.get('note')}")
            continue
        gaps = [b["frequency"] - b["p_mean"] for b in bins]
        _say(f"  {name:12} n={block['n']:>7,}  mean gap {np.mean(gaps):+.4f}  "
             f"worst {max(gaps, key=abs):+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
