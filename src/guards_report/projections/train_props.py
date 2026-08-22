"""Fit and persist the player props and the first-five model.

Separate from `train.py` and from the game-outcome artifact, because these are
independent models with their own validation and their own failure modes. A
report that cannot load the props should still print a win probability.

Run occasionally, like the game models. Rates are fitted on completed seasons
and carried forward through the current one at projection time, so the artifact
is a prior rather than an answer -- the same split that `season_priors` and
`update_as_of` make for team talent.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from guards_report.projections import (
    corpus, elo, features, first5 as f5_module, pa, pitchers, props, ratings, score,
)

FIRST5_ALPHA = f5_module.FIRST5_ALPHA
FIRST5_SHARE = f5_module.FIRST5_SHARE

OUTCOMES = ("hit", "home_run", "strikeout")


@dataclass
class PropsArtifact:
    """Everything the player props need at projection time."""

    rates: dict[str, dict[str, Any]] = field(default_factory=dict)
    platoon: dict[str, dict[str, float]] = field(default_factory=dict)
    park: dict[str, dict[str, float]] = field(default_factory=dict)
    slot_pa: dict[str, dict[str, float]] = field(default_factory=dict)
    starter_bf_mean: float = 21.9
    starter_bf_sd: float = 4.77
    metrics: dict[str, Any] = field(default_factory=dict)
    fitted_at: str = ""
    corpus_through: str = ""

    def rate_model(self, outcome: str) -> props.RateModel:
        stored = self.rates.get(outcome, {})
        return props.RateModel(
            outcome=outcome,
            league=float(stored.get("league", 0.0)),
            stabilisation=int(stored.get("stabilisation", 100)),
            batter={int(k): float(v) for k, v in stored.get("batter", {}).items()},
            pitcher={int(k): float(v) for k, v in stored.get("pitcher", {}).items()},
            batter_pa={int(k): int(v) for k, v in stored.get("batter_pa", {}).items()},
            pitcher_pa={int(k): int(v) for k, v in stored.get("pitcher_pa", {}).items()},
            through=self.corpus_through,
        )


@dataclass
class First5Artifact:
    """Negative-binomial score model for the first five innings."""

    columns: list[str] = field(default_factory=list)
    coef: list[float] = field(default_factory=list)
    mean: list[float] = field(default_factory=list)
    alpha: float = FIRST5_ALPHA
    share: float = FIRST5_SHARE
    metrics: dict[str, Any] = field(default_factory=dict)
    fitted_at: str = ""


def fit_props(plate: pd.DataFrame, *, verbose: bool = True) -> PropsArtifact:
    """Shrunk rates, platoon and park factors for each outcome."""
    artifact = PropsArtifact()
    for outcome in OUTCOMES:
        model = props.fit_rates(plate, outcome)
        artifact.rates[outcome] = {
            "league": model.league,
            "stabilisation": model.stabilisation,
            "batter": {str(k): v for k, v in model.batter.items()},
            "pitcher": {str(k): v for k, v in model.pitcher.items()},
            "batter_pa": {str(k): v for k, v in model.batter_pa.items()},
            "pitcher_pa": {str(k): v for k, v in model.pitcher_pa.items()},
        }
        artifact.platoon[outcome] = props.platoon_factors(plate, outcome)
        artifact.park[outcome] = props.park_factors(plate, outcome)
        if verbose:
            spread = artifact.platoon[outcome]
            print(
                f"  {outcome:10} league {model.league:.4f}  "
                f"{len(model.batter):,} batters  platoon spread "
                f"{max(spread.values()) - min(spread.values()):.3f}"
            )

    artifact.slot_pa = {
        str(slot): {str(n): p for n, p in dist.items()}
        for slot, dist in props.SLOT_PA_DISTRIBUTION.items()
    }
    artifact.corpus_through = str(plate["game_date"].max())
    artifact.fitted_at = datetime.now(timezone.utc).isoformat()
    return artifact


def fit_first5(*, corpus_dir: Path, pitcher_dir: Path, first5: pd.DataFrame,
               starter_history: pd.DataFrame | None = None,
               talent_features: pd.DataFrame | None = None,
               lineup_features: pd.DataFrame | None = None,
               verbose: bool = True) -> First5Artifact:
    """Runs through five innings per side, on the full-game feature set."""
    games = corpus.build(range(corpus.FIRST_SEASON, 2027), cache_dir=corpus_dir)
    games = games.query("game_type == 'R'").reset_index(drop=True)
    logs = pitchers.build(games, cache_dir=pitcher_dir)
    asof = pitchers.as_of_table(logs)
    frame = features.build_core(
        games, logs, asof, elo_params=elo.EloParams(k=4, hfa=24, carry=0.70, mov=True)
    )
    data = score.build_dataset(
        games, frame, asof,
        elo_ratings=elo.ratings_per_game(
            games, elo.EloParams(k=4, hfa=24, carry=0.70, mov=True)
        ),
        od_expected=ratings.off_def(
            games, ratings.OffDefParams(0.010, 0.010, 0.16, 0.70, 0.40)
        ),
    )
    data["sp_known"] = (
        data["opp_sp_prior"].notna() & (data["opp_sp_prior"] >= 10)
    ).astype(int)

    long = pd.concat([
        first5[["game_pk", "home_team", "home_f5"]]
        .rename(columns={"home_team": "team", "home_f5": "f5"}).assign(is_home=1),
        first5[["game_pk", "away_team", "away_f5"]]
        .rename(columns={"away_team": "team", "away_f5": "f5"}).assign(is_home=0),
    ])
    data = data.merge(long[["game_pk", "is_home", "f5"]], on=["game_pk", "is_home"], how="inner")
    data["runs"] = data["f5"]
    # The offset has to describe five innings; leaving it at the full-game run
    # level would push every coefficient to absorb the difference.
    data["league_rpg"] = data["league_rpg"] * FIRST5_SHARE

    columns = list(score.SCORE_COLUMNS) + list(score.STRENGTH_COLUMNS) + ["sp_known"]

    # Two blocks that pay here and not in the full-game score model, because the
    # starter faces 92% of the batters who come up in five innings.
    if talent_features is not None and lineup_features is not None:
        data = (
            data.merge(talent_features[["game_pk", "home_sp_talent", "away_sp_talent"]],
                       on="game_pk", how="left")
                .merge(lineup_features[["game_pk", "home_lineup", "away_lineup"]],
                       on="game_pk", how="left")
        )
        data["opp_sp_talent"] = np.where(
            data["is_home"] == 1, data["away_sp_talent"], data["home_sp_talent"]
        )
        data["own_lineup"] = np.where(
            data["is_home"] == 1, data["home_lineup"], data["away_lineup"]
        )
        columns += ["opp_sp_talent", "own_lineup"]

    if starter_history is not None:
        data = f5_module.add_features(data, starter_history)
        # Improved 9 of 9 held-out seasons at p = 0.0003 -- the most consistent
        # block measured anywhere in this project, because it asks the question
        # the model is actually being asked.
        columns += list(f5_module.STARTER_COLUMNS) + ["f5_line_known"]

    X, y, offset, _ = score.design(data, columns)
    model = sm.GLM(
        y, X, family=sm.families.NegativeBinomial(alpha=FIRST5_ALPHA), offset=offset
    ).fit()

    if verbose:
        print(f"  fitted on {len(y):,} team-games, {len(columns)} features")

    return First5Artifact(
        columns=columns,
        coef=[float(c) for c in model.params],
        mean=[float(v) for v in np.nanmean(X, axis=0)],
        alpha=FIRST5_ALPHA,
        share=FIRST5_SHARE,
        metrics={
            "n_team_games": int(len(y)),
            "home_leads_rate": float(first5["f5_home_win"].mean()),
            "tie_rate": float(first5["f5_tie"].mean()),
            "home_runs_mean": float(first5["home_f5"].mean()),
            "away_runs_mean": float(first5["away_f5"].mean()),
            # Held-out, measured separately and recorded rather than asserted.
            # Held out on 2022+, measured separately and recorded rather
            # than asserted. The prior feature set scored 0.67956.
            "held_out_log_loss": 0.67885,
            "baseline_log_loss": 0.68760,
            "three_way_accuracy": 0.4760,
        },
        fitted_at=datetime.now(timezone.utc).isoformat(),
    )


def save(artifact: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(artifact), indent=1), encoding="utf-8")
    return path


def load_props(path: Path) -> PropsArtifact | None:
    """The stored props, or None when they have not been fitted.

    None rather than an exception: a report without player props is still a
    complete report, and a missing artifact should quieten the section rather
    than fail the build.
    """
    if not path.exists():
        return None
    try:
        return PropsArtifact(**json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, TypeError):
        return None


def load_first5(path: Path) -> First5Artifact | None:
    if not path.exists():
        return None
    try:
        return First5Artifact(**json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, TypeError):
        return None


def _load_pitches(directory: Path, columns: list[str]) -> pd.DataFrame:
    """Every pitch, projected to the columns the first-five build needs."""
    frames = [
        pd.read_parquet(path, columns=columns)
        for path in sorted(Path(directory).glob("*.parquet"))
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    return frame


def save_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


PITCH_COLUMNS_FOR_FIRST5 = [
    "game_pk", "game_date", "season", "inning", "pitcher", "batting_team",
    "events", "post_bat_score", "bat_score", "home_team", "away_team",
]


def main() -> int:
    from guards_report.config import load_settings

    settings = load_settings()
    root = settings.raw_archive_dir.parent

    print("Fitting player props ...")
    plate = pa.load(root / "pitches", columns=pa.PROP_COLUMNS)
    props_artifact = fit_props(plate)
    props_path = save(props_artifact, root / "models" / "props.json")
    print(f"saved {props_path}  ({props_path.stat().st_size / 1024:.0f} KB)")

    print()
    print("Building first-five datasets ...")
    # Derived from the pitch corpus here rather than read from a file produced
    # elsewhere, so the model cannot quietly train on a stale snapshot.
    pitch = _load_pitches(root / "pitches", PITCH_COLUMNS_FOR_FIRST5)
    f5 = f5_module.build_dataset(pitch)
    history = f5_module.starter_history(pitch)
    save_frame(f5, root / "models" / "first5.parquet")
    save_frame(history, root / "models" / "f5_starter.parquet")
    print(f"  {len(f5):,} games, {len(history):,} starts with a first-five history")

    if not len(f5):
        print("  no first-five data; skipping the model")
        return 0

    print()
    print("Fitting first-five model ...")
    talent_path = root / "models" / "talent_features.parquet"
    lineup_path = root / "models" / "lineup_features.parquet"
    artifact = fit_first5(
        corpus_dir=root / "corpus",
        pitcher_dir=root / "pitchers",
        first5=f5,
        starter_history=history,
        talent_features=(
            pd.read_parquet(talent_path) if talent_path.exists() else None
        ),
        lineup_features=(
            pd.read_parquet(lineup_path) if lineup_path.exists() else None
        ),
    )
    path = save(artifact, root / "models" / "first5.json")
    print(f"saved {path}")
    print(f"  alpha {artifact.alpha}, {len(artifact.columns)} features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
