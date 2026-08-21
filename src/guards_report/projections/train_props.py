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

from guards_report.projections import corpus, elo, features, pa, pitchers, ratings, score, props

# Chosen by held-out sweep with an interior minimum. Five innings are more
# overdispersed than nine -- a single big inning is a larger share of a shorter
# game -- so the full-game value of 0.275 understates the spread here.
FIRST5_ALPHA = 0.45

# Share of a full game's scoring that happens in the first five innings,
# measured at 5.100 of 8.99 runs. Used as the offset scale so the same feature
# set can predict a shorter game.
FIRST5_SHARE = 5.100 / 8.99

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
            "held_out_log_loss": 0.67956,
            "baseline_log_loss": 0.68760,
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


def main() -> int:
    from guards_report.config import load_settings

    settings = load_settings()
    root = settings.raw_archive_dir.parent

    print("Fitting player props ...")
    plate = pa.load(root / "pitches", columns=pa.PROP_COLUMNS)
    props_artifact = fit_props(plate)
    props_path = save(props_artifact, root / "models" / "props.json")
    print(f"saved {props_path}  ({props_path.stat().st_size/1024:.0f} KB)")

    first5_path = root / "models" / "first5.parquet"
    if first5_path.exists():
        print("\nFitting first-five model ...")
        f5 = pd.read_parquet(first5_path)
        f5_artifact = fit_first5(
            corpus_dir=root / "corpus", pitcher_dir=root / "pitchers", first5=f5
        )
        path = save(f5_artifact, root / "models" / "first5.json")
        print(f"saved {path}")
    else:
        print("\nno first-five dataset; skipping", first5_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
