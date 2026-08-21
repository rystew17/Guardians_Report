"""Fitting, persisting and applying the game-outcome models.

Everything up to this point has predicted games whose outcome the corpus already
contains. Projecting a game that has not been played needs three things that
backtesting does not: the fitted parameters stored rather than recomputed, the
rating state carried forward to today, and a feature path that works without a
result to look up.

Two models are persisted together because they are checked against each other:

* **Model A** — regularized logistic on the Core block. Held-out log loss
  0.67668, accuracy 0.5730 (+4.30pt over the 0.5332 baseline), ECE 0.0074.
* **Model B** — negative binomial per side, alpha 0.275. Its simulated win
  probability agrees with Model A to +0.00029 log loss, which is the coherence
  condition the two were built to satisfy.

The artifact records the metrics measured at fit time, so a projection can
always be shown next to the accuracy it actually achieved rather than asking a
reader to take it on trust.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Fitted on the training seasons and held fixed thereafter. Re-estimating alpha
# per run would make two reports of the same game disagree for no reason.
NB_ALPHA = 0.275

# Simulations per projection. 20k keeps the Monte Carlo error on a win
# probability near +/-0.0035, which is far below the model's own error and small
# enough to be invisible at the precision the report displays.
SIMULATIONS = 20_000


@dataclass
class FitMetrics:
    """How the model scored when it was fitted, carried with the artifact."""

    log_loss: float = 0.0
    accuracy: float = 0.0
    baseline_accuracy: float = 0.5332
    calibration_error: float = 0.0
    n_games: int = 0
    seasons: str = ""

    @property
    def lift_points(self) -> float:
        return (self.accuracy - self.baseline_accuracy) * 100


@dataclass
class OutcomeModel:
    """Both fitted models plus everything needed to apply them to a new game."""

    # Model A: standardized logistic
    win_columns: list[str] = field(default_factory=list)
    win_coef: list[float] = field(default_factory=list)
    win_intercept: float = 0.0
    win_mean: list[float] = field(default_factory=list)
    win_scale: list[float] = field(default_factory=list)

    # Model B: negative binomial, one coefficient set applied to both sides
    score_columns: list[str] = field(default_factory=list)
    score_coef: list[float] = field(default_factory=list)
    score_mean: list[float] = field(default_factory=list)
    alpha: float = NB_ALPHA

    # Latent batter and pitcher quality, fitted on seasons before the corpus
    # ends. Stored for the same reason as the ratings: refitting a ridge over
    # nineteen hundred thousand plate appearances during report generation would
    # cost a minute and change nothing, and the current season is layered on at
    # projection time by `talent.update_as_of`.
    talent_batter: dict[str, float] = field(default_factory=dict)
    talent_pitcher: dict[str, float] = field(default_factory=dict)
    talent_platoon: dict[str, float] = field(default_factory=dict)
    talent_intercept: float = 0.0
    talent_alpha: float = 800.0
    talent_batter_pa: dict[str, int] = field(default_factory=dict)
    talent_pitcher_pa: dict[str, int] = field(default_factory=dict)
    slot_weights: list[float] = field(default_factory=list)

    # State carried forward so an unplayed game can be rated
    elo_ratings: dict[str, float] = field(default_factory=dict)
    elo_params: dict[str, Any] = field(default_factory=dict)
    off_def: dict[str, list[float]] = field(default_factory=dict)
    park_factors: dict[str, float] = field(default_factory=dict)
    league_rpg: float = 4.5

    metrics: dict[str, Any] = field(default_factory=dict)
    # Distributions of this model's own past output, for placing one projection
    # against what it normally says.
    reference: dict[str, Any] = field(default_factory=dict)
    fitted_at: str = ""
    corpus_through: str = ""

    # -- Model A ---------------------------------------------------------
    def win_probability(self, features: dict[str, float]) -> float:
        """P(home win) from the Core block."""
        z = self.win_intercept
        for name, coef, mean, scale in zip(
            self.win_columns, self.win_coef, self.win_mean, self.win_scale
        ):
            value = features.get(name)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = mean
            z += coef * ((value - mean) / (scale or 1.0))
        return float(1.0 / (1.0 + np.exp(-z)))

    def win_contributions(self, features: dict[str, float]) -> list[dict]:
        """Each feature's push on the log-odds, relative to a league-average game.

        A single probability says what the model concluded but not why, and the
        why is the part that differs from night to night. Because the model is
        linear on the log-odds scale, the decomposition is exact rather than an
        attribution heuristic: the intercept is the average game, and these
        terms sum to the rest.

        Reported in log-odds, since that is the scale they are additive on;
        converting each to "percentage points" separately would not sum to the
        total and would invite exactly the wrong reading.
        """
        out = []
        for name, coef, mean, scale in zip(
            self.win_columns, self.win_coef, self.win_mean, self.win_scale
        ):
            value = features.get(name)
            imputed = value is None or (
                isinstance(value, float) and np.isnan(value)
            )
            if imputed:
                value = mean
            out.append({
                "name": name,
                "value": float(value),
                "league_mean": float(mean),
                "contribution": float(coef * ((value - mean) / (scale or 1.0))),
                "imputed": bool(imputed),
            })
        return sorted(out, key=lambda row: -abs(row["contribution"]))

    def percentile_of(self, probability: float) -> float:
        """Where a win probability sits in this model's own historical spread.

        Answers the question the bare number cannot: is this a confident call or
        a routine one? Returns 50.0 when no reference was stored.
        """
        grid = self.reference.get("win_prob_sorted")
        if not grid:
            return 50.0
        side = max(probability, 1.0 - probability)
        # The reference is one-sided around the home team; fold it so the
        # percentile describes conviction rather than which club is favoured.
        folded = sorted(max(v, 1.0 - v) for v in grid)
        rank = float(np.searchsorted(folded, side)) / max(len(folded) - 1, 1)
        return float(min(max(rank * 100.0, 0.0), 100.0))

    # -- Model B ---------------------------------------------------------
    def expected_runs(self, features: dict[str, float]) -> float:
        """Expected runs for one side, on the trailing league run level."""
        linear = 0.0
        for name, coef, mean in zip(
            self.score_columns, self.score_coef, self.score_mean
        ):
            value = features.get(name)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = mean
            linear += coef * value
        offset = np.log(max(self.league_rpg, 0.5))
        return float(np.exp(linear + offset))

    def simulate(
        self, home_features: dict, away_features: dict, *, draws: int = SIMULATIONS
    ) -> dict[str, Any]:
        """Full joint score distribution, and everything derived from it.

        Sampling rather than solving analytically is deliberate: once the two
        sides are drawn, any question about the game -- who wins, by how much,
        the chance of a shutout, the most likely scoreline -- is answered by
        counting, with no further approximation.

        Ties are resolved by re-drawing the tied games rather than splitting
        them, because baseball has no ties: extra innings are played until
        somebody leads.
        """
        mu_home = self.expected_runs(home_features)
        mu_away = self.expected_runs(away_features)

        rng = np.random.default_rng(20260819)
        n = 1.0 / self.alpha

        def draw(mu: float, size: int) -> np.ndarray:
            return rng.negative_binomial(n, n / (n + mu), size=size)

        home = draw(mu_home, draws)
        away = draw(mu_away, draws)

        # Extra innings: keep re-drawing the tied subset until it resolves.
        for _ in range(20):
            tied = home == away
            if not tied.any():
                break
            # Extra innings are low-scoring; sample a short additional frame
            # rather than a whole second game.
            home = home + tied * draw(mu_home / 9.0, draws)
            away = away + tied * draw(mu_away / 9.0, draws)
        still_tied = home == away
        if still_tied.any():
            home = home + still_tied * rng.integers(0, 2, size=draws)

        margin = home - away
        p_home = float((home > away).mean())

        # Most likely exact scoreline over the plausible grid.
        grid: dict[tuple[int, int], int] = {}
        for h, a in zip(home[:5000], away[:5000]):
            grid[(int(h), int(a))] = grid.get((int(h), int(a)), 0) + 1
        modal = max(grid.items(), key=lambda kv: kv[1])[0] if grid else (4, 3)

        return {
            "exp_home_runs": mu_home,
            "exp_away_runs": mu_away,
            "home_win_probability": p_home,
            "away_win_probability": 1.0 - p_home,
            "modal_score": {"home": modal[0], "away": modal[1]},
            "median_home": int(np.median(home)),
            "median_away": int(np.median(away)),
            "home_runs_p10": int(np.percentile(home, 10)),
            "home_runs_p90": int(np.percentile(home, 90)),
            "away_runs_p10": int(np.percentile(away, 10)),
            "away_runs_p90": int(np.percentile(away, 90)),
            "expected_total": float((home + away).mean()),
            "expected_margin": float(margin.mean()),
            "p_home_shutout": float((away == 0).mean()),
            "p_away_shutout": float((home == 0).mean()),
            "p_one_run_game": float((np.abs(margin) == 1).mean()),
            "p_blowout": float((np.abs(margin) >= 5).mean()),
            "score_grid": _grid(home, away),
            "total_distribution": _totals(home + away),
            # The margin is the most readable view of the same simulation: the
            # win probability becomes the area on one side of zero, one-run
            # games are the two bars nearest the middle, and blowouts are the
            # tails. A joint grid carries the same information but spreads it
            # over a hundred near-identical cells.
            "margin_distribution": _margins(margin),
            "home_runs_distribution": _runs(home),
            "away_runs_distribution": _runs(away),
            "draws": draws,
        }

    def talent(self):
        """Rebuild the fitted talent object from the stored fields."""
        from guards_report.projections.talent import Talent

        return Talent(
            batter={int(k): v for k, v in self.talent_batter.items()},
            pitcher={int(k): v for k, v in self.talent_pitcher.items()},
            platoon=dict(self.talent_platoon),
            intercept=self.talent_intercept,
            alpha=self.talent_alpha,
            batter_pa={int(k): v for k, v in self.talent_batter_pa.items()},
            pitcher_pa={int(k): v for k, v in self.talent_pitcher_pa.items()},
            through=self.corpus_through,
        )




def _grid(home: np.ndarray, away: np.ndarray, *, limit: int = 12) -> list[dict]:
    """Joint probability over plausible finals, for the heat grid."""
    out = []
    total = len(home)
    for h in range(limit + 1):
        for a in range(limit + 1):
            count = int(((home == h) & (away == a)).sum())
            if count:
                out.append({"home": h, "away": a, "p": count / total})
    return sorted(out, key=lambda row: -row["p"])


def _margins(margin: np.ndarray, *, limit: int = 9) -> list[dict]:
    """Distribution of the home margin, tails collected at the edges."""
    out = []
    n = len(margin)
    for m in range(-limit, limit + 1):
        if m == 0:
            continue  # baseball has no ties
        if m == -limit:
            count = int((margin <= m).sum())
        elif m == limit:
            count = int((margin >= m).sum())
        else:
            count = int((margin == m).sum())
        if count:
            out.append({"margin": m, "p": count / n})
    return out


def _runs(runs: np.ndarray, *, limit: int = 12) -> list[dict]:
    out = []
    n = len(runs)
    for r in range(limit + 1):
        count = int((runs >= r).sum()) if r == limit else int((runs == r).sum())
        if count:
            out.append({"runs": r, "p": count / n})
    return out


def _totals(totals: np.ndarray, *, limit: int = 24) -> list[dict]:
    out = []
    n = len(totals)
    for t in range(limit + 1):
        count = int((totals == t).sum())
        if count:
            out.append({"total": t, "p": count / n})
    return out



def save(model: OutcomeModel, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(model), indent=1), encoding="utf-8")
    return path


def load(path: Path) -> OutcomeModel | None:
    """The stored model, or None when it has not been fitted yet.

    Returns None rather than raising: a report without projections is still a
    complete report, and a missing artifact should degrade the page rather than
    fail the build.
    """
    if not path.exists():
        return None
    try:
        return OutcomeModel(**json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, TypeError):
        return None
