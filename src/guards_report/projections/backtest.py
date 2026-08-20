"""Evaluation — walk-forward validation and the metrics that decide things.

Two rules this module exists to enforce.

**Never evaluate out of time order.** Random k-fold trains on the future to
predict the past, which on this problem inflates every metric and hides leakage
rather than exposing it. Splits here are always chronological.

**Judge on log loss and calibration, not accuracy.** Accuracy throws away the
probability: a model that says 51% and one that says 95% score identically when
both are right. For a projection published in a report, being right about *how
sure to be* is the whole product.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np


@dataclass(frozen=True)
class Metrics:
    """How a set of probabilistic predictions did."""

    n: int
    log_loss: float
    brier: float
    accuracy: float
    baseline_accuracy: float
    calibration_error: float

    @property
    def lift(self) -> float:
        """Accuracy above the always-pick-home baseline, in points."""
        return (self.accuracy - self.baseline_accuracy) * 100

    def line(self, label: str = "") -> str:
        return (
            f"{label:<26} n={self.n:>6,}  logloss={self.log_loss:.5f}  "
            f"brier={self.brier:.5f}  acc={self.accuracy:.4f} "
            f"({self.lift:+.2f}pt)  ECE={self.calibration_error:.4f}"
        )


def evaluate(y_true: np.ndarray, p_pred: np.ndarray, *, bins: int = 10) -> Metrics:
    """Score predictions, including how well calibrated they are.

    Probabilities are clipped before the log: a confident miss at exactly 0 or 1
    would otherwise return infinity and destroy the mean, which tells us nothing
    useful about the model.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_pred, dtype=float), 1e-15, 1 - 1e-15)

    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    brier = float(np.mean((p - y) ** 2))
    accuracy = float(np.mean((p >= 0.5) == (y == 1)))

    # Always picking the home team is the benchmark a projection must clear to
    # be worth publishing at all.
    baseline = float(max(y.mean(), 1 - y.mean()))

    return Metrics(
        n=len(y),
        log_loss=log_loss,
        brier=brier,
        accuracy=accuracy,
        baseline_accuracy=baseline,
        calibration_error=expected_calibration_error(y, p, bins=bins),
    )


def expected_calibration_error(
    y_true: np.ndarray, p_pred: np.ndarray, *, bins: int = 10
) -> float:
    """Mean gap between stated confidence and observed frequency.

    Bins predictions, then compares each bin's average prediction to how often
    that bin actually won, weighted by bin size. Zero means a stated 62% wins
    62% of the time -- which is the property a published projection lives or
    dies on.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        total += mask.sum() * abs(p[mask].mean() - y[mask].mean())

    return float(total / len(y)) if len(y) else 0.0


def reliability_table(
    y_true: np.ndarray, p_pred: np.ndarray, *, bins: int = 10
) -> list[dict[str, Any]]:
    """Per-bin predicted vs observed, for a reliability curve."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        rows.append({
            "bin_low": float(lo),
            "bin_high": float(hi),
            "n": int(mask.sum()),
            "predicted": float(p[mask].mean()),
            "observed": float(y[mask].mean()),
        })
    return rows


def walk_forward(frame, *, min_train_seasons: int = 3) -> Iterator[tuple[Any, Any, int]]:
    """Expanding-window splits: train on everything before season N, test on N.

    Yields (train, test, season). Every season after the warm-up becomes test
    data exactly once, which gives a spread across years rather than a single
    number from one arbitrarily chosen holdout -- the difference between knowing
    a model is better and knowing it had one good season.
    """
    seasons = sorted(frame["season"].unique())
    for i in range(min_train_seasons, len(seasons)):
        test_season = seasons[i]
        train = frame[frame["season"] < test_season]
        test = frame[frame["season"] == test_season]
        if len(test):
            yield train, test, int(test_season)


def summarize(per_season: dict[int, Metrics]) -> dict[str, float]:
    """Pool fold results, reporting spread as well as centre.

    The standard deviation across folds is the honest part: a mean log loss that
    looks like an improvement but swings wildly year to year has not established
    anything.
    """
    losses = np.array([m.log_loss for m in per_season.values()])
    accs = np.array([m.accuracy for m in per_season.values()])
    lifts = np.array([m.lift for m in per_season.values()])
    weights = np.array([m.n for m in per_season.values()], dtype=float)

    return {
        "seasons": len(per_season),
        "n": int(weights.sum()),
        "log_loss": float(np.average(losses, weights=weights)),
        "log_loss_sd": float(losses.std(ddof=1)) if len(losses) > 1 else 0.0,
        "accuracy": float(np.average(accs, weights=weights)),
        "lift_pt": float(np.average(lifts, weights=weights)),
        "lift_sd": float(lifts.std(ddof=1)) if len(lifts) > 1 else 0.0,
        "seasons_beating_baseline": int(sum(1 for m in per_season.values() if m.lift > 0)),
    }
