"""How wrong our own probability might be.

This is the number the whole feature turns on, and it is easy to miss why.
Running the selector backwards -- given a threshold, what disagreement with the
book would it actually take to place a bet -- the answer moves like this:

    standard error   disagreement needed to clear z > 2.5 on a -110 moneyline
    -------------------------------------------------------------------------
    0.030            15.4 points
    0.020             9.4 points
    0.015             7.2 points

A fifteen-point disagreement with a closing moneyline does not happen. A
seven-point one happens occasionally. So sigma, not tau and not the threshold,
decides whether this feature ever fires at all -- and unlike everything else in
the pipeline it was never measured, only assumed.

Three estimators, and the split between them is the point.

`delta_se` is the textbook standard error of a logistic prediction, from the
stored coefficient covariance. Exact for what it covers, blind to everything
else. Our features are themselves estimates -- a pitcher's talent inferred from
a finite sample, a lineup projected before it is posted -- and none of that
variance appears here. Neither does the possibility that the model is simply
wrong about this matchup. Against 27,000 games it returns roughly 0.005, which
is a true statement about the coefficients and a misleading one about tonight.

`fold_spread` pushes one game through several independently fitted models and
measures how far apart they land. It uses disjoint four-year eras rather than
the walk-forward folds, for a reason worth recording: the folds are nested and
share over ninety percent of their rows, so built as the conservative estimator
it came back *smaller* than the delta method -- the signature of measuring
nothing at all.

`calibration_sigma` is the only one that looks outside. The first two ask how
much the model agrees with itself, and a model can be perfectly self-consistent
and perfectly wrong. This one compares out-of-sample predictions against
realized frequencies, with the binomial component subtracted so what remains is
systematic rather than sampling noise.

Taking the largest is the conservative reading and the honest one. Understating
sigma is the more expensive mistake by some distance: it inflates z, which fires
more bets, *and* it shrinks less toward the market, which stakes each one
bigger. Both errors push the same way at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

# Floor on any reported standard error. A model that claims to know a baseball
# game's outcome to better than half a point is not being confident, it is
# being wrong -- and the consequence lands directly on the stake.
MINIMUM_SIGMA = 0.005


@dataclass(frozen=True)
class Sigma:
    """Both estimates and the one to use, so a reader can see which dominates."""

    delta: float | None
    folds: float | None
    calibration: float | None
    n_folds: int
    value: float
    source: str          # "delta" | "folds" | "calibration" | "floor"

    @property
    def measured(self) -> bool:
        """False when neither estimator could run.

        The caller must treat that as "cannot price this", never as agreement
        between the two -- an artifact fitted before these were stored produces
        exactly this state, and it would otherwise read as certainty.
        """
        return any(v is not None
                   for v in (self.delta, self.folds, self.calibration))


def delta_se(outcome_model, features: dict[str, float]) -> float | None:
    """Standard error of the predicted win probability, from the covariance.

    On the log-odds scale the prediction is linear, so its variance is
    `x' Cov x`. Mapping back through the logistic gives the delta-method form:

        d p / d eta = p (1 - p)
        SE(p)       = p (1 - p) * sqrt(x' Cov x)

    The same design vector that produced the probability is used here, so the
    error cannot end up qualifying a different number than the one shown.
    """
    cov = getattr(outcome_model, "win_cov", None)
    if not cov:
        return None
    row = outcome_model.win_design_vector(features)
    if len(row) != len(cov):
        # The artifact was fitted against a different feature set than the one
        # being served. Refuse rather than silently truncating to the overlap.
        return None

    matrix = np.asarray(cov, dtype=float)
    vector = np.asarray(row, dtype=float)
    variance = float(vector @ matrix @ vector)
    if variance < 0 or not np.isfinite(variance):
        return None

    p = outcome_model.win_probability(features)
    return float(p * (1.0 - p) * sqrt(variance))


def fold_spread(outcome_model, features: dict[str, float]) -> tuple[float | None, int]:
    """Spread of this game's probability across independently fitted models.

    Prefers the disjoint-era fits over the walk-forward folds, and the reason is
    a mistake worth recording. The folds are nested -- the 2026 fit trains on
    2015-2025, the 2025 fit on 2015-2024 -- so they share over ninety percent of
    their rows and agree with each other almost by construction. Built as the
    conservative estimator, the fold spread came back *smaller* than the delta
    method, which is the signature of measuring nothing.

    The era blocks share no games at all. Where they disagree, the disagreement
    is real. They also absorb genuine change in the sport, so this runs large --
    the safe direction, given that understating sigma both fires more bets and
    stakes each one bigger.

    Returns the sample standard deviation and how many fits contributed. The
    n-1 denominator matters at three or five models.
    """
    blocks = outcome_model.win_probability_blocks(features)
    folds = outcome_model.win_probability_folds(features)
    chosen = blocks if len(blocks) >= 2 else folds
    if len(chosen) < 2:
        # Report the largest ensemble that existed, not the empty one we fell
        # through to -- the count is a diagnostic, and "0 fits" and "1 fit" are
        # different problems.
        return None, max(len(blocks), len(folds))
    values = np.asarray(chosen, dtype=float)
    return float(values.std(ddof=1)), len(chosen)


def calibration_sigma(outcome_model, features: dict[str, float]) -> float | None:
    """Systematic error measured against outcomes, in this prediction's region.

    The other two estimators ask how much the model agrees with itself. This one
    asks whether it is right, by comparing out-of-sample predictions to realized
    frequencies with the binomial component subtracted out.

    Looked up by region rather than pooled, because miscalibration is rarely
    uniform and bets live in the tails, where it is usually worst.

    A bin whose systematic term came back at zero is calibrated to within what
    that much data can resolve -- which is not the same as calibrated. The bin's
    own resolution floor is returned in that case, so a region we cannot measure
    reports the limit of our measurement instead of a confident zero.
    """
    bins = getattr(outcome_model, "win_calibration", None)
    if not bins:
        return None
    p = outcome_model.win_probability(features)

    chosen = None
    for entry in bins:
        if entry["p_low"] <= p <= entry["p_high"]:
            chosen = entry
            break
    if chosen is None:
        # Outside every observed region, so take the nearest bin by centre. A
        # prediction the model has never made before is not one to be confident
        # about, but it still has to be priced with something.
        chosen = min(bins, key=lambda e: abs(e["p_mean"] - p))

    systematic = float(chosen.get("sigma_systematic", 0.0))
    resolution = float(chosen.get("resolution", 0.0))
    return max(systematic, resolution)


def estimate(outcome_model, features: dict[str, float]) -> Sigma:
    """All three estimators, with the largest selected.

    When none can run, `value` falls back to the floor and `measured` is
    False. The caller has to check that: pricing a play against a floor sigma
    would produce a confident-looking number from no information at all.
    """
    delta = delta_se(outcome_model, features)
    folds, n_folds = fold_spread(outcome_model, features)
    calibrated = calibration_sigma(outcome_model, features)

    candidates = [(v, name) for v, name in
                  ((delta, "delta"), (folds, "folds"),
                   (calibrated, "calibration"))
                  if v is not None and np.isfinite(v)]

    if not candidates:
        return Sigma(delta=delta, folds=folds, calibration=calibrated,
                     n_folds=n_folds, value=MINIMUM_SIGMA, source="floor")

    value, source = max(candidates)
    if value < MINIMUM_SIGMA:
        return Sigma(delta=delta, folds=folds, calibration=calibrated,
                     n_folds=n_folds, value=MINIMUM_SIGMA, source="floor")
    return Sigma(delta=delta, folds=folds, calibration=calibrated,
                 n_folds=n_folds, value=value, source=source)
