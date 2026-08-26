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


def pooled_systematic(bins: list[dict[str, float]]) -> float:
    """One systematic-error figure for the model, from all bins at once.

    Per-bin figures cannot be used directly, and the reason is a mistake this
    project made and caught. Each bin's systematic term is a noisy unbiased
    estimate floored at zero, and flooring a noisy unbiased estimate biases it
    upward: whichever bin happened to deviate reports an inflated sigma. Read
    per-bin, this model appeared overconfident by three points in exactly the
    band where a moderate favorite sits -- a finding produced by scanning ten
    bins and keeping the two largest.

    The global test says otherwise. Summing the squared standardized residuals
    gave chi-square 13.62 on 10 degrees of freedom, p about 0.19, with a largest
    single deviation of 2.14 against an expected maximum near 1.96 for ten
    draws. The model is calibrated to within what this much data can resolve.

    Pooling uses the excess of that statistic over its expectation:

        E[sum z^2]  =  df  +  sigma^2 * sum(1 / se_b^2)

    which here returns about 0.009 rather than 0.027. Below the per-bin
    resolution floor, so the floor is what ends up being reported -- the honest
    statement that miscalibration under about 1.4 points is not measurable with
    eleven thousand games, rather than a claim about which regions are worse.
    """
    if not bins:
        return 0.0
    excess = 0.0
    precision = 0.0
    for entry in bins:
        n = float(entry.get("n", 0))
        mean_p = float(entry.get("p_mean", 0.0))
        frequency = float(entry.get("frequency", 0.0))
        if n <= 0 or not (0.0 < mean_p < 1.0):
            continue
        variance = mean_p * (1.0 - mean_p) / n
        if variance <= 0:
            continue
        excess += ((frequency - mean_p) ** 2) / variance - 1.0
        precision += 1.0 / variance
    if precision <= 0 or excess <= 0:
        # Calibrated to within the resolution of the data. Not the same as
        # calibrated, which is why the per-bin floor still applies downstream.
        return 0.0
    return float(sqrt(excess / precision))


def calibration_sigma(outcome_model, features: dict[str, float]) -> float | None:
    """Systematic error measured against outcomes.

    The other two estimators ask how much the model agrees with itself. This one
    asks whether it is right, by comparing out-of-sample predictions to realized
    frequencies with the binomial component subtracted out.

    Pooled across bins rather than read per-region. Regional variation is real in
    principle -- miscalibration is usually worst in the tails, where bets live --
    but claiming to have measured it here would be overreach: see
    `pooled_systematic` for the test that says this model's apparent regional
    structure is noise.

    The bin's own resolution is still the floor, because a region we cannot
    measure should report the limit of our measurement rather than a confident
    zero. In practice that floor is what binds, which makes sigma limited by how
    much history we have rather than by how good the model is.
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

    return max(pooled_systematic(bins), float(chosen.get("resolution", 0.0)))


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


# ---------------------------------------------------------------------------
# The other markets
# ---------------------------------------------------------------------------
# The win model carries its calibration inside the fitted artifact. Everything
# else -- totals, first five, strikeouts, hits, home runs -- is measured by
# scripts/calibrate.py and written to one file beside the models. Same method,
# same decomposition, same floor: what differs is only which model was asked.

CALIBRATION_FILE = "market_calibration.json"

# Keys as the calibration script writes them, mapped from the market names the
# betting side uses.
def _line_key(line: float) -> str:
    """How a line is spelled in the stored record. 4.5 and 4.50 are one line."""
    return f"{float(line):g}"


# Each market reads the record measured on the question it asks. Two markets
# sharing one record is how a first-five *total* came to be priced off a
# measurement of who was *leading* after five -- the same error as pricing an
# 8.5 strikeout bet off a 4.5 record, one market across.
MARKET_KEYS = {
    "total": "total",
    "runline": "runline",
    "f5_moneyline": "first_five",
    "f5_total": "first_five_total",
    "strikeouts": "strikeout",
    "hits": "hit",
    "home_runs": "home_run",
}


def load_market_calibration(root) -> dict:
    """Every market's record, or an empty mapping if it has not been measured.

    Empty is a legitimate state and must not read as agreement: `market_sigma`
    returns None for it, and the guide refuses to stake anything without a
    measured figure.
    """
    from pathlib import Path as _Path
    import json

    target = _Path(root) / "models" / CALIBRATION_FILE
    if not target.is_file():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def market_sigma(
    calibration: dict, market: str, probability: float,
    line: float | None = None,
) -> float | None:
    """Systematic error for one market, at the line and region of this bet.

    Pooled across bins for the same reason the win model's is: a per-bin figure
    is a noisy estimate floored at zero, and flooring one biases it upward, so
    whichever bin happened to deviate would set the standard error. The bin's
    own resolution is the floor, which in practice is what binds.

    Read per line, because the error is not the same at each of them. Measured
    on strikeouts, the model overstates by 1.8 points at 4.5 and understates by
    3.5 at 6.5 and 8.5 -- the opposite direction and twice the size -- while the
    standard error doubles. Returns None when the posted line has no record of
    its own, which the guide treats as unstakeable rather than reaching for a
    neighbour's figure.
    """
    if not calibration:
        return None
    record = calibration.get(MARKET_KEYS.get(market, market))
    if not record:
        return None

    by_line = record.get("by_line")
    if by_line:
        if line is None:
            return None
        chosen_line = by_line.get(_line_key(line))
        if not chosen_line:
            return None
        bins = chosen_line.get("bins") or []
    else:
        bins = record.get("bins") or []
    if not bins:
        return None

    chosen = None
    for entry in bins:
        if entry.get("p_low", 0.0) <= probability <= entry.get("p_high", 1.0):
            chosen = entry
            break
    if chosen is None:
        chosen = min(bins, key=lambda e: abs(e.get("p_mean", 0.5) - probability))

    return max(pooled_systematic(bins), float(chosen.get("resolution", 0.0)),
               MINIMUM_SIGMA)
