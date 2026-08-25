"""How wrong our own probability might be.

The whole feature turns on this number, and the first two attempts at it were
wrong in the same direction for the same reason: they measured how much the
model agrees with itself. A model can be perfectly self-consistent and perfectly
wrong, so a third estimator compares predictions against outcomes, and it is
allowed to overrule the other two.

The third attempt was wrong as well, and more instructively. Read per bin, the
calibration table appeared to show this model overstating favorites by three
points in the 0.56 to 0.61 band -- exactly where a moderate favorite sits, and
exactly where a bet would land. That finding did not survive its own test:
chi-square 13.62 on 10 degrees of freedom, p about 0.19, largest deviation 2.14
against an expected maximum near 1.96 for ten draws. It was ten bins scanned and
the two largest kept, which is precisely the selection effect this feature exists
to stop us making with money.

So sigma is pooled across the table, the per-bin resolution is the floor, and
the tests below pin both of those rather than any particular bin's number.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from guards_report.betting import uncertainty
from guards_report.projections import backtest, model

ARTIFACT = Path(__file__).resolve().parents[1] / "data" / "models" / "game_outcome.json"


def _model(**overrides) -> model.OutcomeModel:
    """A two-feature model with an identity-ish covariance, easy to reason about."""
    kwargs = dict(
        win_columns=["a", "b"],
        win_coef=[0.5, 0.5],
        win_intercept=0.0,
        win_mean=[0.0, 0.0],
        win_scale=[1.0, 1.0],
        win_cov=[[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]],
    )
    kwargs.update(overrides)
    return model.OutcomeModel(**kwargs)


FEATURES = {"a": 0.0, "b": 0.0}


# ---------------------------------------------------------------------------
# The delta method
# ---------------------------------------------------------------------------

def test_the_delta_method_needs_a_covariance_and_says_so_without_one():
    """An artifact fitted before covariances were stored must read as "not
    measurable", never as certainty."""
    assert uncertainty.delta_se(_model(win_cov=[]), FEATURES) is None


def test_a_covariance_of_the_wrong_size_is_refused():
    """The artifact was fitted against a different feature set than the one
    being served. Truncating to the overlap would produce a number."""
    wrong = _model(win_cov=[[0.01, 0.0], [0.0, 0.01]])
    assert uncertainty.delta_se(wrong, FEATURES) is None


def test_the_delta_standard_error_is_positive_and_small():
    value = uncertainty.delta_se(_model(), FEATURES)
    assert value is not None and 0.0 < value < 0.5


def test_a_larger_coefficient_covariance_gives_a_larger_standard_error():
    tight = _model(win_cov=[[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]])
    loose = _model(win_cov=[[0.10, 0, 0], [0, 0.10, 0], [0, 0, 0.10]])
    assert (uncertainty.delta_se(loose, FEATURES)
            > uncertainty.delta_se(tight, FEATURES))


# ---------------------------------------------------------------------------
# The spread across independent fits
# ---------------------------------------------------------------------------

def _with_blocks(intercepts):
    return _model(
        win_block_labels=[f"b{i}" for i in range(len(intercepts))],
        win_block_coef=[[0.5, 0.5]] * len(intercepts),
        win_block_intercept=list(intercepts),
        win_block_mean=[[0.0, 0.0]] * len(intercepts),
        win_block_scale=[[1.0, 1.0]] * len(intercepts),
    )


def test_models_that_disagree_more_produce_a_wider_spread():
    close, _ = uncertainty.fold_spread(_with_blocks([0.0, 0.02, -0.02]), FEATURES)
    far, _ = uncertainty.fold_spread(_with_blocks([0.0, 0.60, -0.60]), FEATURES)
    assert far > close


def test_a_single_fit_cannot_produce_a_spread():
    value, n = uncertainty.fold_spread(_with_blocks([0.1]), FEATURES)
    assert value is None and n == 1


def test_the_disjoint_eras_are_preferred_over_the_nested_folds():
    """The mistake this file exists to prevent repeating.

    Walk-forward folds are nested -- consecutive fits share over ninety percent
    of their rows -- so their spread measures almost nothing. Built as the
    conservative estimator, it came back smaller than the delta method.
    """
    both = _model(
        win_block_labels=["x", "y"],
        win_block_coef=[[0.5, 0.5]] * 2,
        win_block_intercept=[0.0, 1.0],          # wide
        win_block_mean=[[0.0, 0.0]] * 2,
        win_block_scale=[[1.0, 1.0]] * 2,
        win_fold_coef=[[0.5, 0.5]] * 2,
        win_fold_intercept=[0.0, 0.001],         # nearly identical
        win_fold_mean=[[0.0, 0.0]] * 2,
        win_fold_scale=[[1.0, 1.0]] * 2,
    )
    wide, _ = uncertainty.fold_spread(both, FEATURES)
    assert wide > 0.1, "the era blocks must be the ones consulted"


def test_the_nested_folds_are_still_used_when_no_blocks_exist():
    """Backward compatibility with the artifact fitted between these two
    changes -- it has folds and no blocks."""
    folds_only = _model(
        win_fold_coef=[[0.5, 0.5]] * 3,
        win_fold_intercept=[0.0, 0.5, -0.5],
        win_fold_mean=[[0.0, 0.0]] * 3,
        win_fold_scale=[[1.0, 1.0]] * 3,
    )
    value, n = uncertainty.fold_spread(folds_only, FEATURES)
    assert value is not None and n == 3


# ---------------------------------------------------------------------------
# Calibration — the estimator that looks outside
# ---------------------------------------------------------------------------

def _calibrated(bins):
    return _model(win_calibration=bins)


BINS = [
    {"n": 1000, "p_low": 0.30, "p_high": 0.48, "p_mean": 0.40,
     "frequency": 0.40, "sigma_systematic": 0.000, "resolution": 0.015},
    {"n": 1000, "p_low": 0.48, "p_high": 0.70, "p_mean": 0.58,
     "frequency": 0.55, "sigma_systematic": 0.030, "resolution": 0.015},
]


def test_one_deviant_bin_does_not_set_sigma_on_its_own():
    """The correction this module needed, and the reason for pooling.

    A per-bin systematic term is a noisy unbiased estimate floored at zero, and
    flooring one biases it upward -- so whichever bin happened to deviate
    reports an inflated figure. Read per-bin, the real model looked overconfident
    by three points exactly where a moderate favorite sits. The global test put
    that at chi-square 13.62 on 10 df, p about 0.19: ten bins scanned and the
    two largest kept, which is the selection effect this whole feature exists to
    avoid making with money.
    """
    value = uncertainty.calibration_sigma(_calibrated(BINS), FEATURES)
    assert value < 0.030, "a single bin's deviation must not drive sigma"
    assert value == pytest.approx(0.015, abs=1e-3), "the resolution floor binds"


def test_the_pooled_estimate_reads_the_whole_table_not_one_row():
    """Every bin deviating in the same direction is a different claim from one
    bin deviating, and must produce a bigger number."""
    one_bad = uncertainty.pooled_systematic(BINS)
    all_bad = uncertainty.pooled_systematic([
        dict(entry, frequency=entry["p_mean"] - 0.03) for entry in BINS])
    assert all_bad > one_bad


def test_a_calibrated_table_pools_to_nothing():
    """Predictions matching frequencies leave only noise, and subtracting it
    must not leave a residue to stake against."""
    clean = [dict(entry, frequency=entry["p_mean"]) for entry in BINS]
    assert uncertainty.pooled_systematic(clean) == 0.0
    assert uncertainty.pooled_systematic([]) == 0.0


def test_the_resolution_floor_is_never_undercut_by_pooling():
    """Zero systematic error means "we cannot detect any", not "there is none".

    Reporting zero would claim a precision a thousand games cannot support, and
    the claim lands directly on the stake.
    """
    low = dict(BINS[0]); low.update(p_low=0.0, p_high=1.0, frequency=0.40)
    value = uncertainty.calibration_sigma(_calibrated([low]), FEATURES)
    assert value == pytest.approx(0.015)


def test_an_artifact_without_calibration_says_so():
    assert uncertainty.calibration_sigma(_model(), FEATURES) is None


def test_a_prediction_outside_every_measured_region_still_gets_a_floor():
    """A prediction the model has never made before is not one to be confident
    about, but it still has to be priced with something."""
    far = [dict(BINS[1], p_low=0.90, p_high=0.99, p_mean=0.95)]
    value = uncertainty.calibration_sigma(_calibrated(far), FEATURES)
    assert value is not None and value > 0


# ---------------------------------------------------------------------------
# Combining them
# ---------------------------------------------------------------------------

def test_the_largest_estimate_wins():
    combined = uncertainty.estimate(_calibrated(BINS), FEATURES)
    assert combined.measured
    assert combined.value == max(
        v for v in (combined.delta, combined.folds, combined.calibration)
        if v is not None)


def test_measuring_against_outcomes_can_overrule_measuring_against_ourselves():
    """The reason this module carries a third estimator.

    The delta method sees 27,000 games and reports about 0.005, which is a true
    statement about the coefficients and a misleading one about tonight. When
    the calibration data says we are further off than that, it has to be the
    figure that reaches the stake.
    """
    tight = _model(
        win_cov=[[1e-6, 0, 0], [0, 1e-6, 0], [0, 0, 1e-6]],
        win_calibration=BINS)
    combined = uncertainty.estimate(tight, FEATURES)
    assert combined.delta < combined.calibration
    assert combined.value == pytest.approx(combined.calibration)
    assert combined.source == "calibration"


def test_nothing_measurable_falls_to_the_floor_and_admits_it():
    """Pricing against a floor sigma would turn no information at all into a
    confident-looking number."""
    bare = model.OutcomeModel(
        win_columns=["a"], win_coef=[0.5], win_mean=[0.0], win_scale=[1.0])
    combined = uncertainty.estimate(bare, {"a": 0.0})
    assert combined.source == "floor"
    assert combined.value == uncertainty.MINIMUM_SIGMA
    assert not combined.measured


def test_the_floor_is_never_undercut():
    tiny = _calibrated([dict(BINS[0], p_low=0.0, p_high=1.0,
                             sigma_systematic=0.0, resolution=0.0)])
    assert uncertainty.estimate(tiny, FEATURES).value >= uncertainty.MINIMUM_SIGMA


# ---------------------------------------------------------------------------
# The decomposition itself
# ---------------------------------------------------------------------------

def test_a_perfectly_calibrated_model_shows_no_systematic_error():
    """Predictions that match reality leave only binomial noise behind, and
    subtracting it must not leave a residue."""
    import numpy as np
    rng = np.random.default_rng(11)
    p = rng.uniform(0.35, 0.65, 40_000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    bins = backtest.calibration_bins(y, p)
    assert bins
    assert max(b["sigma_systematic"] for b in bins) < 0.02


def test_a_model_that_overstates_favorites_is_caught():
    """The actual defect in our own model, reproduced deliberately."""
    import numpy as np
    rng = np.random.default_rng(12)
    p = rng.uniform(0.35, 0.65, 40_000)
    truth = np.where(p > 0.55, p - 0.05, p)      # overconfident above 0.55
    y = (rng.uniform(size=p.size) < truth).astype(float)
    bins = backtest.calibration_bins(y, p)
    top = [b for b in bins if b["p_mean"] > 0.57]
    assert top and max(b["sigma_systematic"] for b in top) > 0.02


def test_the_noise_floor_is_reported_alongside_every_bin():
    """So a caller can say what could not be measured rather than implying a
    precision the data does not support."""
    import numpy as np
    rng = np.random.default_rng(13)
    p = rng.uniform(0.4, 0.6, 5_000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    for entry in backtest.calibration_bins(y, p):
        assert entry["resolution"] > 0
        assert entry["resolution"] == pytest.approx(
            (entry["p_mean"] * (1 - entry["p_mean"]) / entry["n"]) ** 0.5)


def test_too_little_data_yields_no_bins_rather_than_noisy_ones():
    import numpy as np
    assert backtest.calibration_bins(np.array([1.0, 0.0]), np.array([0.5, 0.5])) == []


# ---------------------------------------------------------------------------
# The shipped artifact
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not ARTIFACT.is_file(), reason="model not fitted here")
def test_the_fitted_artifact_carries_everything_the_pricing_needs():
    fitted = model.load(ARTIFACT)
    assert fitted is not None
    assert fitted.win_cov, "no covariance: the delta method cannot run"
    assert len(fitted.win_block_coef) >= 2, "fewer than two eras to compare"
    assert fitted.win_calibration, "no calibration: sigma would be self-referential"

    features = {c: v for c, v in zip(fitted.win_columns, fitted.win_mean)}
    combined = uncertainty.estimate(fitted, features)
    assert combined.measured
    assert combined.value > uncertainty.MINIMUM_SIGMA, (
        "a sigma at the floor on a fitted model means nothing measured it")


@pytest.mark.skipif(not ARTIFACT.is_file(), reason="model not fitted here")
def test_the_eras_used_for_the_spread_do_not_overlap():
    """Nested spans were the original defect. If these ever start overlapping,
    the spread quietly stops measuring anything again."""
    fitted = model.load(ARTIFACT)
    spans = []
    for label in fitted.win_block_labels:
        start, end = label.split("-")
        spans.append((int(start), int(end)))
    spans.sort()
    for (_, first_end), (second_start, _) in zip(spans, spans[1:]):
        assert first_end < second_start, f"eras overlap: {spans}"
