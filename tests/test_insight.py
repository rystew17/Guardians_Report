"""The deterministic analysis machinery.

This package exists to replace generated prose with computed findings, and its
whole risk is that it produces *plausible* observations that are not true. The
tests below are therefore mostly about restraint: that small samples lose, that
noise stays quiet, and that saying nothing remains a valid outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from guards_report.insight import evaluators, render, select
from guards_report.insight.types import Finding, Reference, significance_floor


def _finding(**kwargs) -> Finding:
    base = dict(
        subject=1, subject_kind="batter", code="bat.rate.hit", family="contact",
        kind="skill", value=0.30,
        reference=Reference(mean=0.25, sd=0.03, population="league"),
        evidence=500, stabilisation=100,
    )
    base.update(kwargs)
    return Finding(**base)


# --------------------------------------------------------------------------
# The multiple-comparisons floor
# --------------------------------------------------------------------------

def test_the_floor_rises_with_the_number_of_criteria():
    """Scanning more criteria must cost more to clear.

    With forty independent noise metrics the chance one clears the 97.5th
    percentile is 87%. Adding evaluators without raising the bar is how this
    fails while appearing to improve.
    """
    assert significance_floor(5) < significance_floor(20) < significance_floor(44)
    assert significance_floor(44) == pytest.approx(2.53, abs=0.02)


def test_the_floor_keeps_expected_false_findings_below_one_in_two():
    from scipy import stats

    for criteria in (5, 20, 44, 60):
        z = significance_floor(criteria)
        expected = criteria * 2 * (1 - stats.norm.cdf(z))
        assert expected <= 0.51


# --------------------------------------------------------------------------
# Reliability shrinkage
# --------------------------------------------------------------------------

def test_a_small_sample_is_shrunk_below_a_large_one():
    """A .400 over twenty at-bats must lose to a .340 over four hundred.

    Extremes live in the smallest samples, so ranking on the raw distance would
    fill the page with September call-ups.
    """
    hot = _finding(value=0.400, evidence=20, stabilisation=475)
    steady = _finding(value=0.340, evidence=400, stabilisation=475)
    assert hot.z_raw > steady.z_raw, "raw distance favours the small sample"
    assert steady.significance > hot.significance, "shrinkage must reverse it"


def test_reliability_is_bounded_and_monotone():
    previous = -1.0
    for evidence in (0, 10, 100, 1000, 10_000):
        r = _finding(evidence=evidence, stabilisation=100).reliability
        assert 0.0 <= r < 1.0
        assert r > previous
        previous = r


def test_a_metric_that_stabilises_faster_keeps_more_of_its_signal():
    """Strikeout rate settles at 55 plate appearances, hit rate at 475."""
    strikeouts = _finding(evidence=300, stabilisation=55)
    hits = _finding(evidence=300, stabilisation=475)
    assert strikeouts.reliability > hits.reliability


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def test_nothing_is_printed_when_nothing_clears_the_floor():
    """An unremarkable player should yield silence, not a paragraph."""
    quiet = [_finding(value=0.252, evidence=400) for _ in range(5)]
    assert select.select(quiet, criteria=5) == []


def test_selection_refuses_two_findings_from_one_family():
    """A player good at everything gets his two most distinctive traits."""
    findings = [
        _finding(value=0.40, family="power", code="a", evidence=900),
        _finding(value=0.39, family="power", code="b", evidence=900),
        _finding(value=0.38, family="contact", code="c", evidence=900),
    ]
    chosen = select.select(findings, criteria=5, limit=3)
    assert len({f.family for f in chosen}) == len(chosen)


def test_selection_is_ordered_by_shrunk_significance():
    findings = [
        _finding(value=0.30, evidence=100, code="small"),
        _finding(value=0.30, evidence=2000, family="power", code="large"),
    ]
    chosen = select.select(findings, criteria=2, limit=2)
    assert chosen[0].code == "large"


def test_the_floor_comes_from_criteria_scanned_not_findings_surviving():
    """The multiple-comparisons cost is paid per criterion tested.

    Passing the surviving count would understate it, and understating it is the
    whole failure mode.
    """
    findings = [_finding(value=0.31, evidence=800)]
    assert select.select(findings, criteria=1, limit=1)
    assert select.select(findings, criteria=200, limit=1) == []


# --------------------------------------------------------------------------
# Contrasts
# --------------------------------------------------------------------------

def test_contrasts_pair_findings_that_point_opposite_ways():
    """Tension is the interesting shape; agreement is not."""
    good = _finding(value=0.34, family="power", evidence=900)
    bad = _finding(value=0.16, family="discipline", evidence=900, code="k")
    pairs = select.contrasts([good, bad], floor=1.0)
    assert pairs and pairs[0][0].direction != pairs[0][1].direction


def test_two_findings_in_the_same_family_are_not_a_contrast():
    a = _finding(value=0.34, family="power", evidence=900)
    b = _finding(value=0.16, family="power", evidence=900, code="b")
    assert select.contrasts([a, b], floor=1.0) == []


# --------------------------------------------------------------------------
# Change-point trends
# --------------------------------------------------------------------------

def test_a_real_change_is_found_with_its_window():
    rng = np.random.default_rng(3)
    series = pd.Series(np.concatenate([
        rng.normal(0.200, 0.15, 60), rng.normal(0.400, 0.15, 30),
    ]))
    found = evaluators.longest_significant_window(series, baseline=0.250)
    assert found is not None
    length, mean, p = found
    assert length >= 15 and mean > 0.30 and p < 0.01


def test_noise_around_the_baseline_produces_no_trend():
    rng = np.random.default_rng(5)
    flat = pd.Series(rng.normal(0.250, 0.15, 120))
    assert evaluators.longest_significant_window(flat, baseline=0.250) is None


def test_false_trends_stay_rare_across_many_players():
    """Trying L5, L10, L15 and L30 and printing the best always finds something.

    Correcting for how many windows were tried is what keeps this honest, and it
    holds the false rate well under the nominal alpha.
    """
    rng = np.random.default_rng(9)
    fired = sum(
        evaluators.longest_significant_window(
            pd.Series(rng.normal(0.250, 0.15, 110)), baseline=0.250
        ) is not None
        for _ in range(200)
    )
    assert fired / 200 <= 0.03


def test_too_short_a_series_yields_nothing():
    assert evaluators.longest_significant_window(
        pd.Series([0.3, 0.4, 0.2]), baseline=0.25
    ) is None


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def test_a_sentence_states_the_number_and_its_reference():
    finding = _finding(
        code="bat.rate.strikeout", family="discipline",
        value=0.10, reference=Reference(mean=0.22, sd=0.04, population="league"),
        evidence=500, stabilisation=55, detail={"rate": 0.10},
    )
    text = render.render(finding)
    assert "10.0%" in text and "22.0%" in text


def test_the_same_player_always_reads_the_same_way():
    finding = _finding(code="bat.rate.strikeout", detail={"rate": 0.10})
    assert render.render(finding) == render.render(finding)


def test_an_unknown_code_renders_nothing_rather_than_raising():
    assert render.render(_finding(code="not.a.real.code")) == ""


def test_two_findings_pointing_opposite_ways_are_joined_with_a_contrast():
    """Built the way `rate_quality` builds them: signed so positive means better.

    A batter's strikeout rate is negated, because a high one is a weakness. That
    sign is what makes a contrast detectable at all -- without it both findings
    read as strengths and the pair is invisible.
    """
    good = _finding(code="bat.rate.hit", detail={"rate": 0.31}, value=0.31)
    bad = _finding(
        code="bat.rate.strikeout", family="discipline", value=-0.30,
        reference=Reference(mean=-0.22, sd=0.04, population="league"),
        detail={"rate": 0.30},
    )
    assert good.direction != bad.direction
    text = render.sentence([good, bad], subject="He")
    assert text.startswith("He ") and ", but " in text


def test_no_findings_produce_no_sentence():
    assert render.sentence([]) == ""
