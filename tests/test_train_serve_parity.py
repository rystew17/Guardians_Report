"""Training and prediction must compute the same feature.

This exists because they did not. `own_lineup` was built during training as a
slot-weighted mean of batter *effects* -- deviations centered near zero -- while
prediction returned the absolute run value, which is the league mean plus the
same deviation. Roughly four tenths larger, against a fitted coefficient of
+4.4, which multiplied projected runs by four and produced a nineteen-run game.

It was caught only because nineteen runs is absurd on its face. A feature
mismatched by ten percent rather than a factor of four would have shipped
looking plausible, so the parity is asserted here rather than left to eyesight.
"""

from __future__ import annotations

import numpy as np
import pytest

from guards_report.projections import lineup, predict, talent


@pytest.fixture
def fitted():
    model = talent.Talent(
        batter={10: 0.030, 11: -0.020, 12: 0.005, 13: 0.0, 14: -0.011,
                15: 0.018, 16: -0.004, 17: 0.009, 18: -0.015},
        pitcher={90: -0.012},
        platoon={"RR": -0.009, "LR": 0.010},
        intercept=0.3267,
        alpha=800.0,
        batter_pa={i: 400 for i in range(10, 19)},
    )
    return model


def test_lineup_value_is_a_deviation_not_an_absolute_rate(fitted):
    """The served feature must sit on the same scale training fitted on."""
    card = list(range(10, 19))
    served = predict.lineup_value(card, fitted, weights=lineup.DEFAULT_SLOT_PA)

    weights = lineup.DEFAULT_SLOT_PA
    expected = sum(w * fitted.batter[b] for w, b in zip(weights, card)) / sum(weights)

    assert served == pytest.approx(expected)
    # The failure mode: anywhere near the league mean means the intercept crept
    # back in and every projected score is multiplied several-fold.
    assert abs(served) < 0.05, "lineup value drifted onto the absolute scale"
    assert abs(served - fitted.intercept) > 0.2


def test_a_short_card_returns_nothing_rather_than_a_partial_average(fitted):
    assert predict.lineup_value([10, 11, 12], fitted,
                                weights=lineup.DEFAULT_SLOT_PA) is None
    assert predict.lineup_value([], fitted, weights=lineup.DEFAULT_SLOT_PA) is None
    assert predict.lineup_value(None, fitted, weights=lineup.DEFAULT_SLOT_PA) is None


def test_better_hitters_raise_the_value(fitted):
    weak = predict.lineup_value(list(range(10, 19)), fitted,
                                weights=lineup.DEFAULT_SLOT_PA)
    strong_model = talent.Talent(
        batter={b: v + 0.05 for b, v in fitted.batter.items()},
        pitcher=fitted.pitcher, platoon=fitted.platoon,
        intercept=fitted.intercept, alpha=fitted.alpha,
        batter_pa=fitted.batter_pa,
    )
    strong = predict.lineup_value(list(range(10, 19)), strong_model,
                                  weights=lineup.DEFAULT_SLOT_PA)
    assert strong == pytest.approx(weak + 0.05)


def test_the_top_of_the_order_counts_for_more(fitted):
    """Slot weighting has to bite, or moving a hot bat up means nothing."""
    star, scrub = 10, 11
    card = [star] + [13] * 7 + [scrub]
    flipped = [scrub] + [13] * 7 + [star]
    lead = predict.lineup_value(card, fitted, weights=lineup.DEFAULT_SLOT_PA)
    trail = predict.lineup_value(flipped, fitted, weights=lineup.DEFAULT_SLOT_PA)
    assert lead > trail


def test_unknown_batters_count_as_league_average(fitted):
    """An unrecognised hitter is average, not worthless."""
    known = predict.lineup_value(list(range(10, 19)), fitted,
                                 weights=lineup.DEFAULT_SLOT_PA)
    with_stranger = predict.lineup_value(
        list(range(10, 18)) + [999_999], fitted, weights=lineup.DEFAULT_SLOT_PA
    )
    assert with_stranger is not None
    # Replacing a below-average hitter with an average one raises the value.
    assert with_stranger > known


# --------------------------------------------------------------------------
# Starter features
# --------------------------------------------------------------------------

def test_innings_are_divided_by_appearances_not_starts():
    """The third train/serve skew found in this project, and the worst.

    Training builds `ip_per_start` from a log carrying relief outings as well --
    `starts_prior` counts every prior row, not every prior start -- so the
    fitted feature is innings per *appearance*, averaging 3.72 and topping out
    at 8.11. Dividing by starts instead looks more correct and is not: a
    swingman with seventeen appearances and five starts returned 14.27, beyond
    anything in training, and the model extrapolated to a fourteen-point error
    in the win probability.
    """
    from guards_report.projections.predict import _starter_features

    class Box:
        # 17 appearances, 5 of them starts, 71.3 innings.
        season = {"outs": 214, "games": 17, "gamesStarted": 5,
                  "fip": 4.7, "kPct": 0.199, "bbPct": 0.042}

    stats = _starter_features(Box())
    assert stats["ip_per_start"] == pytest.approx(214 / 3 / 17, abs=1e-9)
    assert stats["ip_per_start"] < 8.11, "outside the fitted range"


def test_a_true_starter_is_unaffected_by_the_denominator():
    """Where starts equal appearances the two definitions agree, which is why
    the bug survived: it is invisible on every everyday starter."""
    from guards_report.projections.predict import _starter_features

    class Box:
        season = {"outs": 377, "games": 26, "gamesStarted": 26, "fip": 3.9}

    assert _starter_features(Box())["ip_per_start"] == pytest.approx(377 / 3 / 26)


def test_the_model_flags_a_feature_outside_its_fitted_range():
    """The guard that would have caught it without anyone reading a chart."""
    from guards_report.projections.model import OutcomeModel

    model = OutcomeModel(
        win_columns=["sp_ip_per_start"], win_coef=[0.08],
        win_mean=[3.72], win_scale=[1.99], win_intercept=0.1,
    )
    assert not model.out_of_range({"sp_ip_per_start": 5.0})
    flagged = model.out_of_range({"sp_ip_per_start": 14.27})
    assert flagged and flagged[0]["name"] == "sp_ip_per_start"
    assert flagged[0]["sigma"] > 5


def test_a_missing_feature_is_not_flagged_as_out_of_range():
    """Absence is handled by mean imputation and is not an anomaly."""
    from guards_report.projections.model import OutcomeModel

    model = OutcomeModel(
        win_columns=["sp_ip_per_start"], win_coef=[0.08],
        win_mean=[3.72], win_scale=[1.99],
    )
    assert not model.out_of_range({"sp_ip_per_start": None})
    assert not model.out_of_range({})


def test_fit_metrics_are_not_restated_in_prose():
    """A refit rewrites the artifact and cannot rewrite a comment.

    `model.py` once advertised held-out log loss 0.67668 and +4.30pt while the
    artifact beside it measured 0.67834 and +3.99pt -- prose describing a better
    model than the one actually running. Nothing read those numbers, so the
    drift was invisible until someone quoted them.

    Any figure precise enough to go stale belongs in the artifact, which is
    written at fit time and is what the report reads.
    """
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1]
              / "src" / "guards_report" / "projections" / "model.py")
    text = source.read_text(encoding="utf-8")
    head = text.split('"""')[1] if '"""' in text else ""

    # Four or more decimal places is a measurement, not a round design constant.
    quoted = re.findall(r"\b0\.\d{4,}\b", head)
    assert not quoted, (
        "fit metrics restated in the module docstring, where a refit cannot "
        f"update them: {quoted}")
