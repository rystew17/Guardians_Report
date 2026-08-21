"""Training and prediction must compute the same feature.

This exists because they did not. `own_lineup` was built during training as a
slot-weighted mean of batter *effects* -- deviations centred near zero -- while
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
