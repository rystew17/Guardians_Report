"""The lineup block reaches the win model, and fails safe when it cannot.

This block is wired across two files that are fitted and served months apart:
`train.fit` names the columns and `predict` fills them. A name that matches on
one side and not the other does not raise -- `win_probability` reads a missing
feature as the fitted mean and returns a perfectly ordinary number. The failure
is a projection that silently ignores who is playing, which is the whole point
of the feature.

Three things are pinned here:

  * the serving path offers every lineup column the fitted model asks for
  * a card too incomplete to value falls back to the fitted mean, not to zero,
    because zero is a real value on this scale and a low one
  * the value is a slot-weighted mean of batter effects, so it stays on the
    centred scale training built the column from
"""

from __future__ import annotations

import numpy as np
import pytest

from guards_report.projections import features, model as model_module, predict


def test_the_lineup_columns_the_model_asks_for_are_the_ones_predict_fills():
    """The two sides of the wiring agree on the names."""
    served = {"home_lineup", "away_lineup"}
    assert set(features.LINEUP_COLUMNS) == served
    # And the labels exist, or the contribution chart prints a raw column name
    # at the reader.
    for column in features.LINEUP_COLUMNS:
        assert column in predict.FEATURE_LABELS


def test_the_lineup_block_is_not_inside_the_core_block():
    """Kept separate on purpose, so the report can say which is which.

    Every other column in the core block clears its own standard error. This one
    does not, and folding it in would erase the distinction.
    """
    for column in features.LINEUP_COLUMNS:
        assert column not in features.CORE_COLUMNS


def _model(**kwargs):
    return model_module.OutcomeModel(
        win_columns=["elo_logit", "home_lineup", "away_lineup"],
        win_coef=[1.0, 4.0, -4.0],
        win_intercept=0.0,
        win_mean=[0.0, 0.02, 0.02],
        win_scale=[1.0, 0.01, 0.01],
        **kwargs,
    )


def test_a_missing_lineup_lands_on_the_fitted_mean_not_on_zero():
    """Zero is a real value on this scale, and a bad one.

    Batter effects are deviations from league average, so an absent lineup
    written as 0.0 is not "unknown" -- it is "league average", and paired with
    the other club's real value it invents an advantage nobody has. Reading it
    as the fitted mean leaves the projection where the rest of the block puts
    it.
    """
    m = _model()
    known = m.win_probability(
        {"elo_logit": 0.0, "home_lineup": 0.02, "away_lineup": 0.02})
    absent = m.win_probability({"elo_logit": 0.0})
    nan = m.win_probability(
        {"elo_logit": 0.0, "home_lineup": float("nan"),
         "away_lineup": float("nan")})

    assert known == pytest.approx(0.5, abs=1e-9)
    assert absent == pytest.approx(known, abs=1e-9)
    assert nan == pytest.approx(known, abs=1e-9)

    # And a zero would NOT have been neutral, which is why the fallback matters.
    zeroed = m.win_probability(
        {"elo_logit": 0.0, "home_lineup": 0.0, "away_lineup": 0.02})
    assert zeroed < known - 0.05


def test_one_club_batting_better_moves_the_projection_its_way():
    m = _model()
    even = m.win_probability(
        {"elo_logit": 0.0, "home_lineup": 0.02, "away_lineup": 0.02})
    home_stronger = m.win_probability(
        {"elo_logit": 0.0, "home_lineup": 0.03, "away_lineup": 0.02})
    away_stronger = m.win_probability(
        {"elo_logit": 0.0, "home_lineup": 0.02, "away_lineup": 0.03})
    assert home_stronger > even > away_stronger


class _Talent:
    """Batter effects, on the centred scale `running_scores` produces."""

    def __init__(self, scores):
        self._scores = scores

    def batter_score(self, batter):
        return self._scores.get(int(batter))


def test_lineup_value_is_the_slot_weighted_mean_of_batter_effects():
    """Arithmetic, not a figure copied from a run.

    Weights 2 and 1 across nine slots: the first batter counts twice. With the
    leadoff man at 0.10 and the other eight at 0.01, the value is
    (2*0.10 + 8*0.01) / 10 = 0.028.
    """
    weights = (2.0,) + (1.0,) * 8
    scores = {1: 0.10, **{i: 0.01 for i in range(2, 10)}}
    value = predict.lineup_value(
        list(range(1, 10)), _Talent(scores), weights=weights)
    assert value == pytest.approx((2 * 0.10 + 8 * 0.01) / 10.0)


def test_an_unknown_batter_counts_as_league_average_rather_than_breaking():
    """A September call-up with no fitted effect must not void the whole card."""
    weights = (1.0,) * 9
    value = predict.lineup_value(
        list(range(1, 10)), _Talent({i: 0.02 for i in range(1, 9)}),
        weights=weights)
    # Eight hitters at 0.02 and one unknown read as 0.0.
    assert value == pytest.approx(8 * 0.02 / 9.0)


def test_a_card_too_short_to_value_returns_nothing():
    """Better no number than a number built from five hitters."""
    assert predict.lineup_value([1, 2, 3], _Talent({}), weights=(1.0,) * 9) is None
    assert predict.lineup_value([], _Talent({}), weights=(1.0,) * 9) is None
    assert predict.lineup_value(None, _Talent({}), weights=(1.0,) * 9) is None


def test_the_fitted_column_and_the_served_one_are_on_the_same_scale():
    """The skew this whole block is most likely to die of.

    Training builds the column from `running_scores`, which are deviations from
    league average centred near zero. If the serving path ever added the
    intercept back the served value would sit a few tenths above anything the
    fit saw, and with a large coefficient that is not a small drift. The guard
    is that a lineup of exactly league-average hitters values to zero.
    """
    weights = (1.0,) * 9
    value = predict.lineup_value(
        list(range(1, 10)), _Talent({i: 0.0 for i in range(1, 10)}),
        weights=weights)
    assert value == pytest.approx(0.0, abs=1e-12)
    assert abs(value) < 0.05


def test_the_design_matrix_imputes_a_missing_lineup_to_the_column_mean():
    """The training-side counterpart of the serving fallback."""
    import pandas as pd

    frame = pd.DataFrame({
        "elo_logit": [0.1, -0.2, 0.3, 0.0],
        "home_lineup": [0.02, 0.04, np.nan, 0.06],
        "away_lineup": [0.03, 0.03, 0.03, 0.03],
        "home_win": [1.0, 0.0, 1.0, 0.0],
    })
    x, y = features.design_matrix(
        frame, ["elo_logit", "home_lineup", "away_lineup"])
    assert not np.isnan(x).any()
    # The imputed cell is the mean of the three observed values.
    assert x[2, 1] == pytest.approx((0.02 + 0.04 + 0.06) / 3.0)
    assert list(y) == [1.0, 0.0, 1.0, 0.0]


def _card(pk, team, batters, day, season=2024):
    """One team-game of plate appearances, in batting order."""
    import pandas as pd

    return pd.DataFrame({
        "game_pk": pk,
        "batting_team": team,
        "batter": batters,
        "at_bat_number": range(1, len(batters) + 1),
        "game_date": pd.Timestamp(f"{season}-04-{day:02d}"),
        "season": season,
    })


def _corpus(cards):
    import pandas as pd

    return pd.concat(cards, ignore_index=True)


def test_a_projected_lineup_never_sees_the_game_it_projects():
    """The guarantee the whole feature rests on.

    If the projection could see tonight's card it would be tonight's card, the
    held-out gain would be the posted-lineup number, and none of it would
    survive contact with a morning build.
    """
    from guards_report.projections import lineup

    regulars = list(range(1, 10))
    intruder = [99] + list(range(2, 10))
    corpus = _corpus([
        _card(1, "AAA", regulars, 1),
        _card(2, "AAA", regulars, 2),
        _card(3, "AAA", intruder, 3),      # 99 starts, once, today
    ])
    proj = lineup.projected_lineups(corpus)

    today = proj[proj["game_pk"] == 3]
    assert 99 not in set(today["batter"]), "the projection read today's card"
    assert today.sort_values("slot")["batter"].tolist() == regulars


def test_a_club_with_no_history_gets_no_projection():
    """Better absent than invented; `design_matrix` imputes the column mean."""
    from guards_report.projections import lineup

    corpus = _corpus([_card(1, "AAA", list(range(1, 10)), 1)])
    assert lineup.projected_lineups(corpus).empty


def test_the_projection_takes_each_slot_s_most_frequent_occupant():
    """Two of three starts leads the slot; a one-off does not."""
    from guards_report.projections import lineup

    regulars = list(range(1, 10))
    rested = [50] + list(range(2, 10))     # 50 leads off once
    corpus = _corpus([
        _card(1, "AAA", regulars, 1),
        _card(2, "AAA", rested, 2),
        _card(3, "AAA", regulars, 3),
        _card(4, "AAA", regulars, 4),
    ])
    proj = lineup.projected_lineups(corpus)
    leadoff = proj[(proj["game_pk"] == 4) & (proj["slot"] == 1)]
    assert leadoff["batter"].tolist() == [1]


def test_no_batter_is_listed_in_two_slots():
    """A regular who moves in the order would otherwise appear twice, and one
    slot would come back empty -- nine weights over eight hitters."""
    from guards_report.projections import lineup

    corpus = _corpus([
        _card(1, "AAA", [1, 2, 3, 4, 5, 6, 7, 8, 9], 1),
        _card(2, "AAA", [2, 1, 3, 4, 5, 6, 7, 8, 9], 2),   # 1 and 2 swap
        _card(3, "AAA", [2, 1, 3, 4, 5, 6, 7, 8, 9], 3),
        _card(4, "AAA", [1, 2, 3, 4, 5, 6, 7, 8, 9], 4),
    ])
    card = lineup.projected_lineups(corpus)
    card = card[card["game_pk"] == 4]
    assert len(card) == len(set(card["batter"])), "a batter was listed twice"
    assert sorted(card["slot"]) == list(range(1, 10))


def test_the_projection_has_the_same_shape_as_a_posted_card():
    """`_add_pa_block` values both through one function, which only works if the
    two frames are interchangeable."""
    from guards_report.projections import lineup

    corpus = _corpus([
        _card(1, "AAA", list(range(1, 10)), 1),
        _card(2, "AAA", list(range(1, 10)), 2),
    ])
    posted = lineup.starting_lineups(corpus)
    proj = lineup.projected_lineups(corpus)
    assert list(posted.columns) == list(proj.columns)
