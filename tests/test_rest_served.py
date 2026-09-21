"""Rest reaches the model as a measurement, not as a zero.

`team_rest_diff` and `sp_rest_diff` were literal 0.0 in the serving path while
training fitted them on real spread, so the coefficients were only ever
multiplied by zero. The caps here are the ones `features.team_rest` and
`features.starter_rest` impute, because serving a number from a different
distribution than the fit saw is the same bug wearing a better disguise.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from guards_report.projections import predict


def _plate(rows):
    return pd.DataFrame(rows, columns=["game_date", "batting_team", "pitcher"])


def test_team_rest_counts_days_since_the_club_last_played():
    plate = _plate([
        ("2026-04-10", "CLE", 1), ("2026-04-12", "CLE", 2),
    ])
    assert predict.rest_days(plate, on=date(2026, 4, 15), team="CLE") == 3.0


def test_team_rest_is_capped_at_six_exactly_as_training_caps_it():
    plate = _plate([("2026-04-01", "CLE", 1)])
    assert predict.rest_days(plate, on=date(2026, 5, 1), team="CLE") == 6.0


def test_pitcher_rest_is_clipped_to_the_training_range():
    plate = _plate([("2026-04-01", "CLE", 7), ("2026-04-29", "CLE", 7)])
    assert predict.rest_days(plate, on=date(2026, 4, 30), pitcher=7) == 1.0
    assert predict.rest_days(plate, on=date(2026, 6, 1), pitcher=7) == 10.0


def test_today_s_own_game_never_counts_as_rest():
    """A row dated today would otherwise read as zero days of rest."""
    plate = _plate([("2026-04-12", "CLE", 1), ("2026-04-15", "CLE", 1)])
    assert predict.rest_days(plate, on=date(2026, 4, 15), team="CLE") == 3.0


def test_missing_history_falls_back_to_what_training_imputes():
    assert predict.rest_days(None, on=date(2026, 4, 15), team="CLE") == 6.0
    assert predict.rest_days(None, on=date(2026, 4, 15), pitcher=9) == 5.0
    empty = _plate([])
    assert predict.rest_days(empty, on=date(2026, 4, 15), team="XXX") == 6.0
    assert predict.rest_days(empty, on=date(2026, 4, 15), pitcher=99) == 5.0


def test_an_unknown_starter_does_not_invent_a_rest_advantage():
    """Both sides fall back to the same value, so the difference is zero."""
    plate = _plate([("2026-04-12", "CLE", 1)])
    home = predict.rest_days(plate, on=date(2026, 4, 15), pitcher=None)
    away = predict.rest_days(plate, on=date(2026, 4, 15), pitcher=None)
    assert home - away == 0.0
