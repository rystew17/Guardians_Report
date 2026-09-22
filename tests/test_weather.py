"""First-pitch weather reaches the runs model, and fails safe when it cannot.

Weather is the one game-day input that cleared the bar when seven were tested
together. These pin the plumbing that makes it usable: roofs are applied the
same way in the fit and at serve time, both sides of a game read one set of
conditions, and a missing forecast is an average night -- never a zero-degree
one, which on this scale would be a large and entirely invented effect.

No network here. The cache is seeded in a temporary directory, and the one
function that would reach out returns before it has to.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from guards_report.projections import model as model_module
from guards_report.projections import weather


def _seed(root, venues, history=None):
    d = root / "weather"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(venues).to_parquet(d / "venues.parquet", index=False)
    if history is not None:
        pd.DataFrame(history).to_parquet(d / "history.parquet", index=False)


VENUES = [
    {"venue_id": 5, "venue_name": "Open Park", "lat": 41.5, "lon": -81.7,
     "roof": "Open", "tz_id": "America/New_York", "tz_offset": -5},
    {"venue_id": 12, "venue_name": "Dome Park", "lat": 27.8, "lon": -82.7,
     "roof": "Dome", "tz_id": "America/New_York", "tz_offset": -5},
    {"venue_id": 99, "venue_name": "Nowhere", "lat": None, "lon": None,
     "roof": "Open", "tz_id": "Europe/London", "tz_offset": 0},
]


def test_a_dome_is_played_at_room_temperature_and_calm():
    temp, wind = weather.apply_roof([95.0, 40.0], [20.0, 30.0], ["Dome", "Open"])
    assert temp[0] == weather.DOME_TEMP_F and wind[0] == 0.0
    assert temp[1] == 40.0 and wind[1] == 30.0


def test_a_retractable_roof_keeps_the_outdoor_reading():
    """Its state on the night is in no available source, so nothing is invented."""
    temp, wind = weather.apply_roof([88.0], [12.0], ["Retractable"])
    assert temp[0] == 88.0 and wind[0] == 12.0


def test_both_sides_of_a_game_read_the_same_conditions(tmp_path):
    _seed(tmp_path, VENUES, [
        {"game_pk": 1, "temp_f": 81.0, "wind_mph": 9.0, "venue_id": 5, "roof": "Open"},
    ])
    games = pd.DataFrame({"game_pk": [1], "season": [2026],
                          "game_date": ["2026-07-01"], "venue_id": [5]})
    data = pd.DataFrame({"game_pk": [1, 1], "is_home": [1, 0], "runs": [4, 3]})
    out = weather.attach(data, tmp_path, games, fetch=False)
    assert out["temp_f"].tolist() == [81.0, 81.0]
    assert out["wind_mph"].tolist() == [9.0, 9.0]


def test_the_fit_applies_the_roof_exactly_as_serving_does(tmp_path):
    """Train and serve must agree, or a dome night is two different inputs."""
    _seed(tmp_path, VENUES, [
        {"game_pk": 2, "temp_f": 97.0, "wind_mph": 18.0, "venue_id": 12, "roof": "Dome"},
    ])
    games = pd.DataFrame({"game_pk": [2], "season": [2026],
                          "game_date": ["2026-07-01"], "venue_id": [12]})
    data = pd.DataFrame({"game_pk": [2], "is_home": [1], "runs": [5]})
    fitted = weather.attach(data, tmp_path, games, fetch=False)
    served = weather.forecast(tmp_path, 12, datetime(2026, 7, 1, 23, tzinfo=timezone.utc))
    assert fitted["temp_f"].iloc[0] == served["temp_f"] == weather.DOME_TEMP_F
    assert fitted["wind_mph"].iloc[0] == served["wind_mph"] == 0.0


def test_a_game_with_no_reading_stays_missing_rather_than_zero(tmp_path):
    _seed(tmp_path, VENUES, [])
    games = pd.DataFrame({"game_pk": [3], "season": [2026],
                          "game_date": ["2026-07-01"], "venue_id": [5]})
    data = pd.DataFrame({"game_pk": [3], "is_home": [1], "runs": [2]})
    out = weather.attach(data, tmp_path, games, fetch=False)
    assert out["temp_f"].isna().all()


def test_a_forecast_needs_a_park_a_time_and_coordinates(tmp_path):
    _seed(tmp_path, VENUES)
    t = datetime(2026, 7, 1, 23, tzinfo=timezone.utc)
    assert weather.forecast(tmp_path, None, t) is None
    assert weather.forecast(tmp_path, 5, None) is None
    # A park with no coordinates cannot be forecast; None, not a guess.
    assert weather.forecast(tmp_path, 99, t) is None


def test_venue_lookup_accepts_a_numpy_array(tmp_path):
    """`array or []` raises; the lookup has to take what callers really pass."""
    _seed(tmp_path, VENUES)
    got = weather.venue_facts(tmp_path, np.array([5, 12]))
    assert {5, 12} <= set(got["venue_id"])


def _runs_model():
    return model_module.OutcomeModel(
        score_columns=["temp_f", "wind_mph"],
        score_coef=[0.00228, 0.00197],
        score_mean=[75.0, 8.0],
        league_rpg=4.5,
    )


def test_warmer_air_means_more_runs_at_serve_time():
    m = _runs_model()
    cold = m.expected_runs({"temp_f": 45.0, "wind_mph": 8.0})
    warm = m.expected_runs({"temp_f": 95.0, "wind_mph": 8.0})
    assert warm > cold
    # 50 degrees at the fitted slope is about 12%, the size the evidence showed.
    assert warm / cold == pytest.approx(np.exp(0.00228 * 50), rel=1e-9)


def test_no_forecast_is_an_average_night_not_a_frozen_one():
    m = _runs_model()
    missing = m.expected_runs({"temp_f": None, "wind_mph": None})
    average = m.expected_runs({"temp_f": 75.0, "wind_mph": 8.0})
    frozen = m.expected_runs({"temp_f": 0.0, "wind_mph": 0.0})
    assert missing == pytest.approx(average)
    assert frozen < average * 0.9      # which is why zero must never stand in


def test_the_weather_columns_are_the_ones_serving_fills():
    """The fit names them; `predict` fills them. A mismatch raises nothing and
    silently serves average weather forever."""
    import inspect

    from guards_report.projections import predict

    source = inspect.getsource(predict.project)
    for column in weather.WEATHER_COLUMNS:
        assert f'"{column}"' in source


def test_the_weather_cache_is_synced_to_cloud_storage():
    """A refit run in the cloud reads the bucket, not this machine.

    Without this the cloud refit would find no weather history, fall below the
    coverage threshold, and fit the runs model without the one input that
    earned its place -- with every other test still passing.
    """
    from guards_report.publish import datasets

    assert "weather" in datasets.DATASETS
