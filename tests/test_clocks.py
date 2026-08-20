"""Displayed times are US Eastern, because that is how baseball publishes them."""

from __future__ import annotations

from datetime import datetime, timezone

from guards_report.metrics import clocks


def test_first_pitch_shown_in_eastern_not_utc():
    # 23:10 UTC is a 7:10 PM local first pitch -- and a different calendar day.
    assert clocks.game_time(datetime(2026, 8, 18, 23, 10, tzinfo=timezone.utc)) == "7:10 PM ET"


def test_hour_is_not_zero_padded():
    assert clocks.game_time(datetime(2026, 8, 19, 17, 5, tzinfo=timezone.utc)) == "1:05 PM ET"


def test_naive_input_is_treated_as_utc_not_machine_local():
    naive = datetime(2026, 8, 18, 23, 10)
    aware = datetime(2026, 8, 18, 23, 10, tzinfo=timezone.utc)
    assert clocks.game_time(naive) == clocks.game_time(aware)


def test_daylight_saving_is_resolved_per_date():
    assert clocks.stamp(datetime(2026, 8, 18, 23, 10, tzinfo=timezone.utc)).endswith("EDT")
    assert clocks.stamp(datetime(2026, 1, 5, 1, 0, tzinfo=timezone.utc)).endswith("EST")


def test_iso_strings_are_accepted():
    # Provenance rows carry ISO text, since that is what BigQuery stores.
    assert clocks.stamp("2026-08-18T22:42:11+00:00") == "2026-08-18 06:42 PM EDT"
    assert clocks.stamp("2026-08-18T22:42:11Z") == "2026-08-18 06:42 PM EDT"


def test_missing_or_malformed_values_render_empty():
    for bad in (None, "", "not a timestamp"):
        assert clocks.stamp(bad) == ""
        assert clocks.game_time(bad) == ""
