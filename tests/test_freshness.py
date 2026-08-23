"""Keeping every corpus current, and admitting when one is not.

This module exists because the same bug has now appeared four times in this
project wearing different clothes: Elo ratings frozen at the previous September,
a talent model fitted on completed seasons and never carried forward, a pull
loop whose range stopped short of the current year, and a derived starter table
that was refreshed only when the model happened to be retrained.

None of them raised. In every case the report rendered, the numbers looked
plausible, and they described a league that had moved on. So the tests here are
about the two things that make that failure survivable: noticing, and saying so.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from guards_report.projections import freshness


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _corpus(tmp_path, name: str, through: str, season: int = 2026):
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "game_date": pd.to_datetime([through]),
        "season": [season],
        "game_pk": [1],
    })
    stem = f"{season}_04" if name == "pitches" else f"{name}_{season}"
    frame.to_parquet(directory / f"{stem}.parquet")
    return directory


def _artifact(tmp_path, name: str, days_old: int):
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc) - timedelta(days=days_old)
    (models / name).write_text(json.dumps({"fitted_at": stamp.isoformat()}))


def _full(tmp_path, through: str = "2026-08-21"):
    for name in ("corpus", "pitchers", "pitches"):
        _corpus(tmp_path, name, through)
    _corpus(tmp_path, "models", through)
    (tmp_path / "models" / "models_2026.parquet").rename(
        tmp_path / "models" / "f5_starter.parquet"
    )
    # The graded reference populations, stamped with how far the corpus behind
    # them reached. They are read at serve time and can freeze like any other
    # derived table, so a fixture without them is not a current install.
    pd.DataFrame({
        "batter": [1], "season": [2026], "pa": [400],
        "built_through": pd.to_datetime([through]),
    }).to_parquet(tmp_path / "models" / "batter_profiles.parquet")
    return tmp_path


# ---------------------------------------------------------------------------
# Noticing
# ---------------------------------------------------------------------------

def test_a_survey_reports_what_each_corpus_actually_reaches(tmp_path):
    _corpus(tmp_path, "corpus", "2026-08-21")
    _corpus(tmp_path, "pitchers", "2026-08-19")
    _corpus(tmp_path, "pitches", "2026-08-20")
    standing = freshness.survey(tmp_path, 2026)
    assert standing.corpus_through == date(2026, 8, 21)
    assert standing.pitchers_through == date(2026, 8, 19)
    assert standing.pitches_through == date(2026, 8, 20)


def test_a_missing_corpus_reads_as_unknown_rather_than_as_today(tmp_path):
    """The distinction that matters on opening day.

    `None` means "nothing here"; a date means "this is what it reaches". If an
    absent corpus reported today's date, the first report of the season would
    declare itself current with no data behind it at all.
    """
    standing = freshness.survey(tmp_path, 2026)
    assert standing.corpus_through is None
    assert standing.stale_days(date(2026, 8, 22))["corpus"] is None
    assert not standing.is_current(date(2026, 8, 22))


def test_one_day_behind_is_current_because_tonights_game_has_not_been_played(tmp_path):
    _full(tmp_path, through="2026-08-21")
    assert freshness.survey(tmp_path, 2026).is_current(date(2026, 8, 22))


def test_two_days_behind_is_not_current(tmp_path):
    _full(tmp_path, through="2026-08-20")
    assert not freshness.survey(tmp_path, 2026).is_current(date(2026, 8, 22))


def test_one_stale_corpus_is_enough_to_fail_the_check(tmp_path):
    """Everything downstream reads from disk and assumes the disk is current.

    A check that passed on three of four would hand that assumption to a model
    reading the fourth.
    """
    _full(tmp_path, through="2026-08-21")
    _corpus(tmp_path, "pitchers", "2026-07-01")
    assert not freshness.survey(tmp_path, 2026).is_current(date(2026, 8, 22))


def test_the_derived_table_counts_as_a_corpus(tmp_path):
    """It is read at serve time, so it can be stale in exactly the same way.

    Leaving it out of the check is what let it sit three days behind while the
    survey reported everything current -- and the survey is what decides whether
    a refresh runs at all, so the staleness was self-perpetuating.
    """
    _full(tmp_path, through="2026-08-21")
    (tmp_path / "models" / "f5_starter.parquet").unlink()
    standing = freshness.survey(tmp_path, 2026)
    assert standing.derived_through is None
    assert not standing.is_current(date(2026, 8, 22))
    assert "derived" in standing.stale_days(date(2026, 8, 22))


# ---------------------------------------------------------------------------
# Leakage — worse than being behind
# ---------------------------------------------------------------------------

def test_a_corpus_reaching_past_the_game_being_projected_is_a_warning(tmp_path, monkeypatch):
    """Being ahead is worse than being behind: the model would see its answer.

    A stale report is wrong by a couple of days. A leaking one is right for a
    reason that will not survive contact with tomorrow.
    """
    _full(tmp_path, through="2026-08-25")
    monkeypatch.setattr(freshness, "rebuild_derived", lambda root: None)
    for name in ("corpus", "pitchers", "pitches"):
        monkeypatch.setattr(
            freshness, {"corpus": "corpus"}.get(name, name), _Stub(), raising=False
        )
    result = freshness.refresh_all(
        tmp_path, on=date(2026, 8, 22), verbose=False, skip_pitches=True
    )
    assert any("would see its own outcome" in w for w in result.warnings)


class _Stub:
    """Stands in for a fetching module: every call is a no-op that succeeds."""

    def __getattr__(self, name):
        def call(*args, **kwargs):
            return pd.DataFrame({"game_type": [], "season": []})
        return call


def test_negative_staleness_is_reported_as_negative_not_absolute(tmp_path):
    _full(tmp_path, through="2026-08-25")
    days = freshness.survey(tmp_path, 2026).stale_days(date(2026, 8, 22))
    assert days["corpus"] == -3
    assert not freshness.survey(tmp_path, 2026).is_current(date(2026, 8, 22))


# ---------------------------------------------------------------------------
# Fitted artifacts — deliberately fixed, not quietly ancient
# ---------------------------------------------------------------------------

def test_an_artifacts_age_comes_from_its_own_stamp_not_the_file(tmp_path):
    """A sync, a restore or a copy resets mtime without the model changing.

    This project restores its artifacts from cloud storage, so mtime would say
    every model was fitted the day of the last restore.
    """
    _artifact(tmp_path, "game_outcome.json", days_old=30)
    # UTC on both sides, matching what the function does. A local 
    # here would make this test pass or fail depending on the hour it runs.
    ages = freshness.fitted_ages(
        tmp_path / "models", now=datetime.now(timezone.utc).date())
    assert ages["game_outcome"] == 30


def test_an_unreadable_stamp_is_skipped_rather_than_fatal(tmp_path):
    models = tmp_path / "models"
    models.mkdir(parents=True)
    (models / "props.json").write_text("{ not json")
    (models / "first5.json").write_text(json.dumps({"fitted_at": "not a date"}))
    _artifact(tmp_path, "game_outcome.json", days_old=2)
    ages = freshness.fitted_ages(models)
    assert ages == {"game_outcome": 2}


def test_a_missing_artifact_is_absent_rather_than_reported_as_age_zero(tmp_path):
    """Age zero would read as "fitted today", which is the opposite of the truth."""
    (tmp_path / "models").mkdir(parents=True)
    assert freshness.fitted_ages(tmp_path / "models") == {}


def test_a_fit_older_than_a_fortnight_warns(tmp_path, monkeypatch):
    _full(tmp_path, through="2026-08-21")
    _artifact(tmp_path, "game_outcome.json", days_old=freshness.REFIT_AFTER_DAYS + 1)
    monkeypatch.setattr(freshness, "rebuild_derived", lambda root: 0)

    result = freshness.refresh_all(
        tmp_path, on=date.today(), verbose=False, skip_pitches=True
    )
    assert any("should be refitted" in w for w in result.warnings)


def test_a_recent_fit_does_not_warn(tmp_path, monkeypatch):
    _full(tmp_path, through="2026-08-21")
    _artifact(tmp_path, "game_outcome.json", days_old=1)
    monkeypatch.setattr(freshness, "rebuild_derived", lambda root: 0)

    result = freshness.refresh_all(
        tmp_path, on=date.today(), verbose=False, skip_pitches=True
    )
    assert not any("refitted" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Cost, and failing softly
# ---------------------------------------------------------------------------

def test_a_second_report_the_same_day_fetches_nothing(tmp_path):
    """Otherwise generating twice refetches thirty team-seasons to learn nothing."""
    _full(tmp_path, through="2026-08-21")
    result = freshness.refresh_all(tmp_path, on=date(2026, 8, 22), verbose=False)
    assert result.refreshed == ["already current"]
    assert result.requests == 0


def test_one_corpus_failing_leaves_the_others_refreshed(tmp_path, monkeypatch):
    """A report on slightly stale pitch data is worth having.

    One that failed to build because Savant was briefly unreachable is not.
    """
    _full(tmp_path, through="2026-08-01")

    def explode(*args, **kwargs):
        raise ConnectionError("savant is down")

    monkeypatch.setattr(freshness.corpus, "build", explode)
    monkeypatch.setattr(freshness, "rebuild_derived", lambda root: 12)

    result = freshness.refresh_all(
        tmp_path, on=date(2026, 8, 22), verbose=False, skip_pitches=True
    )
    assert any("savant is down" in w for w in result.warnings)
    assert "derived" in result.refreshed


def test_the_derived_rebuild_failing_is_a_warning_not_an_exception(tmp_path, monkeypatch):
    _full(tmp_path, through="2026-08-01")

    def explode(root):
        raise ValueError("corrupt parquet")

    monkeypatch.setattr(freshness, "rebuild_derived", explode)
    monkeypatch.setattr(freshness.corpus, "build", lambda *a, **k: pd.DataFrame(
        {"game_type": [], "season": []}))

    result = freshness.refresh_all(
        tmp_path, on=date(2026, 8, 22), verbose=False, skip_pitches=True
    )
    assert any("corrupt parquet" in w for w in result.warnings)


def test_no_pitch_corpus_means_no_derived_table_rather_than_an_empty_one(tmp_path):
    """An empty table would look like a league in which no starter has a record."""
    (tmp_path / "pitches").mkdir(parents=True)
    assert freshness.rebuild_derived(tmp_path) is None


def test_the_line_names_every_corpus_it_tracks(tmp_path):
    _full(tmp_path, through="2026-08-21")
    line = freshness.survey(tmp_path, 2026).line()
    for name in ("corpus", "pitchers", "pitches", "derived"):
        assert name in line
