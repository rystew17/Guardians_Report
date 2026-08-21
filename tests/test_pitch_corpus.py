"""Guarantees for the pitch corpus loader and its durable storage.

The in-season refresh is the piece most likely to corrupt quietly. It appends to
a file it has already written, so a mistake does not raise -- it leaves a corpus
with a missing week or a doubled game, and every talent estimate downstream
inherits it without complaint.

Network calls are replaced throughout; these test the assembly logic, not
Savant.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from guards_report.projections import lineup, pitches
from guards_report.publish import datasets


def _rows(team: str, season: int, days: list[int], *, value: str = "single"):
    """Pitches for one club on the given days of May."""
    out = []
    for day in days:
        for ab in range(1, 4):
            for pitch in range(1, 3):
                out.append({
                    "game_date": date(season, 5, day),
                    "game_pk": season * 1000 + day,
                    "at_bat_number": ab,
                    "pitch_number": pitch,
                    "batter": 100 + ab,
                    "pitcher": 900,
                    "events": value if pitch == 2 else None,
                    "batting_team": team,
                    "season": season,
                })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Typing
# --------------------------------------------------------------------------

def test_numeric_columns_are_stored_as_numbers():
    """Everything arrives as text; storing it that way triples the corpus."""
    frame = pd.DataFrame({
        "release_speed": ["93.6", "94.1", ""],
        "pitch_name": ["4-Seam Fastball", "Slider", "Curveball"],
        "game_pk": ["745123", "745123", "745124"],
    })
    typed = pitches._typed(frame)
    assert pd.api.types.is_numeric_dtype(typed["release_speed"])
    assert pd.api.types.is_numeric_dtype(typed["game_pk"])
    assert not pd.api.types.is_numeric_dtype(typed["pitch_name"])


def test_a_mostly_text_column_survives_a_stray_number():
    """A description that happens to be numeric once must not become a float."""
    frame = pd.DataFrame({"description": ["called_strike", "ball", "7", "foul"]})
    typed = pitches._typed(frame)
    assert not pd.api.types.is_numeric_dtype(typed["description"])


# --------------------------------------------------------------------------
# In-season refresh
# --------------------------------------------------------------------------

@pytest.fixture
def cache(tmp_path: Path) -> Path:
    return tmp_path / "pitches"


def test_first_refresh_writes_the_season_so_far(monkeypatch, cache):
    monkeypatch.setattr(
        pitches, "fetch_range",
        lambda team, season, *, start, end, **kw: _rows(team, season, [1, 2, 3]),
    )
    monkeypatch.setattr(pitches, "REQUEST_PAUSE", 0)

    added = pitches.refresh_current_season(
        2026, ["CLE"], cache_dir=cache, through=date(2026, 5, 3), verbose=False
    )
    assert added["CLE"] == 18
    assert len(pd.read_parquet(cache / "2026_CLE.parquet")) == 18


def test_refresh_appends_only_what_is_new(monkeypatch, cache):
    monkeypatch.setattr(pitches, "REQUEST_PAUSE", 0)
    monkeypatch.setattr(
        pitches, "fetch_range",
        lambda team, season, *, start, end, **kw: _rows(team, season, [1, 2]),
    )
    pitches.refresh_current_season(
        2026, ["CLE"], cache_dir=cache, through=date(2026, 5, 2), verbose=False
    )

    seen: dict = {}

    def later(team, season, *, start, end, **kw):
        seen["start"] = start
        return _rows(team, season, [2, 3, 4])

    monkeypatch.setattr(pitches, "fetch_range", later)
    added = pitches.refresh_current_season(
        2026, ["CLE"], cache_dir=cache, through=date(2026, 5, 4), verbose=False
    )

    # Refetched from the last day on disk, not from the start of the season.
    assert seen["start"] == date(2026, 5, 2)
    assert added["CLE"] == 12          # two new days, six pitches each
    stored = pd.read_parquet(cache / "2026_CLE.parquet")
    assert len(stored) == 24
    assert not stored.duplicated(["game_pk", "at_bat_number", "pitch_number"]).any()


def test_running_twice_in_a_day_changes_nothing(monkeypatch, cache):
    monkeypatch.setattr(pitches, "REQUEST_PAUSE", 0)
    monkeypatch.setattr(
        pitches, "fetch_range",
        lambda team, season, *, start, end, **kw: _rows(team, season, [1, 2]),
    )
    for _ in range(3):
        pitches.refresh_current_season(
            2026, ["CLE"], cache_dir=cache, through=date(2026, 5, 2), verbose=False
        )
    stored = pd.read_parquet(cache / "2026_CLE.parquet")
    assert len(stored) == 12


def test_a_revised_pitch_replaces_the_stored_one(monkeypatch, cache):
    """Statcast reclassifies recent games; the newer copy has to win.

    Skipping the overlap day would be cheaper and would freeze the first version
    of every game permanently.
    """
    monkeypatch.setattr(pitches, "REQUEST_PAUSE", 0)
    monkeypatch.setattr(
        pitches, "fetch_range",
        lambda team, season, *, start, end, **kw: _rows(team, season, [1]),
    )
    pitches.refresh_current_season(
        2026, ["CLE"], cache_dir=cache, through=date(2026, 5, 1), verbose=False
    )
    assert set(pd.read_parquet(cache / "2026_CLE.parquet")["events"].dropna()) == {"single"}

    monkeypatch.setattr(
        pitches, "fetch_range",
        lambda team, season, *, start, end, **kw: _rows(team, season, [1], value="double"),
    )
    pitches.refresh_current_season(
        2026, ["CLE"], cache_dir=cache, through=date(2026, 5, 1), verbose=False
    )
    stored = pd.read_parquet(cache / "2026_CLE.parquet")
    assert set(stored["events"].dropna()) == {"double"}
    assert len(stored) == 6, "a revision must replace, not duplicate"


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

def test_object_names_mirror_the_local_layout(tmp_path):
    path = tmp_path / "pitches" / "2024_CLE.parquet"
    assert datasets._object_name(tmp_path, path) == "data/pitches/2024_CLE.parquet"


def test_local_files_finds_every_corpus(tmp_path):
    for name in ("corpus", "pitches", "models"):
        (tmp_path / name).mkdir()
    (tmp_path / "corpus" / "2024.parquet").write_bytes(b"x")
    (tmp_path / "pitches" / "2024_CLE.parquet").write_bytes(b"y")
    (tmp_path / "models" / "game_outcome.json").write_text("{}")

    found = {p.name for p in datasets.local_files(tmp_path)}
    assert found == {"2024.parquet", "2024_CLE.parquet", "game_outcome.json"}


def test_dry_run_never_touches_the_network(tmp_path):
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "a.parquet").write_bytes(b"x")
    result = datasets.sync(tmp_path, bucket_name="", project="", dry_run=True)
    assert result.uploaded == 0 and result.skipped == 1


# --------------------------------------------------------------------------
# Era-dependent lineup weighting
# --------------------------------------------------------------------------

def test_slot_weights_are_measured_per_season():
    """The universal designated hitter changed what the ninth slot is worth.

    Measured on the corpus, the ninth spot took 2.87 plate appearances in 2016
    against 3.40 in 2024 -- a pitcher pinch-hit for early versus a real hitter.
    A single vector across the whole corpus describes neither.
    """
    frames = []
    for season, turns in ((2016, 2), (2024, 3)):
        for day in range(1, 4):
            order = list(range(200, 209))
            sequence = order * turns + ([] if season == 2016 else order[:2])
            for n, batter in enumerate(sequence, start=1):
                frames.append({
                    "game_date": date(season, 5, day),
                    "game_pk": season * 100 + day,
                    "at_bat_number": n, "pitch_number": 1,
                    "batter": batter, "pitcher": 900,
                    "events": "single", "batting_team": "AAA", "season": season,
                })
    corpus = pd.DataFrame(frames)
    by_season = lineup.slot_weights_by_season(corpus)

    assert set(by_season) == {2016, 2024}
    assert len(by_season[2016]) == 9
    # The season with a partial extra turn gives its top slots more turns.
    assert by_season[2024][0] > by_season[2016][0]
