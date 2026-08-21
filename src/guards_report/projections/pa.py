"""Plate-appearance view over the pitch corpus.

A plate appearance is not a separate download. Savant returns the full pitch
stream for a team-season whatever you ask for, so the PA table is derived here
rather than fetched: the last pitch of each at-bat carries the outcome, the run
value and the expected run value, and every other pitch in the at-bat carries
the process that produced it.

Keeping the two as one dataset with two views matters beyond saving a pull. It
guarantees the talent model and the pitch-level models are looking at the same
plate appearances -- if they were fetched separately, a Savant revision between
the two pulls would put them quietly out of step.
"""

from __future__ import annotations

import pandas as pd

from guards_report.projections.pitches import (  # noqa: F401  (re-exported)
    FIRST_SEASON,
    build as build_pitches,
    fetch_chunk,
    season_teams,
)

# The columns a plate-appearance row needs. Everything else on the pitch is
# process detail that the PA-level talent model has no use for.
COLUMNS = [
    "game_date", "game_pk", "at_bat_number", "batter", "pitcher",
    "stand", "p_throws", "events",
    "woba_value", "woba_denom", "estimated_woba_using_speedangle",
    "home_team", "away_team", "inning", "inning_topbot",
    "outs_when_up", "on_1b", "on_2b", "on_3b", "delta_run_exp",
    "batting_team", "season",
]


def from_pitches(pitch_corpus: pd.DataFrame) -> pd.DataFrame:
    """One row per plate appearance, taken from the pitch that ended it.

    `events` is populated only on the final pitch of an at-bat, which makes it
    the natural selector -- and a stricter one than taking the last pitch by
    number, since an at-bat interrupted by a caught stealing or the end of an
    inning never records an event at all and should not become a PA row.
    """
    if pitch_corpus.empty:
        return pd.DataFrame(columns=COLUMNS)

    ending = pitch_corpus[
        pitch_corpus["events"].notna() & (pitch_corpus["events"] != "")
    ]
    available = [c for c in COLUMNS if c in ending.columns]
    frame = ending[available].copy().reset_index(drop=True)

    assert not frame.duplicated(["game_pk", "at_bat_number"]).any(), (
        "a plate appearance must end exactly once"
    )
    return frame


def pitch_count(pitch_corpus: pd.DataFrame) -> pd.DataFrame:
    """Pitches seen per plate appearance, for models that care how it got there.

    A six-pitch strikeout and a first-pitch strikeout are the same event and
    different plate appearances, and the difference matters for pitch-count
    driven questions like how long a starter will last.
    """
    return (
        pitch_corpus.groupby(["game_pk", "at_bat_number"], sort=False)
        .size().rename("pitches").reset_index()
    )


# Columns the talent model reads. Projecting to these at read time is the
# difference between a 10 GB frame and a manageable one: the corpus keeps all
# 121 columns because re-fetching is expensive, but no single model wants them
# all, and pandas holds what it reads.
TALENT_COLUMNS = [
    "game_date", "game_pk", "at_bat_number", "batter", "pitcher",
    "stand", "p_throws", "events", "woba_value", "woba_denom",
    "estimated_woba_using_speedangle", "batting_team", "season",
]


def load(
    cache_dir, *, columns: list[str] | None = None, seasons=None
) -> pd.DataFrame:
    """Plate appearances read straight from the cached chunks.

    Reads only the requested columns from each Parquet file rather than loading
    everything and selecting afterwards. Parquet is columnar, so the unread
    columns are never touched -- which on this corpus turns a four-minute,
    ten-gigabyte load into seconds.
    """
    from pathlib import Path

    cache_dir = Path(cache_dir)
    columns = columns or TALENT_COLUMNS
    if "events" not in columns:
        columns = columns + ["events"]

    frames = []
    for path in sorted(cache_dir.glob("*.parquet")):
        if seasons is not None and int(path.name[:4]) not in seasons:
            continue
        frame = pd.read_parquet(path, columns=columns)
        frames.append(frame[frame["events"].notna() & (frame["events"] != "")])

    if not frames:
        return pd.DataFrame(columns=columns)

    corpus = pd.concat(frames, ignore_index=True)
    corpus["game_date"] = pd.to_datetime(corpus["game_date"]).dt.date
    corpus = corpus.sort_values(
        ["game_date", "game_pk", "at_bat_number"]
    ).reset_index(drop=True)

    assert not corpus.duplicated(["game_pk", "at_bat_number"]).any(), (
        "a plate appearance must end exactly once"
    )
    return corpus
