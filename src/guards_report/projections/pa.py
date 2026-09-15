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


# What the player props read. Handedness drives the platoon split and the home
# club identifies the park, neither of which the talent model needs.
PROP_COLUMNS = [
    "game_date", "game_pk", "at_bat_number", "batter", "pitcher",
    "stand", "p_throws", "events", "batting_team", "season",
    "home_team", "away_team",
]


def cached_pitch_frame(root, columns, cache_name, *, ended_only=False):
    """The pitch corpus, projected to `columns`, read from as few files as it can be.

    Only the current season's thirty files ever change. Every earlier season is
    finished and its rows are fixed forever, so they are concatenated once into
    a single cached file and read as one thereafter -- 360 opens become 31.

    Locally that is seconds. On Cloud Run the corpus is on a GCS mount where the
    price is per file opened rather than per byte, and a refit reads the whole
    corpus three times.

    The cache is rebuilt whenever any source file is newer than it. That is what
    makes it safe rather than merely fast: a club-season can be refetched and
    revised, and serving a copy built before the revision would hide the
    correction behind a file that looks perfectly valid.

    `ended_only` keeps just the pitches that finished a plate appearance, which
    is the plate-appearance view; without it the frame is every pitch.
    """
    from pathlib import Path

    root = Path(root)
    files = sorted((root / "pitches").glob("*.parquet"))
    if not files:
        return None

    def read(path):
        frame = pd.read_parquet(path, columns=columns)
        return _ended(frame) if ended_only else frame

    current = max(int(path.name[:4]) for path in files)
    prior = [p for p in files if int(p.name[:4]) < current]
    live = [p for p in files if int(p.name[:4]) == current]

    frames = []
    if prior:
        store = root / "models" / cache_name
        newest = max(p.stat().st_mtime for p in prior)
        if store.exists() and store.stat().st_mtime >= newest:
            frames.append(pd.read_parquet(store))
        else:
            history = pd.concat([read(p) for p in prior], ignore_index=True)
            store.parent.mkdir(parents=True, exist_ok=True)
            history.to_parquet(store, index=False, compression="zstd")
            frames.append(history)

    frames.extend(read(p) for p in live)
    return pd.concat(frames, ignore_index=True)


def _ended(frame):
    """Only the pitches that finished a plate appearance."""
    return frame[frame["events"].notna() & (frame["events"] != "")]


def _cache_path(cache_dir, columns: list[str]):
    """Where the finished seasons are kept, for this exact column set.

    Keyed by the columns because the callers ask for different projections --
    PROP_COLUMNS and TALENT_COLUMNS -- and a cache built for one would
    otherwise serve the wrong columns to the other.
    """
    import hashlib
    from pathlib import Path

    key = hashlib.sha256("|".join(sorted(columns)).encode()).hexdigest()[:12]
    return Path(cache_dir).parent / "models" / f"pa_cache_{key}.parquet"


def load(
    cache_dir, *, columns: list[str] | None = None, seasons=None, cache=True
) -> pd.DataFrame:
    """Plate appearances read straight from the cached chunks.

    Reads only the requested columns from each Parquet file rather than loading
    everything and selecting afterwards. Parquet is columnar, so the unread
    columns are never touched -- which on this corpus turns a four-minute,
    ten-gigabyte load into seconds.

    Asked for every season, the finished ones come from a single cached file
    rather than three hundred and thirty separate reads. A refit loads the whole
    corpus three times, and on Cloud Run -- where the corpus is on a GCS mount
    and the cost is per file opened, not per byte -- that was six minutes for
    the props stage against forty seconds on a laptop.

    The cache is rebuilt whenever any source file is newer than it, which is
    what makes it safe rather than merely fast: a club-season can be refetched
    and revised, and serving a copy built before the revision would hide the
    correction behind a file that looks perfectly valid.

    Asked for specific seasons the cache is skipped entirely -- that path reads
    thirty files already and is what a report build uses.
    """
    from pathlib import Path

    cache_dir = Path(cache_dir)
    columns = columns or TALENT_COLUMNS
    if "events" not in columns:
        columns = columns + ["events"]

    files = sorted(cache_dir.glob("*.parquet"))
    frames = []

    if seasons is None and cache and files:
        current = max(int(path.name[:4]) for path in files)
        prior = [path for path in files if int(path.name[:4]) < current]
        live = [path for path in files if int(path.name[:4]) == current]

        if prior:
            store = _cache_path(cache_dir, columns)
            newest = max(path.stat().st_mtime for path in prior)
            if store.exists() and store.stat().st_mtime >= newest:
                frames.append(pd.read_parquet(store))
            else:
                history = pd.concat(
                    [_ended(pd.read_parquet(path, columns=columns))
                     for path in prior],
                    ignore_index=True,
                )
                store.parent.mkdir(parents=True, exist_ok=True)
                history.to_parquet(store, index=False, compression="zstd")
                frames.append(history)
        files = live

    for path in files:
        if seasons is not None and int(path.name[:4]) not in seasons:
            continue
        frames.append(_ended(pd.read_parquet(path, columns=columns)))

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
