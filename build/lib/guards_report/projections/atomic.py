"""Writing an artifact so a failed write cannot cost you the old one.

Every fitted model here is written straight over its predecessor. That is fine
until a refit dies halfway -- out of memory, a lost network handle, the machine
sleeping through a scheduled task -- and then `game_outcome.json` is a truncated
file that the next report reads.

The failure is worse than it sounds. `model.load` returns None on an unreadable
artifact rather than raising, because a report without projections is still a
report. So a half-written model does not announce itself: the projections page
simply disappears, quietly, on a morning nobody is watching.

Writing to a neighbouring temporary file and renaming it over the target fixes
this. On both Windows and POSIX `os.replace` is atomic within a filesystem, so a
reader sees either the whole old file or the whole new one and never a partial.
A crash before the rename leaves the previous artifact untouched, which is
exactly the outcome a scheduled refit needs.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any


def write_text(path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Replace `path` with `text`, all at once or not at all."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # The temporary file has to share a filesystem with the target, or the
    # rename stops being atomic and becomes a copy. Writing it into the same
    # directory is what guarantees that.
    handle, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding=encoding, newline="") as stream:
            stream.write(text)
            stream.flush()
            # Without this the rename can land before the bytes do, which turns
            # a crash-safe write back into a truncated one on a power loss.
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        # Includes KeyboardInterrupt: a refit stopped by hand should leave the
        # working model in place rather than a stray temp file beside it.
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def write_frame(frame: Any, path: Path, **kwargs: Any) -> Path:
    """The same guarantee for a parquet artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    handle, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    os.close(handle)
    try:
        frame.to_parquet(temporary, **kwargs)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path
