"""Clock times, expressed in the ballpark's league time zone.

Every source in this project reports timestamps in UTC. Nobody reads a game
preview in UTC: a 7:10 first pitch is 23:10 the same evening, or the following
day's date once it crosses midnight, which reads as an error even though the
underlying value is right.

All displayed times are therefore US Eastern. That is the league's own
convention -- schedules, broadcasts and standings are all published on Eastern
-- so it stays stable regardless of where the report is generated or read, and
one report always agrees with another. Stored values remain UTC; this module is
a display concern only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Windows carries no IANA database, so this resolves only because `tzdata` is a
# declared dependency. Failing loudly here is correct: a silent fallback to UTC
# would relabel every time in the report while still calling it Eastern.
EASTERN = ZoneInfo("America/New_York")


def to_eastern(moment: datetime | str | None) -> datetime | None:
    """Convert to Eastern, treating a naive value as UTC.

    Accepts an ISO string as well as a datetime: provenance rows keep their
    timestamps as ISO text because that is the shape BigQuery stores, and they
    still need to be displayed like every other time.

    Naive datetimes come from sources that omit the offset. They are always UTC
    in practice, and assuming so is safer than letting the local machine's zone
    decide -- that would render the same report differently on two computers.
    """
    if moment is None:
        return None
    if isinstance(moment, str):
        text = moment.strip()
        if not text:
            return None
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(EASTERN)


def game_time(moment: datetime | str | None) -> str:
    """First pitch, as it appears on a schedule: '7:10 PM ET'."""
    local = to_eastern(moment)
    if local is None:
        return ""
    # %I zero-pads the hour on every platform, and "07:10 PM" is not how a
    # start time is written.
    return f"{local.strftime('%I:%M %p').lstrip('0')} ET"


def stamp(moment: datetime | str | None) -> str:
    """A full timestamp for audit lines: '2026-08-18 06:42 PM EDT'.

    Uses the concrete abbreviation rather than "ET" so a reader can tell
    whether daylight time was in effect when the fetch happened.
    """
    local = to_eastern(moment)
    return local.strftime("%Y-%m-%d %I:%M %p %Z") if local else ""


def day_and_time(moment: datetime | str | None) -> str:
    """Weekday, date and first pitch together: 'Tue, Aug 18 · 7:10 PM ET'."""
    local = to_eastern(moment)
    if local is None:
        return ""
    return f"{local.strftime('%a, %b %d')} · {game_time(moment)}"
