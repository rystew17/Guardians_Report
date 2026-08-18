"""Client for Baseball Savant (baseballsavant.mlb.com).

Supplies the Statcast layer: expected stats, pitch arsenals, and percentile
rankings. We deliberately use Savant's *pre-aggregated* leaderboards rather
than pitch-level data. A scouting report displays xwOBA, barrel rate, whiff
rate and pitch mix -- all of which these endpoints give directly, in a few
hundred kilobytes, already computed by the source. Pulling ~700k pitch rows a
season to recompute them would cost roughly 500 MB of warehouse for numbers we
would then have to defend as matching Savant's anyway.

Every leaderboard supports `csv=true`, which is far more stable to parse than
the HTML pages. Note that the CSV header includes a column literally named
`last_name, first_name` -- containing a comma -- so these must be parsed with a
real CSV reader, never split on commas.
"""

from __future__ import annotations

import csv
import io
from typing import Any

from guards_report.config import SAVANT_BASE
from guards_report.sources.http import Archiver, FetchResult, fetch

SOURCE = "baseball_savant"

# Leaderboard identifiers, as they appear in the URL path.
LB_EXPECTED_STATISTICS = "expected_statistics"
LB_PITCH_ARSENAL = "pitch-arsenal-stats"
LB_PERCENTILE_RANKINGS = "percentile-rankings"

TYPE_BATTER = "batter"
TYPE_PITCHER = "pitcher"


def _leaderboard(
    name: str, archiver: Archiver, params: dict[str, Any]
) -> FetchResult:
    return fetch(
        f"{SAVANT_BASE}/leaderboard/{name}",
        source=SOURCE,
        archiver=archiver,
        params={**params, "csv": "true"},
    )


def parse_csv(result: FetchResult) -> list[dict[str, str]]:
    """Parse a Savant CSV response into row dicts.

    `FetchResult.text()` decodes as utf-8-sig, which strips the byte-order mark
    Savant prefixes; without that the first column name would come back as
    '\\ufefflast_name, first_name' and every lookup against it would miss.
    """
    reader = csv.DictReader(io.StringIO(result.text()))
    return [dict(row) for row in reader]


def to_number(value: str | None) -> float | None:
    """Convert a Savant CSV cell to a number, or None if it is blank.

    Savant leaves cells empty rather than zero when a metric does not apply
    (for example spin on a pitch a pitcher does not throw). Blank must stay
    distinguishable from zero.
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Leaderboards
# ---------------------------------------------------------------------------


def expected_statistics(
    archiver: Archiver,
    *,
    year: int,
    player_type: str,
    minimum: str | int = "q",
) -> FetchResult:
    """Expected outcomes from batted-ball quality: xBA, xSLG, xwOBA, and for
    pitchers also xERA.

    `minimum="q"` restricts to qualified players. A scouting report needs
    part-time players too, so callers generally pass an explicit low integer.
    """
    return _leaderboard(
        LB_EXPECTED_STATISTICS,
        archiver,
        {"type": player_type, "year": year, "min": minimum},
    )


def pitch_arsenal_stats(
    archiver: Archiver,
    *,
    year: int,
    minimum: int = 10,
    player_type: str = TYPE_PITCHER,
) -> FetchResult:
    """Per-pitch-type results, for pitchers or for batters.

    For a pitcher this is his arsenal: what he throws, how often, and how each
    offering has played. For a batter the same endpoint returns the mirror
    image -- how he has performed against each pitch type he has faced, with
    `pitch_usage` meaning the share of pitches thrown to him rather than by him.

    Together these are what let the report put a hitter's weakness against
    changeups next to the fact that the starter throws one a quarter of the
    time.
    """
    return _leaderboard(
        LB_PITCH_ARSENAL,
        archiver,
        {"type": player_type, "year": year, "min": minimum},
    )


def league_average_by_pitch_type(
    rows: list[dict[str, str]]
) -> dict[str, dict[str, float | None]]:
    """League-average results for each pitch type, weighted by pitch count.

    Used to benchmark an individual arsenal line: a .320 xwOBA against sliders
    means something different from a .320 against fastballs, and this is what
    supplies that context.

    Weighting by pitches is deliberate. Averaging the per-pitcher rates would
    give a pitcher who threw forty sliders the same say as one who threw two
    thousand, which is not the league rate.
    """
    weighted: dict[str, dict[str, float]] = {}
    totals: dict[str, float] = {}

    metrics = ("whiff_percent", "put_away", "est_woba", "woba", "hard_hit_percent")

    for row in rows:
        pitch = row.get("pitch_type")
        pitches = to_number(row.get("pitches"))
        if not pitch or not pitches:
            continue
        totals[pitch] = totals.get(pitch, 0.0) + pitches
        bucket = weighted.setdefault(pitch, {})
        for metric in metrics:
            value = to_number(row.get(metric))
            if value is not None:
                bucket[metric] = bucket.get(metric, 0.0) + value * pitches

    return {
        pitch: {
            metric: (sums[metric] / totals[pitch] if metric in sums else None)
            for metric in metrics
        }
        for pitch, sums in weighted.items()
        if totals.get(pitch)
    }


def percentile_rankings(
    archiver: Archiver, *, year: int, player_type: str
) -> FetchResult:
    """League percentile ranks (0-100) for the headline Statcast metrics.

    These are what Savant's familiar red-and-blue player sliders show. They are
    already normalised to the league, so they communicate context far faster
    than a raw rate does.
    """
    return _leaderboard(
        LB_PERCENTILE_RANKINGS, archiver, {"type": player_type, "year": year}
    )


# ---------------------------------------------------------------------------
# Indexing helpers
# ---------------------------------------------------------------------------

# Savant identifies players by MLBAM id, the same id the MLB Stats API uses, so
# the two sources join cleanly with no name matching. Name matching would be a
# reliability problem: accents, suffixes and duplicate names all break it.
PLAYER_ID_FIELD = "player_id"


def index_by_player(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    """Index leaderboard rows by MLBAM player id, one row per player."""
    indexed: dict[int, dict[str, str]] = {}
    for row in rows:
        raw_id = row.get(PLAYER_ID_FIELD)
        if not raw_id:
            continue
        indexed[int(raw_id)] = row
    return indexed


def group_by_player(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    """Group rows by player id, preserving order.

    The arsenal leaderboard emits one row per pitch type per pitcher, so a
    starter with five offerings appears five times.
    """
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        raw_id = row.get(PLAYER_ID_FIELD)
        if not raw_id:
            continue
        grouped.setdefault(int(raw_id), []).append(row)
    return grouped
