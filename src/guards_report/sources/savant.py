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

# Leaderboard identifiers, as they appear in the URL path. Savant is not
# consistent about hyphens versus underscores here, so these are literal.
LB_EXPECTED_STATISTICS = "expected_statistics"
LB_PITCH_ARSENAL = "pitch-arsenal-stats"
LB_PERCENTILE_RANKINGS = "percentile-rankings"
LB_BATTED_BALL = "batted-ball"
LB_BAT_TRACKING = "bat-tracking"
LB_OAA = "outs_above_average"
LB_SPRINT_SPEED = "sprint_speed"
LB_BASESTEALING = "basestealing-run-value"
LB_BASERUNNING = "baserunning"
LB_POPTIME = "poptime"

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
#
# The id column is not named consistently across leaderboards: most use
# player_id, the batted-ball and bat-tracking boards use id, and poptime uses
# entity_id. We look for each in turn rather than assuming.
ID_FIELDS = ("player_id", "id", "entity_id", "fielder_id")


def _row_player_id(row: dict[str, str]) -> int | None:
    for field in ID_FIELDS:
        raw = row.get(field)
        if raw:
            try:
                return int(raw)
            except ValueError:
                continue
    return None


def index_by_player(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    """Index leaderboard rows by MLBAM player id, one row per player."""
    indexed: dict[int, dict[str, str]] = {}
    for row in rows:
        pid = _row_player_id(row)
        if pid is not None:
            indexed[pid] = row
    return indexed


def group_by_player(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    """Group rows by player id, preserving order.

    The arsenal leaderboard emits one row per pitch type per pitcher, so a
    starter with five offerings appears five times.
    """
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        pid = _row_player_id(row)
        if pid is not None:
            grouped.setdefault(pid, []).append(row)
    return grouped


# ---------------------------------------------------------------------------
# Batted ball, swing, defense and running
# ---------------------------------------------------------------------------


def batted_ball(archiver: Archiver, *, year: int, minimum: int = 25) -> FetchResult:
    """Batted-ball profile: trajectory crossed with direction.

    Returns gb/ld/fb/pu rates and pull/straight/oppo rates, plus the cross-tab
    (pull_gb_rate, oppo_air_rate and so on). The cross-tab is the part that
    matters: pull-heavy on the ground is a shift candidate, pull-heavy in the
    air is a home-run profile, and collapsing the two axes loses that
    distinction entirely.
    """
    return _leaderboard(LB_BATTED_BALL, archiver, {"year": year, "min": minimum})


def bat_tracking(archiver: Archiver, *, year: int, minimum: int = 25) -> FetchResult:
    """Swing-level tracking: bat speed, squared-up rate, blast rate.

    The newest Statcast layer, and the one that separates a hitter who is
    genuinely swinging well from one carried by contact-quality luck. Moves
    faster than outcome stats, so it reads a slump earlier.
    """
    return _leaderboard(LB_BAT_TRACKING, archiver, {"year": year, "min": minimum})


def outs_above_average(
    archiver: Archiver, *, year: int, minimum: int = 1
) -> FetchResult:
    """Fielding range value, with directional breakdown and runs prevented."""
    return _leaderboard(
        LB_OAA, archiver, {"type": "Fielder", "year": year, "min": minimum}
    )


def sprint_speed(archiver: Archiver, *, year: int, minimum: int = 1) -> FetchResult:
    """Feet per second on competitive runs, plus home-to-first times."""
    return _leaderboard(LB_SPRINT_SPEED, archiver, {"year": year, "min": minimum})


def basestealing_run_value(archiver: Archiver, *, year: int) -> FetchResult:
    """Run value added by stealing, with attempt and success counts.

    Paired with the opposing catcher's pop time, this is what turns "he is
    fast" into "he can run today", which is an actual decision.
    """
    return _leaderboard(LB_BASESTEALING, archiver, {"year": year})


def baserunning_run_value(archiver: Archiver, *, year: int) -> FetchResult:
    """Run value from advancing on batted balls, separate from stealing."""
    return _leaderboard(LB_BASERUNNING, archiver, {"year": year})


def pop_time(archiver: Archiver, *, year: int) -> FetchResult:
    """Catcher pop time to second and third, with exchange times."""
    return _leaderboard(LB_POPTIME, archiver, {"year": year})


# ---------------------------------------------------------------------------
# Pitch-level Statcast
# ---------------------------------------------------------------------------


def player_pitches(
    archiver: Archiver, *, player_id: int, year: int, perspective: str
) -> FetchResult:
    """Every tracked pitch for one player this season.

    The only pitch-level fetch in the project, and it is scoped to the ~50
    people in today's game rather than the league. It is what makes two things
    possible that no aggregate endpoint provides: zone grids split by opposing
    handedness, and a spray chart from real batted-ball coordinates.

    Roughly 1-2 MB and under two thousand rows for a regular; a reliever is a
    fraction of that. Archived like every other payload, but never loaded into
    the warehouse -- only the aggregates computed from it are displayed.
    """
    key = "batters_lookup[]" if perspective == "batter" else "pitchers_lookup[]"
    return fetch(
        f"{SAVANT_BASE}/statcast_search/csv",
        source=SOURCE,
        archiver=archiver,
        params={
            "all": "true",
            "hfSea": f"{year}|",
            "player_type": perspective,
            key: player_id,
            "type": "details",
        },
    )
