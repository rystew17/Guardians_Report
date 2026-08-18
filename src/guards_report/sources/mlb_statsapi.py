"""Client for the MLB Stats API (statsapi.mlb.com).

Free, unauthenticated, and the backbone of this project. It supplies schedule,
probable pitchers, rosters, official lineups, per-game logs, date-range and
situational splits, and MLB's own sabermetric figures (wOBA, wRC+, WAR).

This module only fetches and returns parsed JSON. It does no arithmetic --
every derived number is computed in metrics/formulas.py so that there is one
place to audit. Endpoint quirks worth knowing are documented inline.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date
from typing import Any

from guards_report.config import MLB_SPORT_ID, STATSAPI_BASE
from guards_report.sources.http import Archiver, FetchResult, fetch

SOURCE = "mlb_statsapi"

# statsapi rejects a request whose personIds list is too long. Well under any
# observed limit, and it keeps individual archived payloads a sane size.
PERSON_BATCH_SIZE = 20

# Game logs run about 130 KB per player for a full season, so they get a
# smaller batch to keep any single archived payload manageable.
GAMELOG_BATCH_SIZE = 8

STAT_SEASON = "season"
STAT_GAME_LOG = "gameLog"
STAT_SABERMETRICS = "sabermetrics"
STAT_SPLITS = "statSplits"
STAT_HOT_COLD_ZONES = "hotColdZones"


def _get(
    path: str, archiver: Archiver, params: dict[str, Any] | None = None
) -> FetchResult:
    return fetch(
        f"{STATSAPI_BASE}{path}", source=SOURCE, archiver=archiver, params=params
    )


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


def schedule(
    archiver: Archiver,
    *,
    on: date,
    team_id: int | None = None,
    hydrate: str = "probablePitcher,lineups,venue,team,linescore",
) -> FetchResult:
    """Games on a date, optionally filtered to one team.

    `hydrate=lineups` returns homePlayers/awayPlayers in batting order, but
    only once the club has posted the official lineup -- typically about three
    hours before first pitch. Before that the key is absent or empty. Callers
    must handle that rather than assume a lineup exists; see ingest/lineups.py.
    """
    params: dict[str, Any] = {
        "sportId": MLB_SPORT_ID,
        "date": on.isoformat(),
        "hydrate": hydrate,
    }
    if team_id is not None:
        params["teamId"] = team_id
    return _get("/v1/schedule", archiver, params)


def schedule_range(
    archiver: Archiver,
    *,
    start: date,
    end: date,
    team_id: int | None = None,
    hydrate: str = "probablePitcher,venue,team",
) -> FetchResult:
    params: dict[str, Any] = {
        "sportId": MLB_SPORT_ID,
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "hydrate": hydrate,
    }
    if team_id is not None:
        params["teamId"] = team_id
    return _get("/v1/schedule", archiver, params)


def game_feed(archiver: Archiver, *, game_pk: int) -> FetchResult:
    """Full live feed for one game.

    This is the only place weather and umpire assignments are exposed, but the
    payload is large (roughly 800 KB for a completed game) because it carries
    every play. We fetch it for context fields only and never archive it into
    BigQuery -- ingest extracts the handful of fields it needs.
    """
    return _get(f"/v1.1/game/{game_pk}/feed/live", archiver)


# ---------------------------------------------------------------------------
# Teams and people
# ---------------------------------------------------------------------------


def active_roster(archiver: Archiver, *, team_id: int, season: int) -> FetchResult:
    return _get(
        f"/v1/teams/{team_id}/roster/active", archiver, {"season": season}
    )


def team(archiver: Archiver, *, team_id: int, season: int) -> FetchResult:
    return _get(f"/v1/teams/{team_id}", archiver, {"season": season})


def people_with_stats(
    archiver: Archiver,
    *,
    person_ids: Sequence[int],
    group: str,
    stat_types: Sequence[str],
    season: int,
    sit_codes: Sequence[str] = (),
    batch_size: int = PERSON_BATCH_SIZE,
) -> list[FetchResult]:
    """Fetch several stat types for several players in one request each.

    The `stats(...)` hydrate accepts a list of types and a list of personIds
    simultaneously, so a full report's worth of season lines, sabermetrics,
    splits and zone data collapses from one request per player per stat type
    into a handful of requests. For a 52-player preview that is the difference
    between roughly 200 calls and about a dozen -- materially faster, and much
    gentler on an unauthenticated public API we depend on.

    Game logs are large enough (~130 KB per player) that callers should pass a
    smaller batch_size for them; see GAMELOG_BATCH_SIZE.
    """
    types = ",".join(stat_types)
    inner = f"group=[{group}],type=[{types}]"
    if sit_codes:
        inner += f",sitCodes=[{','.join(sit_codes)}]"
    inner += f",season={season}"

    results: list[FetchResult] = []
    for start in range(0, len(person_ids), batch_size):
        batch = person_ids[start : start + batch_size]
        results.append(
            _get(
                "/v1/people",
                archiver,
                {
                    "personIds": ",".join(str(i) for i in batch),
                    "hydrate": f"stats({inner})",
                },
            )
        )
    return results


def people(
    archiver: Archiver,
    *,
    person_ids: Sequence[int],
    hydrate: str = "currentTeam",
) -> list[FetchResult]:
    """Look up players in batches.

    One request per player would mean ~60 round trips for a two-team preview.
    The batch form takes a comma-separated personIds list, which cuts that to
    two. Returns one FetchResult per batch; callers concatenate the `people`
    arrays.
    """
    results: list[FetchResult] = []
    for start in range(0, len(person_ids), PERSON_BATCH_SIZE):
        batch = person_ids[start : start + PERSON_BATCH_SIZE]
        results.append(
            _get(
                "/v1/people",
                archiver,
                {"personIds": ",".join(str(i) for i in batch), "hydrate": hydrate},
            )
        )
    return results


# ---------------------------------------------------------------------------
# Player statistics
# ---------------------------------------------------------------------------
# The `group` argument matters: asking for hitting stats on a pitcher returns
# an empty splits array rather than an error. Empty is a legitimate answer
# here, so ingest must not treat it as a failure.

GROUP_HITTING = "hitting"
GROUP_PITCHING = "pitching"
GROUP_FIELDING = "fielding"


def _stats(
    archiver: Archiver, *, person_id: int, params: dict[str, Any]
) -> FetchResult:
    return _get(f"/v1/people/{person_id}/stats", archiver, params)


def game_log(
    archiver: Archiver, *, person_id: int, season: int, group: str
) -> FetchResult:
    """Every individual game a player appeared in this season.

    This is the atom the warehouse persists. Season totals and the L5/L15/L30
    windows are all aggregations of these rows, so they are computed rather
    than stored.
    """
    return _stats(
        archiver,
        person_id=person_id,
        params={"stats": "gameLog", "season": season, "group": group},
    )


def season_totals(
    archiver: Archiver, *, person_id: int, season: int, group: str
) -> FetchResult:
    return _stats(
        archiver,
        person_id=person_id,
        params={"stats": "season", "season": season, "group": group},
    )


def last_x_games(
    archiver: Archiver, *, person_id: int, season: int, group: str, limit: int
) -> FetchResult:
    """The source's own last-N-games aggregate.

    We compute our windows from game logs instead, but this endpoint is a
    useful independent check: our L15 built from game logs should agree with
    the source's own L15. Disagreement means a bug in our window logic.
    """
    return _stats(
        archiver,
        person_id=person_id,
        params={
            "stats": "lastXGames",
            "season": season,
            "group": group,
            "limit": limit,
        },
    )


def by_date_range(
    archiver: Archiver,
    *,
    person_id: int,
    season: int,
    group: str,
    start: date,
    end: date,
) -> FetchResult:
    return _stats(
        archiver,
        person_id=person_id,
        params={
            "stats": "byDateRange",
            "season": season,
            "group": group,
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
        },
    )


def stat_splits(
    archiver: Archiver,
    *,
    person_id: int,
    season: int,
    group: str,
    split_codes: Iterable[str],
) -> FetchResult:
    """Situational splits: vl/vr for platoon, h/a for home and away.

    Platoon splits cannot be derived from game logs -- a game log does not
    break plate appearances out by the handedness of the opposing pitcher --
    so this snapshot is one of the few things the warehouse has to store.

    The response returns one entry per requested code in `splits`, each tagged
    with a `split.code`. The order is not guaranteed, so match on the code.
    """
    return _stats(
        archiver,
        person_id=person_id,
        params={
            "stats": "statSplits",
            "season": season,
            "group": group,
            "sitCodes": ",".join(split_codes),
        },
    )


def sabermetrics(
    archiver: Archiver, *, person_id: int, season: int, group: str
) -> FetchResult:
    """MLB's own sabermetric figures: wOBA, wRC+, WAR, wRAA, RAR.

    Taking these from the source rather than deriving them is deliberate. wOBA
    and wRC+ require season-specific linear weights and park factors that MLB
    computes internally; reproducing them would mean guessing at inputs, which
    is exactly what the verifiability rule forbids. A cited source number beats
    a reconstructed one.
    """
    return _stats(
        archiver,
        person_id=person_id,
        params={"stats": "sabermetrics", "season": season, "group": group},
    )


def vs_player(
    archiver: Archiver,
    *,
    batter_id: int,
    pitcher_id: int,
    season: int | None = None,
) -> FetchResult:
    """Career batter-versus-pitcher history.

    Samples are almost always tiny -- a dozen plate appearances is typical --
    so the report shows these as raw counts and never as a rate. Presenting
    "3 for 8" as a .375 average against would imply a signal that is not there.
    """
    params: dict[str, Any] = {
        "stats": "vsPlayer",
        "group": GROUP_HITTING,
        "opposingPlayerId": pitcher_id,
    }
    if season is not None:
        params["season"] = season
    return _stats(archiver, person_id=batter_id, params=params)


# ---------------------------------------------------------------------------
# League-wide totals, for deriving the FIP constant
# ---------------------------------------------------------------------------


def standings(
    archiver: Archiver, *, season: int, league_ids: Sequence[int] = (103, 104)
) -> FetchResult:
    """Current standings for both leagues.

    Carries division and league rank, games back, streak, and a set of record
    splits including home, away, day, night, one-run, extra-inning, last ten,
    and xWinLoss -- MLB's own Pythagorean record, which is the cleanest answer
    to "is this team as good as its record".
    """
    return _get(
        "/v1/standings",
        archiver,
        {
            "leagueId": ",".join(str(i) for i in league_ids),
            "season": season,
            "standingsTypes": "regularSeason",
        },
    )


def head_to_head(
    archiver: Archiver, *, team_id: int, opponent_id: int, season: int, through: date
) -> FetchResult:
    """Every game between two clubs this season, up to and including today.

    Used for the season series record. `opponentId` does the filtering server
    side, so this is one request rather than a scan of the full schedule.
    """
    return _get(
        "/v1/schedule",
        archiver,
        {
            "sportId": MLB_SPORT_ID,
            "teamId": team_id,
            "opponentId": opponent_id,
            "season": season,
            "startDate": f"{season}-01-01",
            "endDate": through.isoformat(),
            "hydrate": "team,linescore",
        },
    )


def league_hitting_totals(archiver: Archiver, *, season: int) -> FetchResult:
    """Season hitting totals for every team, summed to give league averages.

    Every stat in the report is benchmarked against these, so a .750 OPS
    carries its own context rather than requiring the reader to know what an
    average one is this year.
    """
    return _get(
        "/v1/teams/stats",
        archiver,
        {
            "season": season,
            "sportIds": MLB_SPORT_ID,
            "group": GROUP_HITTING,
            "stats": "season",
        },
    )


def league_pitching_totals(archiver: Archiver, *, season: int) -> FetchResult:
    """Season pitching totals for every team.

    Summing all 30 teams gives the league totals needed to derive the FIP
    constant for this run environment, rather than hardcoding a number that
    drifts year to year. See metrics/league_constants.py.
    """
    return _get(
        "/v1/teams/stats",
        archiver,
        {
            "season": season,
            "sportIds": MLB_SPORT_ID,
            "group": GROUP_PITCHING,
            "stats": "season",
        },
    )
