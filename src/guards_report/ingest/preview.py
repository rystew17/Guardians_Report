"""Assemble a four-page game preview: fetch, compute, and return a typed bundle.

Structure is four pages -- each team's pitching and batting -- with one box per
player carrying that player's full stat view.

Fetching is aggressively batched. The statsapi `stats(...)` hydrate accepts a
list of stat types and a list of personIds at once, so season lines,
sabermetrics, platoon splits and zone data for 52 players collapse into a
handful of requests rather than one per player per stat type. That takes a full
preview from roughly 200 calls to about two dozen.

This layer orchestrates only. Every derived number comes from metrics/, and
nothing is written to BigQuery here, so a report can be produced with no cloud
dependency at all.
"""

from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from guards_report.config import (
    CLEVELAND_GUARDIANS_TEAM_ID,
    DEFAULT_SPLIT_CODES,
    REPO_ROOT,
    Settings,
)
from guards_report.metrics import league_averages as la
from guards_report.metrics import league_constants as lc
from guards_report.metrics import windows as w
from guards_report.metrics import zones as zn
from guards_report.sources import mlb_statsapi as api
from guards_report.sources import savant as sv
from guards_report.sources.http import Archiver

POSITION_PLAYER_TYPES = {"Catcher", "Infielder", "Outfielder", "Two-Way Player"}


# ---------------------------------------------------------------------------
# Bundle types
# ---------------------------------------------------------------------------


@dataclass
class WindowLine:
    label: str
    stats: dict[str, Any]
    deltas: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlayerBox:
    """One player's complete stat view -- the unit the report is built from."""

    player_id: int
    name: str
    hand: str | None          # bats, for hitters; throws, for pitchers
    position: str | None
    jersey: str | None
    is_probable_starter: bool = False

    season: dict[str, Any] = field(default_factory=dict)
    season_deltas: dict[str, Any] = field(default_factory=dict)
    windows: list[WindowLine] = field(default_factory=list)

    percentiles: dict[str, str] = field(default_factory=dict)
    expected: dict[str, str] = field(default_factory=dict)
    sabermetrics: dict[str, Any] = field(default_factory=dict)
    arsenal: list[dict[str, Any]] = field(default_factory=list)
    zone_grids: dict[str, zn.ZoneGrid] = field(default_factory=dict)

    # Hitters only: performance against the handedness they will face today.
    vs_hand: dict[str, Any] = field(default_factory=dict)
    vs_hand_label: str = ""

    # Pitchers only.
    availability: w.BullpenAvailability | None = None
    role: str = ""


@dataclass
class TeamSection:
    team_id: int
    name: str
    abbreviation: str
    record: str
    is_home: bool
    batters: list[PlayerBox] = field(default_factory=list)
    pitchers: list[PlayerBox] = field(default_factory=list)
    opposing_hand: str | None = None


@dataclass
class ReportBundle:
    run_id: str
    generated_at: datetime
    as_of_date: date
    git_sha: str | None
    game_pk: int
    game_date: date
    game_datetime: datetime | None
    venue_name: str
    status: str
    weather: dict[str, Any]
    season: int
    home: TeamSection
    away: TeamSection
    league: lc.LeagueConstants
    league_hitting: la.LeagueHitting
    league_pitching: la.LeaguePitching
    provenance: list[dict[str, Any]]

    @property
    def guardians(self) -> TeamSection:
        return (
            self.home
            if self.home.team_id == CLEVELAND_GUARDIANS_TEAM_ID
            else self.away
        )

    @property
    def opponent(self) -> TeamSection:
        return (
            self.away
            if self.home.team_id == CLEVELAND_GUARDIANS_TEAM_ID
            else self.home
        )


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _stat_block(person: dict[str, Any], type_name: str) -> list[dict[str, Any]]:
    """All splits for one stat type from a hydrated person object."""
    for block in person.get("stats") or []:
        if (block.get("type") or {}).get("displayName") == type_name:
            return block.get("splits") or []
    return []


def _first_stat(person: dict[str, Any], type_name: str) -> dict[str, Any]:
    splits = _stat_block(person, type_name)
    return splits[0].get("stat", {}) if splits else {}


def _merge_people(results: list[Any]) -> dict[int, dict[str, Any]]:
    """Merge batched people payloads, combining each person's stats blocks.

    A player appears once per batched request, so the stats blocks from
    separate hydrates (light stats and game logs) have to be concatenated
    rather than overwriting one another.
    """
    merged: dict[int, dict[str, Any]] = {}
    for result in results:
        for person in result.json().get("people", []):
            pid = person.get("id")
            if pid is None:
                continue
            if pid in merged:
                merged[pid].setdefault("stats", [])
                merged[pid]["stats"].extend(person.get("stats") or [])
            else:
                merged[pid] = dict(person)
                merged[pid].setdefault("stats", [])
    return merged


def _find_game(payload: dict[str, Any], *, team_id: int) -> dict[str, Any] | None:
    for day in payload.get("dates", []):
        for game in day.get("games", []):
            teams = game.get("teams", {})
            ids = {
                teams.get("home", {}).get("team", {}).get("id"),
                teams.get("away", {}).get("team", {}).get("id"),
            }
            if team_id in ids:
                return game
    return None


def _split_roster(payload: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Separate an active roster into position players and pitchers.

    A 26-man active roster is 13 and 13 in the current era, which is exactly
    the split the report wants. Two-way players are counted as hitters here and
    appear again among the pitchers if they are listed as such.
    """
    batters, pitchers = [], []
    for entry in payload.get("roster", []):
        position_type = (entry.get("position") or {}).get("type")
        if position_type == "Pitcher":
            pitchers.append(entry)
        elif position_type in POSITION_PLAYER_TYPES:
            batters.append(entry)
    return batters, pitchers


# ---------------------------------------------------------------------------
# Box construction
# ---------------------------------------------------------------------------

HITTER_DELTA_KEYS = ("avg", "obp", "slg", "ops", "iso", "babip", "kPct", "bbPct")
PITCHER_DELTA_KEYS = (
    "era", "whip", "fip", "kPer9", "bbPer9", "hrPer9", "kPct", "bbPct",
    "kMinusBbPct",
)

# Stats where a lower value is the better outcome, per role.
HITTER_LOWER_BETTER = frozenset({"kPct"})
PITCHER_LOWER_BETTER = frozenset(
    {"era", "whip", "fip", "bbPer9", "hrPer9", "bbPct"}
)


def _deltas(
    stats: dict[str, Any],
    league: la.LeagueHitting | la.LeaguePitching,
    keys: tuple[str, ...],
    lower_better: frozenset[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        result = la.delta(
            stats.get(key), league.get(key), lower_is_better=key in lower_better
        )
        if result is not None:
            out[key] = result
    return out


def _arsenal_rows(
    rows: list[dict[str, str]], league_by_pitch: dict[str, dict[str, float | None]]
) -> list[dict[str, Any]]:
    out = []
    for row in sorted(
        rows, key=lambda r: -(sv.to_number(r.get("pitch_usage")) or 0)
    ):
        pitch_type = row.get("pitch_type") or ""
        benchmark = league_by_pitch.get(pitch_type, {})
        out.append(
            {
                "pitch": row.get("pitch_name"),
                "pitch_type": pitch_type,
                "usage": sv.to_number(row.get("pitch_usage")),
                "pitches": sv.to_number(row.get("pitches")),
                "pa": sv.to_number(row.get("pa")),
                "whiff": sv.to_number(row.get("whiff_percent")),
                "put_away": sv.to_number(row.get("put_away")),
                "ba": sv.to_number(row.get("ba")),
                "slg": sv.to_number(row.get("slg")),
                "woba": sv.to_number(row.get("woba")),
                "xwoba": sv.to_number(row.get("est_woba")),
                "hard_hit": sv.to_number(row.get("hard_hit_percent")),
                "run_value_per_100": sv.to_number(row.get("run_value_per_100")),
                "lg_whiff": benchmark.get("whiff_percent"),
                "lg_xwoba": benchmark.get("est_woba"),
            }
        )
    return out


def _build_batter_box(
    person: dict[str, Any],
    entry: dict[str, Any],
    *,
    as_of: date,
    opposing_hand: str | None,
    league_hitting: la.LeagueHitting,
    arsenal_by_player: dict[int, list[dict[str, str]]],
    league_by_pitch: dict[str, dict[str, float | None]],
    percentiles: dict[int, dict[str, str]],
    expected: dict[int, dict[str, str]],
) -> PlayerBox:
    pid = person["id"]

    rows = w.parse_game_logs({"stats": [{"splits": _stat_block(person, "gameLog")}]})
    prior = [r for r in rows if r.game_date < as_of]

    season_stats = w.aggregate_hitting(prior)
    windows = []
    for spec in w.HITTER_WINDOWS:
        stats = w.aggregate_hitting(w.select(rows, spec, as_of=as_of))
        windows.append(
            WindowLine(
                label=spec.label,
                stats=stats,
                deltas=_deltas(
                    stats, league_hitting, HITTER_DELTA_KEYS, HITTER_LOWER_BETTER
                ),
            )
        )

    vs_hand: dict[str, Any] = {}
    vs_hand_label = ""
    if opposing_hand in ("L", "R"):
        wanted = "vl" if opposing_hand == "L" else "vr"
        vs_hand_label = f"vs {'LHP' if opposing_hand == 'L' else 'RHP'}"
        for split in _stat_block(person, "statSplits"):
            if (split.get("split") or {}).get("code") == wanted:
                vs_hand = split.get("stat", {})
                break

    return PlayerBox(
        player_id=pid,
        name=person.get("fullName", "Unknown"),
        hand=(person.get("batSide") or {}).get("code"),
        position=(entry.get("position") or {}).get("abbreviation"),
        jersey=entry.get("jerseyNumber"),
        season=season_stats,
        season_deltas=_deltas(
            season_stats, league_hitting, HITTER_DELTA_KEYS, HITTER_LOWER_BETTER
        ),
        windows=windows,
        percentiles=percentiles.get(pid, {}),
        expected=expected.get(pid, {}),
        sabermetrics=_first_stat(person, "sabermetrics"),
        arsenal=_arsenal_rows(arsenal_by_player.get(pid, []), league_by_pitch),
        zone_grids=zn.parse_zones(person),
        vs_hand=vs_hand,
        vs_hand_label=vs_hand_label,
    )


def _build_pitcher_box(
    person: dict[str, Any],
    entry: dict[str, Any],
    *,
    as_of: date,
    fip_constant: float,
    league_pitching: la.LeaguePitching,
    arsenal_by_player: dict[int, list[dict[str, str]]],
    league_by_pitch: dict[str, dict[str, float | None]],
    percentiles: dict[int, dict[str, str]],
    expected: dict[int, dict[str, str]],
    probable_starter_id: int | None,
) -> PlayerBox:
    pid = person["id"]

    rows = w.parse_game_logs({"stats": [{"splits": _stat_block(person, "gameLog")}]})
    prior = [r for r in rows if r.game_date < as_of]

    season_stats = w.aggregate_pitching(prior, fip_constant=fip_constant)
    windows = []
    for spec in w.PITCHER_WINDOWS:
        stats = w.aggregate_pitching(
            w.select(rows, spec, as_of=as_of), fip_constant=fip_constant
        )
        windows.append(
            WindowLine(
                label=spec.label,
                stats=stats,
                deltas=_deltas(
                    stats, league_pitching, PITCHER_DELTA_KEYS, PITCHER_LOWER_BETTER
                ),
            )
        )

    games = season_stats.get("games", 0) or 0
    starts = season_stats.get("gamesStarted", 0) or 0
    if starts >= max(1, games * 0.5):
        role = "SP"
    elif season_stats.get("saves", 0):
        role = "RP (closer)"
    else:
        role = "RP"

    return PlayerBox(
        player_id=pid,
        name=person.get("fullName", "Unknown"),
        hand=(person.get("pitchHand") or {}).get("code"),
        position="P",
        jersey=entry.get("jerseyNumber"),
        is_probable_starter=(pid == probable_starter_id),
        season=season_stats,
        season_deltas=_deltas(
            season_stats, league_pitching, PITCHER_DELTA_KEYS, PITCHER_LOWER_BETTER
        ),
        windows=windows,
        percentiles=percentiles.get(pid, {}),
        expected=expected.get(pid, {}),
        sabermetrics=_first_stat(person, "sabermetrics"),
        arsenal=_arsenal_rows(arsenal_by_player.get(pid, []), league_by_pitch),
        zone_grids=zn.parse_zones(person),
        availability=w.bullpen_availability(rows, as_of=as_of),
        role=role,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_preview(
    settings: Settings, *, on: date, team_id: int = CLEVELAND_GUARDIANS_TEAM_ID
) -> ReportBundle:
    archiver = Archiver(root=settings.raw_archive_dir)
    season = on.year

    schedule_payload = api.schedule(archiver, on=on, team_id=team_id).json()
    game = _find_game(schedule_payload, team_id=team_id)
    if game is None:
        raise LookupError(f"no game found for team {team_id} on {on.isoformat()}")

    game_pk = game["gamePk"]
    teams = game["teams"]

    # League context first: FIP needs the season's constant, and every stat is
    # benchmarked against a league average derived the same way.
    pitching_totals_payload = api.league_pitching_totals(
        archiver, season=season
    ).json()
    league = lc.derive(
        lc.parse_league_totals(pitching_totals_payload, season=season)
    )
    league_pitching = la.pitching_from_payload(
        pitching_totals_payload, season=season, fip_constant=league.fip_constant
    )
    league_hitting = la.hitting_from_payload(
        api.league_hitting_totals(archiver, season=season).json(), season=season
    )

    # Savant leaderboards: fetched once, indexed, reused for every player.
    pitcher_arsenal_rows = sv.parse_csv(
        sv.pitch_arsenal_stats(
            archiver, year=season, minimum=1, player_type=sv.TYPE_PITCHER
        )
    )
    batter_arsenal_rows = sv.parse_csv(
        sv.pitch_arsenal_stats(
            archiver, year=season, minimum=1, player_type=sv.TYPE_BATTER
        )
    )
    pitcher_arsenal = sv.group_by_player(pitcher_arsenal_rows)
    batter_arsenal = sv.group_by_player(batter_arsenal_rows)
    league_by_pitch_thrown = sv.league_average_by_pitch_type(pitcher_arsenal_rows)
    league_by_pitch_faced = sv.league_average_by_pitch_type(batter_arsenal_rows)

    pitcher_percentiles = sv.index_by_player(
        sv.parse_csv(
            sv.percentile_rankings(archiver, year=season, player_type=sv.TYPE_PITCHER)
        )
    )
    batter_percentiles = sv.index_by_player(
        sv.parse_csv(
            sv.percentile_rankings(archiver, year=season, player_type=sv.TYPE_BATTER)
        )
    )
    pitcher_expected = sv.index_by_player(
        sv.parse_csv(
            sv.expected_statistics(
                archiver, year=season, player_type=sv.TYPE_PITCHER, minimum=1
            )
        )
    )
    batter_expected = sv.index_by_player(
        sv.parse_csv(
            sv.expected_statistics(
                archiver, year=season, player_type=sv.TYPE_BATTER, minimum=1
            )
        )
    )

    # Rosters
    roster_entries: dict[str, tuple[list[dict], list[dict]]] = {}
    for side in ("home", "away"):
        roster_entries[side] = _split_roster(
            api.active_roster(
                archiver, team_id=teams[side]["team"]["id"], season=season
            ).json()
        )

    probable: dict[str, int | None] = {}
    for side in ("home", "away"):
        pitcher = teams[side].get("probablePitcher")
        probable[side] = pitcher["id"] if pitcher else None

    all_batter_ids = [
        e["person"]["id"] for side in ("home", "away") for e in roster_entries[side][0]
    ]
    all_pitcher_ids = [
        e["person"]["id"] for side in ("home", "away") for e in roster_entries[side][1]
    ]

    # Batched fetches. Light stat types together; game logs separately with a
    # smaller batch because they are far larger per player.
    batter_people = _merge_people(
        api.people_with_stats(
            archiver,
            person_ids=all_batter_ids,
            group=api.GROUP_HITTING,
            stat_types=[
                api.STAT_SEASON,
                api.STAT_SABERMETRICS,
                api.STAT_SPLITS,
                api.STAT_HOT_COLD_ZONES,
            ],
            season=season,
            sit_codes=DEFAULT_SPLIT_CODES,
        )
        + api.people_with_stats(
            archiver,
            person_ids=all_batter_ids,
            group=api.GROUP_HITTING,
            stat_types=[api.STAT_GAME_LOG],
            season=season,
            batch_size=api.GAMELOG_BATCH_SIZE,
        )
    )

    pitcher_people = _merge_people(
        api.people_with_stats(
            archiver,
            person_ids=all_pitcher_ids,
            group=api.GROUP_PITCHING,
            stat_types=[
                api.STAT_SEASON,
                api.STAT_SABERMETRICS,
                api.STAT_HOT_COLD_ZONES,
            ],
            season=season,
        )
        + api.people_with_stats(
            archiver,
            person_ids=all_pitcher_ids,
            group=api.GROUP_PITCHING,
            stat_types=[api.STAT_GAME_LOG],
            season=season,
            batch_size=api.GAMELOG_BATCH_SIZE,
        )
    )

    # Handedness of each probable starter, for the opposing hitters' splits.
    hands: dict[str, str | None] = {}
    for side in ("home", "away"):
        pid = probable[side]
        hands[side] = (
            (pitcher_people.get(pid, {}).get("pitchHand") or {}).get("code")
            if pid
            else None
        )

    sections: dict[str, TeamSection] = {}
    for side in ("home", "away"):
        opposite = "away" if side == "home" else "home"
        team_info = teams[side]["team"]
        record = teams[side].get("leagueRecord", {})
        batter_entries, pitcher_entries = roster_entries[side]

        batters = [
            _build_batter_box(
                batter_people[entry["person"]["id"]],
                entry,
                as_of=on,
                opposing_hand=hands[opposite],
                league_hitting=league_hitting,
                arsenal_by_player=batter_arsenal,
                league_by_pitch=league_by_pitch_faced,
                percentiles=batter_percentiles,
                expected=batter_expected,
            )
            for entry in batter_entries
            if entry["person"]["id"] in batter_people
        ]
        # Most plate appearances first: the players most likely to bat today.
        batters.sort(key=lambda b: -(b.season.get("plateAppearances") or 0))

        pitchers = [
            _build_pitcher_box(
                pitcher_people[entry["person"]["id"]],
                entry,
                as_of=on,
                fip_constant=league.fip_constant,
                league_pitching=league_pitching,
                arsenal_by_player=pitcher_arsenal,
                league_by_pitch=league_by_pitch_thrown,
                percentiles=pitcher_percentiles,
                expected=pitcher_expected,
                probable_starter_id=probable[side],
            )
            for entry in pitcher_entries
            if entry["person"]["id"] in pitcher_people
        ]
        # Today's probable starter first, then the bullpen ordered by how
        # rested it is, then the rest of the rotation last.
        #
        # Sorting purely by days rest would float the other starters to the
        # top -- a man who threw four days ago looks maximally available by
        # that measure -- when they are in fact the least likely arms to
        # appear today. Rotation members other than today's starter are
        # therefore pushed below the relievers.
        def _pitcher_order(box: PlayerBox) -> tuple[int, int, int]:
            if box.is_probable_starter:
                group = 0
            elif box.role == "SP":
                group = 2
            else:
                group = 1

            rest = (
                box.availability.days_rest
                if box.availability and box.availability.days_rest is not None
                else 99
            )
            return (group, -rest, -(box.season.get("outs") or 0))

        pitchers.sort(key=_pitcher_order)

        sections[side] = TeamSection(
            team_id=team_info["id"],
            name=team_info.get("name", ""),
            abbreviation=team_info.get("abbreviation", ""),
            record=f"{record.get('wins', 0)}-{record.get('losses', 0)}",
            is_home=(side == "home"),
            batters=batters,
            pitchers=pitchers,
            opposing_hand=hands[opposite],
        )

    weather: dict[str, Any] = {}
    try:
        feed = api.game_feed(archiver, game_pk=game_pk).json()
        weather = (feed.get("gameData") or {}).get("weather") or {}
    except Exception:
        weather = {}

    game_datetime = None
    if game.get("gameDate"):
        game_datetime = datetime.fromisoformat(
            game["gameDate"].replace("Z", "+00:00")
        )

    return ReportBundle(
        run_id=uuid.uuid4().hex[:12],
        generated_at=datetime.now(timezone.utc),
        as_of_date=on,
        git_sha=_git_sha(),
        game_pk=game_pk,
        game_date=date.fromisoformat(game["officialDate"]),
        game_datetime=game_datetime,
        venue_name=(game.get("venue") or {}).get("name", ""),
        status=(game.get("status") or {}).get("detailedState", ""),
        weather=weather,
        season=season,
        home=sections["home"],
        away=sections["away"],
        league=league,
        league_hitting=league_hitting,
        league_pitching=league_pitching,
        provenance=[r.provenance() for r in archiver.written],
    )
