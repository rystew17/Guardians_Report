"""Assemble a game preview: fetch, compute, and return a typed bundle.

This is the orchestration layer. It fetches from the source clients, runs the
numbers through metrics/, and returns a ReportBundle that the renderer turns
into HTML. It performs no arithmetic of its own -- every derived figure comes
from metrics/formulas.py -- and it writes nothing to BigQuery, so a report can
be produced with no cloud dependency at all.
"""

from __future__ import annotations

import subprocess
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from guards_report.config import (
    CLEVELAND_GUARDIANS_TEAM_ID,
    DEFAULT_SPLIT_CODES,
    REPO_ROOT,
    Settings,
)
from guards_report.metrics import formulas as f
from guards_report.metrics import league_constants as lc
from guards_report.metrics import windows as w
from guards_report.sources import mlb_statsapi as api
from guards_report.sources import savant as sv
from guards_report.sources.http import Archiver


# ---------------------------------------------------------------------------
# Bundle types
# ---------------------------------------------------------------------------


@dataclass
class WindowLine:
    """One recent-form window's aggregated line."""

    label: str
    stats: dict[str, Any]


@dataclass
class PitcherProfile:
    player_id: int
    name: str
    throws: str | None
    season: dict[str, Any]
    windows: list[WindowLine]
    arsenal: list[dict[str, Any]] = field(default_factory=list)
    percentiles: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)
    sabermetrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatterProfile:
    player_id: int
    name: str
    bats: str | None
    position: str | None
    batting_order: int | None
    season: dict[str, Any]
    windows: list[WindowLine]
    vs_hand: dict[str, Any] = field(default_factory=dict)
    vs_hand_label: str = ""
    expected: dict[str, Any] = field(default_factory=dict)
    percentiles: dict[str, Any] = field(default_factory=dict)
    sabermetrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class TeamPreview:
    team_id: int
    name: str
    abbreviation: str
    record: str
    is_home: bool
    starter: PitcherProfile | None
    lineup: list[BatterProfile]
    lineup_source: str
    lineup_source_note: str


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
    home: TeamPreview
    away: TeamPreview
    league: lc.LeagueConstants
    provenance: list[dict[str, Any]]

    @property
    def guardians(self) -> TeamPreview:
        return (
            self.home
            if self.home.team_id == CLEVELAND_GUARDIANS_TEAM_ID
            else self.away
        )

    @property
    def opponent(self) -> TeamPreview:
        return (
            self.away
            if self.home.team_id == CLEVELAND_GUARDIANS_TEAM_ID
            else self.home
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_sha() -> str | None:
    """Short commit sha, recorded so a report can be tied to the code that
    produced it. Returns None outside a git checkout rather than failing."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _first_split_stat(payload: dict[str, Any]) -> dict[str, Any]:
    blocks = payload.get("stats") or []
    if not blocks:
        return {}
    splits = blocks[0].get("splits") or []
    return splits[0].get("stat", {}) if splits else {}


def _find_game(schedule_payload: dict[str, Any], *, team_id: int) -> dict[str, Any] | None:
    for day in schedule_payload.get("dates", []):
        for game in day.get("games", []):
            teams = game.get("teams", {})
            ids = {
                teams.get("home", {}).get("team", {}).get("id"),
                teams.get("away", {}).get("team", {}).get("id"),
            }
            if team_id in ids:
                return game
    return None


# ---------------------------------------------------------------------------
# Lineups
# ---------------------------------------------------------------------------


def _resolve_lineup(
    archiver: Archiver,
    game: dict[str, Any],
    *,
    side: str,
    team_id: int,
    season: int,
    as_of: date,
) -> tuple[list[dict[str, Any]], str, str]:
    """Return (players, source, note).

    MLB publishes no projected lineup. The official one appears via
    `hydrate=lineups` roughly three hours before first pitch. Before that we
    fall back to the most recent game's batting order and say so explicitly --
    a preview that silently presents a stale lineup as today's is worse than
    one that admits what it is showing.
    """
    key = "homePlayers" if side == "home" else "awayPlayers"
    posted = (game.get("lineups") or {}).get(key) or []
    if posted:
        return posted, "official", "Official lineup as posted by the club."

    # Fall back to the batting order from the team's most recent completed game.
    recent = api.schedule_range(
        archiver,
        start=date.fromordinal(as_of.toordinal() - 10),
        end=date.fromordinal(as_of.toordinal() - 1),
        team_id=team_id,
        hydrate="lineups,team",
    ).json()

    latest: tuple[date, list[dict[str, Any]]] | None = None
    for day in recent.get("dates", []):
        for past in day.get("games", []):
            teams = past.get("teams", {})
            for past_side in ("home", "away"):
                if teams.get(past_side, {}).get("team", {}).get("id") != team_id:
                    continue
                players = (past.get("lineups") or {}).get(
                    "homePlayers" if past_side == "home" else "awayPlayers"
                ) or []
                if not players:
                    continue
                game_day = date.fromisoformat(past["officialDate"])
                if latest is None or game_day > latest[0]:
                    latest = (game_day, players)

    if latest:
        return (
            latest[1],
            "fallback",
            f"Official lineup not yet posted. Showing the batting order from "
            f"the most recent game with a posted lineup ({latest[0].isoformat()}). "
            f"Today's order will differ.",
        )

    return [], "unavailable", "No lineup available for this game yet."


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


def _build_pitcher(
    archiver: Archiver,
    *,
    player_id: int,
    name: str,
    throws: str | None,
    season: int,
    as_of: date,
    fip_constant: float,
    arsenal_by_player: dict[int, list[dict[str, str]]],
    percentiles_by_player: dict[int, dict[str, str]],
    expected_by_player: dict[int, dict[str, str]],
) -> PitcherProfile:
    log_payload = api.game_log(
        archiver, person_id=player_id, season=season, group=api.GROUP_PITCHING
    ).json()
    rows = w.parse_game_logs(log_payload)

    season_stats = w.aggregate_pitching(
        [r for r in rows if r.game_date < as_of], fip_constant=fip_constant
    )

    is_starter = season_stats.get("gamesStarted", 0) > 0
    specs = w.STARTER_WINDOWS if is_starter else w.RELIEVER_WINDOWS
    window_lines = [
        WindowLine(
            label=spec.label,
            stats=w.aggregate_pitching(
                w.select(rows, spec, as_of=as_of), fip_constant=fip_constant
            ),
        )
        for spec in specs
    ]

    saber = _first_split_stat(
        api.sabermetrics(
            archiver, person_id=player_id, season=season, group=api.GROUP_PITCHING
        ).json()
    )

    arsenal = [
        {
            "pitch": row.get("pitch_name"),
            "usage": sv.to_number(row.get("pitch_usage")),
            "pitches": sv.to_number(row.get("pitches")),
            "whiff": sv.to_number(row.get("whiff_percent")),
            "put_away": sv.to_number(row.get("put_away")),
            "ba": sv.to_number(row.get("ba")),
            "slg": sv.to_number(row.get("slg")),
            "woba": sv.to_number(row.get("woba")),
            "xwoba": sv.to_number(row.get("est_woba")),
            "hard_hit": sv.to_number(row.get("hard_hit_percent")),
            "run_value_per_100": sv.to_number(row.get("run_value_per_100")),
        }
        for row in sorted(
            arsenal_by_player.get(player_id, []),
            key=lambda r: -(sv.to_number(r.get("pitch_usage")) or 0),
        )
    ]

    return PitcherProfile(
        player_id=player_id,
        name=name,
        throws=throws,
        season=season_stats,
        windows=window_lines,
        arsenal=arsenal,
        percentiles=percentiles_by_player.get(player_id, {}),
        expected=expected_by_player.get(player_id, {}),
        sabermetrics=saber,
    )


def _build_batter(
    archiver: Archiver,
    *,
    player_id: int,
    name: str,
    bats: str | None,
    position: str | None,
    batting_order: int | None,
    season: int,
    as_of: date,
    opposing_hand: str | None,
    expected_by_player: dict[int, dict[str, str]],
    percentiles_by_player: dict[int, dict[str, str]],
) -> BatterProfile:
    log_payload = api.game_log(
        archiver, person_id=player_id, season=season, group=api.GROUP_HITTING
    ).json()
    rows = w.parse_game_logs(log_payload)

    season_stats = w.aggregate_hitting([r for r in rows if r.game_date < as_of])
    window_lines = [
        WindowLine(label=spec.label, stats=w.aggregate_hitting(w.select(rows, spec, as_of=as_of)))
        for spec in w.HITTER_WINDOWS
    ]

    # Platoon split against the hand the opposing starter throws.
    vs_hand: dict[str, Any] = {}
    vs_hand_label = ""
    if opposing_hand in ("L", "R"):
        wanted = "vl" if opposing_hand == "L" else "vr"
        vs_hand_label = f"vs {'LHP' if opposing_hand == 'L' else 'RHP'}"
        payload = api.stat_splits(
            archiver,
            person_id=player_id,
            season=season,
            group=api.GROUP_HITTING,
            split_codes=DEFAULT_SPLIT_CODES,
        ).json()
        for block in payload.get("stats") or []:
            for split in block.get("splits") or []:
                if (split.get("split") or {}).get("code") == wanted:
                    vs_hand = split.get("stat", {})
                    break

    saber = _first_split_stat(
        api.sabermetrics(
            archiver, person_id=player_id, season=season, group=api.GROUP_HITTING
        ).json()
    )

    return BatterProfile(
        player_id=player_id,
        name=name,
        bats=bats,
        position=position,
        batting_order=batting_order,
        season=season_stats,
        windows=window_lines,
        vs_hand=vs_hand,
        vs_hand_label=vs_hand_label,
        expected=expected_by_player.get(player_id, {}),
        percentiles=percentiles_by_player.get(player_id, {}),
        sabermetrics=saber,
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
        raise LookupError(
            f"no game found for team {team_id} on {on.isoformat()}"
        )

    game_pk = game["gamePk"]
    teams = game["teams"]

    # League constants first: FIP is meaningless without the season's constant.
    league = lc.derive(
        lc.parse_league_totals(
            api.league_pitching_totals(archiver, season=season).json(), season=season
        )
    )

    # Savant leaderboards are fetched once and indexed, rather than per player.
    arsenal_by_player = sv.group_by_player(
        sv.parse_csv(sv.pitch_arsenal_stats(archiver, year=season, minimum=1))
    )
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

    # Probable starters, and the handedness each lineup will face.
    starters: dict[str, dict[str, Any]] = {}
    for side in ("home", "away"):
        probable = teams[side].get("probablePitcher")
        if probable:
            starters[side] = probable

    starter_ids = [p["id"] for p in starters.values()]
    bios: dict[int, dict[str, Any]] = {}
    if starter_ids:
        for result in api.people(archiver, person_ids=starter_ids):
            for person in result.json().get("people", []):
                bios[person["id"]] = person

    # Resolve both lineups first, then look up every batter's bio in one
    # batched call. The lineups hydrate carries name and position but not
    # batSide, and handedness is the whole basis of the platoon matchup this
    # report is built around -- so it cannot be left blank.
    resolved_lineups: dict[str, tuple[list[dict[str, Any]], str, str]] = {}
    for side in ("home", "away"):
        resolved_lineups[side] = _resolve_lineup(
            archiver,
            game,
            side=side,
            team_id=teams[side]["team"]["id"],
            season=season,
            as_of=on,
        )

    batter_ids = [
        person["id"]
        for players, _, _ in resolved_lineups.values()
        for person in players
    ]
    if batter_ids:
        for result in api.people(archiver, person_ids=batter_ids):
            for person in result.json().get("people", []):
                bios[person["id"]] = person

    previews: dict[str, TeamPreview] = {}
    for side in ("home", "away"):
        opposite = "away" if side == "home" else "home"
        team_info = teams[side]["team"]
        record = teams[side].get("leagueRecord", {})

        starter_profile: PitcherProfile | None = None
        if side in starters:
            sid = starters[side]["id"]
            bio = bios.get(sid, {})
            starter_profile = _build_pitcher(
                archiver,
                player_id=sid,
                name=starters[side].get("fullName", bio.get("fullName", "Unknown")),
                throws=(bio.get("pitchHand") or {}).get("code"),
                season=season,
                as_of=on,
                fip_constant=league.fip_constant,
                arsenal_by_player=arsenal_by_player,
                percentiles_by_player=pitcher_percentiles,
                expected_by_player=pitcher_expected,
            )

        # The hand this side's hitters will face is the *opposing* starter's.
        opposing_starter_id = (
            starters[opposite]["id"] if opposite in starters else None
        )
        opposing_hand = None
        if opposing_starter_id is not None:
            opposing_hand = (
                bios.get(opposing_starter_id, {}).get("pitchHand") or {}
            ).get("code")

        players, lineup_source, note = resolved_lineups[side]

        lineup: list[BatterProfile] = []
        for order, person in enumerate(players, start=1):
            bio = bios.get(person["id"], {})
            lineup.append(
                _build_batter(
                    archiver,
                    player_id=person["id"],
                    name=person.get("fullName", bio.get("fullName", "Unknown")),
                    bats=(
                        (person.get("batSide") or bio.get("batSide") or {})
                    ).get("code"),
                    position=(
                        person.get("primaryPosition")
                        or bio.get("primaryPosition")
                        or {}
                    ).get("abbreviation"),
                    batting_order=order,
                    season=season,
                    as_of=on,
                    opposing_hand=opposing_hand,
                    expected_by_player=batter_expected,
                    percentiles_by_player=batter_percentiles,
                )
            )

        previews[side] = TeamPreview(
            team_id=team_info["id"],
            name=team_info.get("name", ""),
            abbreviation=team_info.get("abbreviation", ""),
            record=f"{record.get('wins', 0)}-{record.get('losses', 0)}",
            is_home=(side == "home"),
            starter=starter_profile,
            lineup=lineup,
            lineup_source=lineup_source,
            lineup_source_note=note,
        )

    # Weather and umpires only exist on the live feed.
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
        home=previews["home"],
        away=previews["away"],
        league=league,
        provenance=[r.provenance() for r in archiver.written],
    )
