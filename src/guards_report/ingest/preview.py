"""Assemble a game preview: fetch, compute, and return a typed bundle.

Five pages: a team comparison that frames the matchup, then each club's
position players and pitchers, one box per player.

Fetching is aggressively batched. The statsapi `stats(...)` hydrate accepts a
list of stat types, a list of situation codes, and a list of personIds all at
once, so season lines, sabermetrics, a dozen situational splits and zone data
for 52 players collapse into a handful of requests rather than one per player
per stat type.

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
    HITTER_SPLIT_CODES,
    PITCHER_SPLIT_CODES,
    REPO_ROOT,
    Settings,
)
from guards_report.metrics import highlights as hl
from guards_report.metrics import league_averages as la
from guards_report.metrics import league_constants as lc
from guards_report.metrics import team_context as tc
from guards_report.metrics import trends as tr
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
    player_id: int
    name: str
    hand: str | None
    position: str | None
    jersey: str | None
    is_probable_starter: bool = False

    season: dict[str, Any] = field(default_factory=dict)
    season_deltas: dict[str, Any] = field(default_factory=dict)
    windows: list[WindowLine] = field(default_factory=list)
    trend: tr.TrendSeries | None = None

    percentiles: dict[str, str] = field(default_factory=dict)
    expected: dict[str, str] = field(default_factory=dict)
    sabermetrics: dict[str, Any] = field(default_factory=dict)
    arsenal: list[dict[str, Any]] = field(default_factory=list)
    zone_grids: dict[str, zn.ZoneGrid] = field(default_factory=dict)

    # Statcast profile layers
    batted_ball: dict[str, float | None] = field(default_factory=dict)
    bat_tracking: dict[str, float | None] = field(default_factory=dict)
    fielding: dict[str, float | None] = field(default_factory=dict)
    running: dict[str, float | None] = field(default_factory=dict)

    # Every requested situational split, keyed by situation code.
    situational: dict[str, dict[str, Any]] = field(default_factory=dict)

    vs_hand: dict[str, Any] = field(default_factory=dict)
    vs_hand_label: str = ""

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
    profile: tc.TeamProfile | None = None


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
    highlights: list[hl.Highlight] = field(default_factory=list)

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

    @property
    def home_profile(self) -> tc.TeamProfile | None:
        return self.home.profile

    @property
    def away_profile(self) -> tc.TeamProfile | None:
        return self.away.profile


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def _stat_block(person: dict[str, Any], type_name: str) -> list[dict[str, Any]]:
    for block in person.get("stats") or []:
        if (block.get("type") or {}).get("displayName") == type_name:
            return block.get("splits") or []
    return []


def _first_stat(person: dict[str, Any], type_name: str) -> dict[str, Any]:
    splits = _stat_block(person, type_name)
    return splits[0].get("stat", {}) if splits else {}


def _situational(person: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index every returned situational split by its situation code.

    A player traded mid-season can return more than one split for the same
    code, one per club. We keep the one with the most plate appearances, which
    is the fuller sample, rather than whichever happened to come first.
    """
    out: dict[str, dict[str, Any]] = {}
    for split in _stat_block(person, "statSplits"):
        code = (split.get("split") or {}).get("code")
        if not code:
            continue
        stat = split.get("stat") or {}
        existing = out.get(code)
        if existing is None:
            out[code] = stat
            continue
        current = int(existing.get("plateAppearances") or existing.get("battersFaced") or 0)
        candidate = int(stat.get("plateAppearances") or stat.get("battersFaced") or 0)
        if candidate > current:
            out[code] = stat
    return out


def _merge_people(results: list[Any]) -> dict[int, dict[str, Any]]:
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

    Two-way players appear in both lists deliberately: they are two different
    players for scouting purposes and belong on both pages.
    """
    batters, pitchers = [], []
    for entry in payload.get("roster", []):
        position_type = (entry.get("position") or {}).get("type")
        if position_type == "Pitcher":
            pitchers.append(entry)
        elif position_type in POSITION_PLAYER_TYPES:
            batters.append(entry)
        if position_type == "Two-Way Player":
            pitchers.append(entry)
    return batters, pitchers


# ---------------------------------------------------------------------------
# Box construction
# ---------------------------------------------------------------------------

HITTER_DELTA_KEYS = ("avg", "obp", "slg", "ops", "iso", "babip", "kPct", "bbPct")
PITCHER_DELTA_KEYS = (
    "era", "whip", "fip", "kPer9", "bbPer9", "hrPer9", "kPct", "bbPct",
    "kMinusBbPct",
)
HITTER_LOWER_BETTER = frozenset({"kPct"})
PITCHER_LOWER_BETTER = frozenset({"era", "whip", "fip", "bbPer9", "hrPer9", "bbPct"})


def _deltas(stats, league, keys, lower_better) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        result = la.delta(
            stats.get(key), league.get(key), lower_is_better=key in lower_better
        )
        if result is not None:
            out[key] = result
    return out


def _pick(row: dict[str, str] | None, fields: tuple[str, ...]) -> dict[str, float | None]:
    """Pull selected numeric fields out of a leaderboard row.

    Always returns every requested key, mapping to None when the player has no
    row or the cell is blank. A stable shape means the renderer can read any
    key without guarding for its existence, and a missing value stays
    distinguishable from a zero.
    """
    if not row:
        return {name: None for name in fields}
    return {name: sv.to_number(row.get(name)) for name in fields}


BATTED_BALL_FIELDS = (
    "bbe", "gb_rate", "ld_rate", "fb_rate", "pu_rate", "air_rate",
    "pull_rate", "straight_rate", "oppo_rate",
    "pull_gb_rate", "straight_gb_rate", "oppo_gb_rate",
    "pull_air_rate", "straight_air_rate", "oppo_air_rate",
)
BAT_TRACKING_FIELDS = (
    "avg_bat_speed", "swing_length", "hard_swing_rate",
    "squared_up_per_swing", "squared_up_per_bat_contact",
    "blast_per_swing", "blast_per_bat_contact", "whiff_per_swing", "swords",
)
FIELDING_FIELDS = (
    "outs_above_average", "fielding_runs_prevented",
    "outs_above_average_rhh", "outs_above_average_lhh",
)
RUNNING_FIELDS = ("sprint_speed", "hp_to_1b", "bolts", "competitive_runs")


def _arsenal_rows(rows, league_by_pitch) -> list[dict[str, Any]]:
    out = []
    for row in sorted(rows, key=lambda r: -(sv.to_number(r.get("pitch_usage")) or 0)):
        pitch_type = row.get("pitch_type") or ""
        benchmark = league_by_pitch.get(pitch_type, {})
        out.append({
            "pitch": row.get("pitch_name"), "pitch_type": pitch_type,
            "usage": sv.to_number(row.get("pitch_usage")),
            "pitches": sv.to_number(row.get("pitches")),
            "pa": sv.to_number(row.get("pa")),
            "whiff": sv.to_number(row.get("whiff_percent")),
            "put_away": sv.to_number(row.get("put_away")),
            "ba": sv.to_number(row.get("ba")), "slg": sv.to_number(row.get("slg")),
            "woba": sv.to_number(row.get("woba")),
            "xwoba": sv.to_number(row.get("est_woba")),
            "hard_hit": sv.to_number(row.get("hard_hit_percent")),
            "run_value_per_100": sv.to_number(row.get("run_value_per_100")),
            "lg_whiff": benchmark.get("whiff_percent"),
            "lg_xwoba": benchmark.get("est_woba"),
        })
    return out


def _build_batter_box(person, entry, *, as_of, opposing_hand, league_hitting,
                      arsenal_by_player, league_by_pitch, savant) -> PlayerBox:
    pid = person["id"]
    rows = w.parse_game_logs({"stats": [{"splits": _stat_block(person, "gameLog")}]})
    prior = [r for r in rows if r.game_date < as_of]

    season_stats = w.aggregate_hitting(prior)
    windows = []
    for spec in w.HITTER_WINDOWS:
        stats = w.aggregate_hitting(w.select(rows, spec, as_of=as_of))
        windows.append(WindowLine(
            label=spec.label, stats=stats,
            deltas=_deltas(stats, league_hitting, HITTER_DELTA_KEYS, HITTER_LOWER_BETTER),
        ))

    situational = _situational(person)
    vs_hand, vs_hand_label = {}, ""
    if opposing_hand in ("L", "R"):
        code = "vl" if opposing_hand == "L" else "vr"
        vs_hand_label = f"vs {'LHP' if opposing_hand == 'L' else 'RHP'}"
        vs_hand = situational.get(code, {})

    return PlayerBox(
        player_id=pid, name=person.get("fullName", "Unknown"),
        hand=(person.get("batSide") or {}).get("code"),
        position=(entry.get("position") or {}).get("abbreviation"),
        jersey=entry.get("jerseyNumber"),
        season=season_stats,
        season_deltas=_deltas(season_stats, league_hitting, HITTER_DELTA_KEYS,
                              HITTER_LOWER_BETTER),
        windows=windows,
        trend=tr.hitter_ops_trend(rows, as_of=as_of),
        percentiles=savant["batter_percentiles"].get(pid, {}),
        expected=savant["batter_expected"].get(pid, {}),
        sabermetrics=_first_stat(person, "sabermetrics"),
        arsenal=_arsenal_rows(arsenal_by_player.get(pid, []), league_by_pitch),
        zone_grids=zn.parse_zones(person),
        batted_ball=_pick(savant["batted_ball"].get(pid), BATTED_BALL_FIELDS),
        bat_tracking=_pick(savant["bat_tracking"].get(pid), BAT_TRACKING_FIELDS),
        fielding=_pick(savant["fielding"].get(pid), FIELDING_FIELDS),
        running=_pick(savant["running"].get(pid), RUNNING_FIELDS),
        situational=situational,
        vs_hand=vs_hand, vs_hand_label=vs_hand_label,
    )


def _build_pitcher_box(person, entry, *, as_of, fip_constant, league_pitching,
                       arsenal_by_player, league_by_pitch, savant,
                       probable_starter_id) -> PlayerBox:
    pid = person["id"]
    rows = w.parse_game_logs({"stats": [{"splits": _stat_block(person, "gameLog")}]})
    prior = [r for r in rows if r.game_date < as_of]

    season_stats = w.aggregate_pitching(prior, fip_constant=fip_constant)
    windows = []
    for spec in w.PITCHER_WINDOWS:
        stats = w.aggregate_pitching(w.select(rows, spec, as_of=as_of),
                                     fip_constant=fip_constant)
        windows.append(WindowLine(
            label=spec.label, stats=stats,
            deltas=_deltas(stats, league_pitching, PITCHER_DELTA_KEYS,
                           PITCHER_LOWER_BETTER),
        ))

    games = season_stats.get("games", 0) or 0
    starts = season_stats.get("gamesStarted", 0) or 0
    if starts >= max(1, games * 0.5):
        role = "SP"
    elif season_stats.get("saves", 0):
        role = "RP (closer)"
    else:
        role = "RP"

    return PlayerBox(
        player_id=pid, name=person.get("fullName", "Unknown"),
        hand=(person.get("pitchHand") or {}).get("code"),
        position="P", jersey=entry.get("jerseyNumber"),
        is_probable_starter=(pid == probable_starter_id),
        season=season_stats,
        season_deltas=_deltas(season_stats, league_pitching, PITCHER_DELTA_KEYS,
                              PITCHER_LOWER_BETTER),
        windows=windows,
        trend=tr.pitcher_era_trend(rows, as_of=as_of),
        percentiles=savant["pitcher_percentiles"].get(pid, {}),
        expected=savant["pitcher_expected"].get(pid, {}),
        sabermetrics=_first_stat(person, "sabermetrics"),
        arsenal=_arsenal_rows(arsenal_by_player.get(pid, []), league_by_pitch),
        zone_grids=zn.parse_zones(person),
        situational=_situational(person),
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

    # -- league context -----------------------------------------------------
    pitching_totals = api.league_pitching_totals(archiver, season=season).json()
    hitting_totals = api.league_hitting_totals(archiver, season=season).json()

    league = lc.derive(lc.parse_league_totals(pitching_totals, season=season))
    league_pitching = la.pitching_from_payload(
        pitching_totals, season=season, fip_constant=league.fip_constant
    )
    league_hitting = la.hitting_from_payload(hitting_totals, season=season)

    # -- team ranks and standings ------------------------------------------
    standings = tc.parse_standings(api.standings(archiver, season=season).json())
    team_hitting = tc.hitting_metrics(hitting_totals)
    team_pitching = tc.pitching_metrics(
        pitching_totals, fip_constant=league.fip_constant
    )
    hitting_ranks = tc.rank_all(team_hitting, lower_is_better=tc.HITTING_LOWER_BETTER)
    pitching_ranks = tc.rank_all(team_pitching, lower_is_better=tc.PITCHING_LOWER_BETTER)

    # -- Savant leaderboards, fetched once and indexed ----------------------
    pitcher_arsenal_rows = sv.parse_csv(sv.pitch_arsenal_stats(
        archiver, year=season, minimum=1, player_type=sv.TYPE_PITCHER))
    batter_arsenal_rows = sv.parse_csv(sv.pitch_arsenal_stats(
        archiver, year=season, minimum=1, player_type=sv.TYPE_BATTER))

    savant = {
        "pitcher_percentiles": sv.index_by_player(sv.parse_csv(
            sv.percentile_rankings(archiver, year=season, player_type=sv.TYPE_PITCHER))),
        "batter_percentiles": sv.index_by_player(sv.parse_csv(
            sv.percentile_rankings(archiver, year=season, player_type=sv.TYPE_BATTER))),
        "pitcher_expected": sv.index_by_player(sv.parse_csv(
            sv.expected_statistics(archiver, year=season,
                                   player_type=sv.TYPE_PITCHER, minimum=1))),
        "batter_expected": sv.index_by_player(sv.parse_csv(
            sv.expected_statistics(archiver, year=season,
                                   player_type=sv.TYPE_BATTER, minimum=1))),
        "batted_ball": sv.index_by_player(sv.parse_csv(
            sv.batted_ball(archiver, year=season, minimum=10))),
        "bat_tracking": sv.index_by_player(sv.parse_csv(
            sv.bat_tracking(archiver, year=season, minimum=10))),
        "fielding": sv.index_by_player(sv.parse_csv(
            sv.outs_above_average(archiver, year=season, minimum=1))),
        "running": sv.index_by_player(sv.parse_csv(
            sv.sprint_speed(archiver, year=season, minimum=1))),
    }

    pitcher_arsenal = sv.group_by_player(pitcher_arsenal_rows)
    batter_arsenal = sv.group_by_player(batter_arsenal_rows)
    league_pitch_thrown = sv.league_average_by_pitch_type(pitcher_arsenal_rows)
    league_pitch_faced = sv.league_average_by_pitch_type(batter_arsenal_rows)

    # -- rosters ------------------------------------------------------------
    roster_entries: dict[str, tuple[list[dict], list[dict]]] = {}
    for side in ("home", "away"):
        roster_entries[side] = _split_roster(api.active_roster(
            archiver, team_id=teams[side]["team"]["id"], season=season).json())

    probable: dict[str, int | None] = {}
    for side in ("home", "away"):
        pitcher = teams[side].get("probablePitcher")
        probable[side] = pitcher["id"] if pitcher else None

    all_batter_ids = [e["person"]["id"] for side in ("home", "away")
                      for e in roster_entries[side][0]]
    all_pitcher_ids = [e["person"]["id"] for side in ("home", "away")
                       for e in roster_entries[side][1]]

    batter_people = _merge_people(
        api.people_with_stats(
            archiver, person_ids=all_batter_ids, group=api.GROUP_HITTING,
            stat_types=[api.STAT_SEASON, api.STAT_SABERMETRICS, api.STAT_SPLITS,
                        api.STAT_HOT_COLD_ZONES],
            season=season, sit_codes=HITTER_SPLIT_CODES,
        )
        + api.people_with_stats(
            archiver, person_ids=all_batter_ids, group=api.GROUP_HITTING,
            stat_types=[api.STAT_GAME_LOG], season=season,
            batch_size=api.GAMELOG_BATCH_SIZE,
        )
    )

    pitcher_people = _merge_people(
        api.people_with_stats(
            archiver, person_ids=all_pitcher_ids, group=api.GROUP_PITCHING,
            stat_types=[api.STAT_SEASON, api.STAT_SABERMETRICS, api.STAT_SPLITS,
                        api.STAT_HOT_COLD_ZONES],
            season=season, sit_codes=PITCHER_SPLIT_CODES,
        )
        + api.people_with_stats(
            archiver, person_ids=all_pitcher_ids, group=api.GROUP_PITCHING,
            stat_types=[api.STAT_GAME_LOG], season=season,
            batch_size=api.GAMELOG_BATCH_SIZE,
        )
    )

    hands: dict[str, str | None] = {}
    for side in ("home", "away"):
        pid = probable[side]
        hands[side] = ((pitcher_people.get(pid, {}).get("pitchHand") or {}).get("code")
                       if pid else None)

    sections: dict[str, TeamSection] = {}
    for side in ("home", "away"):
        opposite = "away" if side == "home" else "home"
        team_info = teams[side]["team"]
        record = teams[side].get("leagueRecord", {})
        batter_entries, pitcher_entries = roster_entries[side]
        tid = team_info["id"]

        batters = [
            _build_batter_box(
                batter_people[e["person"]["id"]], e, as_of=on,
                opposing_hand=hands[opposite], league_hitting=league_hitting,
                arsenal_by_player=batter_arsenal,
                league_by_pitch=league_pitch_faced, savant=savant,
            )
            for e in batter_entries if e["person"]["id"] in batter_people
        ]
        batters.sort(key=lambda b: -(b.season.get("plateAppearances") or 0))

        pitchers = [
            _build_pitcher_box(
                pitcher_people[e["person"]["id"]], e, as_of=on,
                fip_constant=league.fip_constant, league_pitching=league_pitching,
                arsenal_by_player=pitcher_arsenal,
                league_by_pitch=league_pitch_thrown, savant=savant,
                probable_starter_id=probable[side],
            )
            for e in pitcher_entries if e["person"]["id"] in pitcher_people
        ]

        def _order(box: PlayerBox) -> tuple[int, int, int]:
            if box.is_probable_starter:
                group = 0
            elif box.role == "SP":
                group = 2
            else:
                group = 1
            rest = (box.availability.days_rest
                    if box.availability and box.availability.days_rest is not None
                    else 99)
            return (group, -rest, -(box.season.get("outs") or 0))

        pitchers.sort(key=_order)

        # Team-level bullpen state: how many arms are genuinely available, and
        # how hard the pen has been worked. Decides late innings, and is never
        # visible from the individual rows.
        relievers = [p for p in pitchers
                     if not p.is_probable_starter and p.role.startswith("RP")]
        profile = tc.TeamProfile(
            team_id=tid, name=team_info.get("name", ""),
            abbreviation=team_info.get("abbreviation", ""),
            record=standings.get(tid),
            hitting=tc.build_ranked(tid, team_hitting, hitting_ranks,
                                    tc.HITTING_DISPLAY, tc.HITTING_LOWER_BETTER),
            pitching=tc.build_ranked(tid, team_pitching, pitching_ranks,
                                     tc.PITCHING_DISPLAY, tc.PITCHING_LOWER_BETTER),
            run_differential=(
                int((team_hitting.get(tid, {}) or {}).get("runs") or 0)
                - int((team_pitching.get(tid, {}) or {}).get("runsAllowed") or 0)
            ),
            bullpen_pitches_last_3=sum(
                p.availability.pitches_last_3 for p in relievers if p.availability),
            bullpen_available=sum(
                1 for p in relievers
                if p.availability and (p.availability.days_rest or 0) >= 2),
            bullpen_total=len(relievers),
            lineup_hand_counts={
                hand: sum(1 for b in batters[:13] if b.hand == hand)
                for hand in ("L", "R", "S")
            },
        )

        sections[side] = TeamSection(
            team_id=tid, name=team_info.get("name", ""),
            abbreviation=team_info.get("abbreviation", ""),
            record=f"{record.get('wins', 0)}-{record.get('losses', 0)}",
            is_home=(side == "home"), batters=batters, pitchers=pitchers,
            opposing_hand=hands[opposite], profile=profile,
        )

    weather: dict[str, Any] = {}
    try:
        feed = api.game_feed(archiver, game_pk=game_pk).json()
        weather = (feed.get("gameData") or {}).get("weather") or {}
    except Exception:
        weather = {}

    game_datetime = None
    if game.get("gameDate"):
        game_datetime = datetime.fromisoformat(game["gameDate"].replace("Z", "+00:00"))

    bundle = ReportBundle(
        run_id=uuid.uuid4().hex[:12],
        generated_at=datetime.now(timezone.utc),
        as_of_date=on, git_sha=_git_sha(), game_pk=game_pk,
        game_date=date.fromisoformat(game["officialDate"]),
        game_datetime=game_datetime,
        venue_name=(game.get("venue") or {}).get("name", ""),
        status=(game.get("status") or {}).get("detailedState", ""),
        weather=weather, season=season,
        home=sections["home"], away=sections["away"],
        league=league, league_hitting=league_hitting, league_pitching=league_pitching,
        provenance=[r.provenance() for r in archiver.written],
    )
    bundle.highlights = hl.build(bundle)
    return bundle
