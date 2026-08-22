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
import sys
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

from guards_report.config import (
    CLEVELAND_GUARDIANS_TEAM_ID,
    COMPETITIVE_GAME_TYPES,
    HITTER_SPLIT_CODES,
    PITCHER_SPLIT_CODES,
    REPO_ROOT,
    Settings,
)
from guards_report.metrics import highlights as hl
from guards_report.metrics import league_averages as la
from guards_report.metrics import series as sr
from guards_report.metrics import statcast as sc
from guards_report.metrics import league_constants as lc
from guards_report.metrics import team_context as tc
from guards_report.metrics import trends as tr
from guards_report.metrics import windows as w
from guards_report.metrics import zones as zn
from guards_report.sources import mlb_statsapi as api
from guards_report.sources import savant as sv
from guards_report.sources.http import Archiver

# "Hitter" is the source's position type for a designated hitter. Leaving it
# out silently dropped any player listed at DH -- Bryce Eldridge vanished from
# a Giants report while the roster count still looked plausible.
POSITION_PLAYER_TYPES = {
    "Catcher", "Infielder", "Outfielder", "Two-Way Player", "Hitter",
}


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

    # This player's line across the games of the current series, when one is
    # already under way. None means he has not appeared in it.
    series_line: Any | None = None
    # Career totals, for the context a single season cannot give: whether this
    # is a rookie's hot month or a ten-year veteran playing to his norm.
    career: dict[str, Any] = field(default_factory=dict)

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

    # Pitch-level derived: {hand: {metric: ZoneChart}} and a spray chart.
    statcast_zones: dict[str, dict[str, Any]] = field(default_factory=dict)
    spray: Any = None

    # Lineup position when an official lineup has been posted. Players not in
    # it keep a None order and sort to the bottom rather than being dropped.
    batting_order: int | None = None

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
    lineup_source: str = "none"
    lineup_note: str = ""


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
    series: tc.SeriesContext | None = None
    # Completed games of the current series, most recent first, each with its
    # line score, decisions and standout performances.
    series_boxes: list[Any] = field(default_factory=list)
    # Projection for this game, when a fitted model is available. None means the
    # models have not been trained yet -- the report is complete without them.
    projection: Any | None = None
    # Written analysis keyed by subject id, attached after the numbers are
    # final. The report renders correctly whether or not this is populated --
    # the analysis layer is additive and never blocks a build.
    analyses: dict[str, Any] = field(default_factory=dict)
    analysis_summary: dict[str, Any] = field(default_factory=dict)
    # League-wide leaderboard values, kept as populations rather than only as
    # the means beside each box. A player's speed and defense have to be graded
    # against every other player, and twenty-six men on one card is not a
    # league -- so the arrays travel with the bundle rather than being refetched
    # by whatever needs to rank against them.
    savant_populations: dict[str, list[float]] = field(default_factory=dict)
    # League benchmarks for the Savant blocks, keyed by block name, so every
    # displayed stat has something to be measured against.
    savant_benchmarks: dict[str, dict[str, float | None]] = field(
        default_factory=dict
    )

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
    """The club's game on this date, ignoring exhibitions.

    A March date can carry a spring training game, which is not what this
    report covers: the season and recent-form numbers behind it count only
    regular season and postseason play, so previewing an exhibition would
    describe a game with data drawn from different games entirely.
    """
    for day in payload.get("dates", []):
        for game in day.get("games", []):
            if game.get("gameType") not in COMPETITIVE_GAME_TYPES:
                continue
            teams = game.get("teams", {})
            ids = {
                teams.get("home", {}).get("team", {}).get("id"),
                teams.get("away", {}).get("team", {}).get("id"),
            }
            if team_id in ids:
                return game
    return None


def _lineup_order(game: dict[str, Any], side: str) -> tuple[dict[int, int], str, str]:
    """Batting order by player id, if an official lineup has been posted.

    MLB publishes no projected lineup and the official one appears roughly
    three hours before first pitch. When it is absent we say so and fall back
    to sorting by playing time; we never invent an order, and we never drop a
    player who is not in it -- the bench simply sorts below the nine.
    """
    players = (game.get("lineups") or {}).get(
        "homePlayers" if side == "home" else "awayPlayers"
    ) or []
    if not players:
        return (
            {},
            "none",
            "Official lineup not yet posted — MLB releases it about three hours "
            "before first pitch. Position players are ordered by playing time; "
            "no batting order is implied.",
        )
    return (
        {person["id"]: index for index, person in enumerate(players, start=1)},
        "official",
        "Official lineup as posted by the club. Players not in the lineup "
        "follow below, ordered by playing time.",
    )


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
        else:
            # An unrecognised position type must not vanish. Dropping a player
            # is invisible in the output -- the roster simply looks one short --
            # so anything unfamiliar is treated as a position player, which is
            # true of every non-pitcher the source lists.
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

    Fractional rates are converted to percentages on the way through, so a
    single convention holds from here on.
    """
    if not row:
        return {name: None for name in fields}

    return {
        name: sv.scale_rate(name, sv.to_number(row.get(name)))
        for name in fields
    }


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
    career = _first_stat(person, "career")
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
        season=season_stats, career=career,
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
    career = _first_stat(person, "career")
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
        season=season_stats, career=career,
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


def _statcast_for(
    archiver: Archiver, *, player_id: int, season: int, perspective: str,
    bats: str | None, as_of: date,
) -> tuple[dict[str, Any], Any]:
    """Fetch and aggregate one player's pitch-level season.

    `as_of` bounds the data to games completed before the report date, matching
    the game-log windows. Without it the charts would include pitches from a
    game still in progress while every other number on the page excluded it.

    Returns zone charts and, for hitters, a spray chart. A failure here is not
    fatal: the rest of the box is still worth showing, so we return empties and
    let the renderer omit those sections.
    """
    try:
        result = sc.parse_pitches(
            sv.player_pitches(
                archiver, player_id=player_id, year=season, perspective=perspective
            ).text(),
            before=as_of,
        )
    except Exception:
        return {}, None

    zones = sc.build_zone_charts(result, perspective=perspective)
    spray = sc.build_spray(result, bats=bats) if perspective == "batter" else None
    return zones, spray


def _current_series_boxes(
    archiver: Archiver, *, team_id: int, game: dict[str, Any], on: date
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Completed games of the series today's game belongs to.

    Returns the parsed box scores and the raw payloads they came from, so the
    per-player series lines can be aggregated without fetching each game twice.

    The series is walked backwards from today using the source's own
    `seriesGameNumber`, which counts 1..N within a set: today's game 3 means
    games 2 and 1 are the two that came before it. That is more reliable than
    guessing by date, which breaks on off-days and doubleheaders.

    Only finished games are fetched. A game in progress has no final line
    score, and this report does not report on games that are still being
    played.
    """
    number = game.get("seriesGameNumber")
    if not number or number < 2:
        return [], []      # opener: nothing has been played in this set yet

    # A series is a consecutive set against one club, so the opponent is the
    # constraint that matters. `seriesGameNumber` alone is not enough: it
    # restarts at 1 every series, so matching on it picked up game 1 and 2 of
    # the *previous* set against a different team.
    teams = game.get("teams") or {}
    ours = {
        side for side in ("home", "away")
        if ((teams.get(side) or {}).get("team") or {}).get("id") == team_id
    }
    other = "away" if "home" in ours else "home"
    opponent_id = ((teams.get(other) or {}).get("team") or {}).get("id")
    if opponent_id is None:
        return [], []

    # Look back far enough to cover a four-game set containing an off-day.
    payload = api.series_games(
        archiver, team_id=team_id, start=on - timedelta(days=9), end=on,
    ).json()

    def _opponent_of(entry: dict[str, Any]) -> int | None:
        entry_teams = entry.get("teams") or {}
        for side in ("home", "away"):
            side_id = ((entry_teams.get(side) or {}).get("team") or {}).get("id")
            if side_id != team_id:
                return side_id
        return None

    candidates = [
        g
        for day in payload.get("dates", [])
        for g in day.get("games", [])
        if g.get("gamePk") != game.get("gamePk")
        and _opponent_of(g) == opponent_id
        and (g.get("status") or {}).get("abstractGameState") == "Final"
        and g.get("seriesGameNumber") is not None
        and date.fromisoformat(g["officialDate"]) < on
    ]
    # Newest first, then walk back while the game number decrements by one.
    candidates.sort(
        key=lambda g: (g["officialDate"], g["seriesGameNumber"]), reverse=True
    )

    chosen: list[dict[str, Any]] = []
    expected = number - 1
    for entry in candidates:
        if entry["seriesGameNumber"] != expected:
            break          # a gap means the earlier games are a different set
        chosen.append(entry)
        expected -= 1
        if expected < 1:
            break
    chosen.reverse()       # present them in the order they were played

    boxes, payloads = [], []
    for entry in chosen:
        pk = entry["gamePk"]
        try:
            boxscore = api.game_boxscore(archiver, game_pk=pk).json()
            boxes.append(
                sr.parse_game_box(
                    schedule_game=entry,
                    boxscore=boxscore,
                    linescore=api.game_linescore(archiver, game_pk=pk).json(),
                )
            )
            payloads.append(boxscore)
        except Exception:
            # One unreadable game must not cost the whole section.
            continue
    return boxes, payloads


def _f5_history(settings):
    """The stored first-five starter table, or None when it has not been built."""
    try:
        import pandas as pd

        path = settings.raw_archive_dir.parent / "models" / "f5_starter.parquet"
        return pd.read_parquet(path) if path.exists() else None
    except Exception:  # noqa: BLE001 -- an absent table degrades the block only
        return None


def _plate_appearances(settings, on):
    """This season's plate appearances, for carrying talent forward.

    Read straight from the cached pitch corpus and projected to the columns the
    talent model needs, which keeps it seconds rather than minutes. A missing
    corpus is not fatal: the projection falls back to the stored prior and says
    so, exactly as it does for an unannounced starter.
    """
    try:
        from guards_report.projections import pa as pa_module

        directory = settings.raw_archive_dir.parent / "pitches"
        if not directory.exists():
            return None
        return pa_module.load(directory, seasons={on.year})
    except Exception as exc:  # noqa: BLE001 -- reported, never fatal
        print(f"  warning: plate appearances unavailable ({exc})", file=sys.stderr)
        return None


def _posted_lineup(section) -> list[int]:
    """Batting order from the official card, when one has been posted.

    Returns an empty list rather than a guess when it has not. The projection
    then values the side from team form instead, and the page says which.
    """
    if getattr(section, "lineup_source", "none") != "official":
        return []
    ordered = [
        box for box in section.batters
        if getattr(box, "batting_order", None) is not None
    ]
    ordered.sort(key=lambda box: box.batting_order)
    return [box.player_id for box in ordered[:9]]


def build_preview(
    settings: Settings,
    *,
    on: date,
    team_id: int = CLEVELAND_GUARDIANS_TEAM_ID,
    include_statcast: bool = True,
) -> ReportBundle:
    archiver = Archiver(root=settings.raw_archive_dir)
    season = on.year

    schedule_payload = api.schedule(archiver, on=on, team_id=team_id).json()
    game = _find_game(schedule_payload, team_id=team_id)
    if game is None:
        exhibition = any(
            g.get("gameType") not in COMPETITIVE_GAME_TYPES
            for d in schedule_payload.get("dates", [])
            for g in d.get("games", [])
        )
        raise LookupError(
            f"no regular season or postseason game for team {team_id} on "
            f"{on.isoformat()}"
            + (" (only spring training or exhibition games scheduled)"
               if exhibition else "")
        )

    game_pk = game["gamePk"]
    teams = game["teams"]

    # -- league context -----------------------------------------------------
    # Bounded to games finished before the report date, so the benchmarks and
    # the FIP constant do not drift while a game is being played.
    through = on - timedelta(days=1)
    pitching_totals = api.league_pitching_totals(
        archiver, season=season, through=through
    ).json()
    hitting_totals = api.league_hitting_totals(
        archiver, season=season, through=through
    ).json()

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

    # Each leaderboard is fetched once, then used twice: indexed by player for
    # the boxes, and reduced to a league benchmark for the deltas beside them.
    batted_ball_rows = sv.parse_csv(sv.batted_ball(archiver, year=season, minimum=10))
    bat_tracking_rows = sv.parse_csv(sv.bat_tracking(archiver, year=season, minimum=10))
    sprint_rows = sv.parse_csv(sv.sprint_speed(archiver, year=season, minimum=1))
    fielding_rows = sv.parse_csv(sv.outs_above_average(archiver, year=season, minimum=1))

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
        "batted_ball": sv.index_by_player(batted_ball_rows),
        "bat_tracking": sv.index_by_player(bat_tracking_rows),
        "fielding": sv.index_by_player(fielding_rows),
        "running": sv.index_by_player(sprint_rows),
    }

    # Weighted by playing time: an unweighted mean would let a player with
    # twelve batted balls count as much as a regular with four hundred.
    savant_benchmarks = {
        "batted_ball": la.leaderboard_means(
            batted_ball_rows, BATTED_BALL_FIELDS, weight_field="bbe"
        ),
        "bat_tracking": la.leaderboard_means(
            bat_tracking_rows, BAT_TRACKING_FIELDS, weight_field="swings_competitive"
        ),
        "running": la.leaderboard_means(
            sprint_rows, RUNNING_FIELDS, weight_field="competitive_runs"
        ),
        "fielding": la.leaderboard_means(fielding_rows, FIELDING_FIELDS),
    }

    def _population(rows, field_name: str) -> list[float]:
        values = []
        for row in rows:
            raw = row.get(field_name)
            try:
                if raw not in (None, ""):
                    values.append(float(raw))
            except (TypeError, ValueError):
                continue
        return values

    savant_populations = {
        "sprint_speed": _population(sprint_rows, "sprint_speed"),
        "outs_above_average": _population(fielding_rows, "outs_above_average"),
        "fielding_runs_prevented": _population(
            fielding_rows, "fielding_runs_prevented"),
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
                        api.STAT_CAREER, api.STAT_HOT_COLD_ZONES],
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
                        api.STAT_CAREER, api.STAT_HOT_COLD_ZONES],
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
        order_map, lineup_source, lineup_note = _lineup_order(game, side)
        for box in batters:
            box.batting_order = order_map.get(box.player_id)

        # Lineup order first when it exists, then everyone else by playing
        # time. Bench players are pushed down, never removed -- they are the
        # pinch-hit and defensive-replacement options a manager needs to see.
        batters.sort(
            key=lambda b: (
                b.batting_order if b.batting_order is not None else 99,
                -(b.season.get("plateAppearances") or 0),
            )
        )

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
            series_record=tc.parse_series_records(
                # Completed series only: today's game is still being played,
                # and fetching it would make the archived payload differ
                # between two runs that produce identical figures.
                api.season_schedule(
                    archiver, team_id=tid, season=season, through=through
                ).json(),
                team_id=tid,
            ),
        )

        sections[side] = TeamSection(
            team_id=tid, name=team_info.get("name", ""),
            abbreviation=team_info.get("abbreviation", ""),
            record=f"{record.get('wins', 0)}-{record.get('losses', 0)}",
            is_home=(side == "home"), batters=batters, pitchers=pitchers,
            opposing_hand=hands[opposite], profile=profile,
            lineup_source=lineup_source, lineup_note=lineup_note,
        )

    # Pitch-level Statcast, per player. This is the slow part of a run -- one
    # request each -- so it happens last, after everything cheap has succeeded.
    if include_statcast:
        for section in sections.values():
            for box in section.batters:
                box.statcast_zones, box.spray = _statcast_for(
                    archiver, player_id=box.player_id, season=season,
                    perspective="batter", bats=box.hand, as_of=on,
                )
            for box in section.pitchers:
                box.statcast_zones, _ = _statcast_for(
                    archiver, player_id=box.player_id, season=season,
                    perspective="pitcher", bats=None, as_of=on,
                )

    # -- current series ------------------------------------------------------
    # Box scores for the games already played in this set, plus each player's
    # line across them. This is the context a broadcaster reaches for first and
    # no season split can supply: what has happened in *these* games.
    series_boxes, series_payloads = _current_series_boxes(
        archiver, team_id=team_id, game=game, on=on
    )
    if series_boxes:
        series_batting, series_pitching = sr.player_series_lines(series_payloads)
        for section in sections.values():
            for box in section.batters:
                box.series_line = series_batting.get(box.player_id)
            for box in section.pitchers:
                box.series_line = series_pitching.get(box.player_id)

    # -- projection ----------------------------------------------------------
    # Loaded from a stored fit rather than trained here: a report must not
    # re-estimate a model, or two reports of the same game would disagree.
    projection = None
    freshness_note = None
    try:
        from guards_report.projections import freshness as projection_freshness
        from guards_report.projections import model as projection_model
        from guards_report.projections import predict as projection_predict
        from guards_report.projections import tonight as projection_tonight
        from guards_report.projections import train_props as projection_props

        root = settings.raw_archive_dir.parent
        # One explicit refresh before anything reads from disk. Each corpus
        # skips a season it already has cached, which is right for a finished
        # season and wrong for the one in progress -- without this the first
        # report of the year writes a snapshot and every later report projects
        # from it while looking perfectly current.
        freshness_note = projection_freshness.refresh_all(root, on=on, verbose=False)
        for warning in freshness_note.warnings:
            print(f"  warning: {warning}", file=sys.stderr)

        fitted = projection_model.load(
            settings.raw_archive_dir.parent / "models" / "game_outcome.json"
        )
        if fitted is not None:
            # The artifact's coefficients are fixed at fit time, but its rating
            # state ages every day the league keeps playing. Walk it forward to
            # the game being scouted so the clubs are rated on this season's
            # form, not on last year's team.
            from guards_report.projections import refresh as projection_refresh

            fitted, ratings_note = projection_refresh.refresh(
                fitted,
                on=on,
                cache_dir=settings.raw_archive_dir.parent / "corpus",
            )

            def _rpg(section) -> float | None:
                profile = getattr(section, "profile", None)
                for stat in (getattr(profile, "hitting", None) or []):
                    if getattr(stat, "key", "") == "runsPerGame":
                        return getattr(stat, "value", None)
                return None

            home_section, away_section = sections["home"], sections["away"]
            projection = projection_predict.project(
                fitted,
                home_team_id=home_section.team_id,
                away_team_id=away_section.team_id,
                home_team=home_section.abbreviation,
                away_team=away_section.abbreviation,
                venue_id=(game.get("venue") or {}).get("id"),
                home_starter=next(
                    (p for p in home_section.pitchers if p.is_probable_starter), None
                ),
                away_starter=next(
                    (p for p in away_section.pitchers if p.is_probable_starter), None
                ),
                home_offense_rpg=_rpg(home_section),
                away_offense_rpg=_rpg(away_section),
                plate_appearances=_plate_appearances(settings, on),
                home_lineup=_posted_lineup(home_section),
                away_lineup=_posted_lineup(away_section),
                lineup_source=home_section.lineup_source,
                on=on,
            )
            projection.ratings_note = ratings_note
            projection.freshness = {
                "corpus_through": str(freshness_note.corpus_through),
                "pitchers_through": str(freshness_note.pitchers_through),
                "pitches_through": str(freshness_note.pitches_through),
                "derived_through": str(freshness_note.derived_through),
                "fitted_ages": dict(freshness_note.fitted_ages),
                "requests": freshness_note.requests,
                "seconds": round(freshness_note.seconds, 1),
                "current": freshness_note.is_current(on),
            }

            # Props and first-five are separate artifacts with separate failure
            # modes: losing one should quieten a section, not the page.
            try:
                plate = _plate_appearances(settings, on)
                props_art = projection_props.load_props(root / "models" / "props.json")
                if props_art is not None and plate is not None:
                    stands = (
                        plate.drop_duplicates("batter")
                        .set_index("batter")["stand"].to_dict()
                    )
                    names = {
                        b.player_id: b.name
                        for side in (home_section, away_section)
                        for b in side.batters
                    }
                    for side, other, key in (
                        (home_section, away_section, "home"),
                        (away_section, home_section, "away"),
                    ):
                        card = _posted_lineup(side)
                        opposing = next(
                            (p for p in other.pitchers if p.is_probable_starter), None
                        )
                        projection.player_props[key] = projection_tonight.batter_props(
                            props_art, plate, on=on, lineup=card, names=names,
                            opposing_starter=getattr(opposing, "player_id", None),
                            opposing_throws=getattr(opposing, "hand", "R") or "R",
                            stands=stands, home_team=home_section.abbreviation,
                        )
                        own_starter = next(
                            (p for p in side.pitchers if p.is_probable_starter), None
                        )
                        if own_starter is not None:
                            projection.strikeouts[key] = (
                                projection_tonight.starter_strikeouts(
                                    props_art, plate, on=on,
                                    pitcher_id=own_starter.player_id,
                                    name=own_starter.name,
                                    opposing_lineup=_posted_lineup(other),
                                    stands=stands,
                                    throws=getattr(own_starter, "hand", "R") or "R",
                                )
                            )
            except Exception as exc:  # noqa: BLE001
                print(f"  warning: player props skipped ({exc})", file=sys.stderr)

            # First five innings, from the same inputs Model B uses. The
            # starter faces 92% of the batters who come up in five innings, so
            # his columns carry most of the weight here.
            try:
                f5_art = projection_props.load_first5(root / "models" / "first5.json")
                if f5_art is not None:
                    f5_features = {"league_rpg": projection.league_rpg}
                    for prefix, batting, fielding in (
                        ("home_", projection.home, projection.away),
                        ("away_", projection.away, projection.home),
                    ):
                        f5_features.update({
                            f"{prefix}off_rpg": batting.offense_rpg,
                            f"{prefix}opp_off_rpg": fielding.offense_rpg,
                            f"{prefix}opp_sp_fip": fielding.starter_fip,
                            f"{prefix}opp_sp_k": fielding.starter_k_pct,
                            f"{prefix}opp_sp_bb": fielding.starter_bb_pct,
                            f"{prefix}opp_sp_ip": fielding.starter_ip_per_start,
                            f"{prefix}park_factor": projection.park_factor,
                            f"{prefix}is_home": 1.0 if prefix == "home_" else 0.0,
                            f"{prefix}elo_diff": batting.elo - fielding.elo,
                            f"{prefix}od_exp_runs": (
                                projection.league_rpg
                                + batting.offense_rating + fielding.defense_rating
                                + (0.16 if prefix == "home_" else 0.0)
                            ),
                            f"{prefix}sp_known": (
                                1.0 if fielding.starter_fip is not None else 0.0
                            ),
                            f"{prefix}opp_sp_talent": fielding.starter_talent,
                            f"{prefix}own_lineup": batting.lineup_value,
                            f"{prefix}f5_line_known": 0.0,
                        })
                        # The opposing starter's own first-five record, which is
                        # the block that improved 9 of 9 held-out seasons.
                        opposing_box = next(
                            (
                                b for b in (
                                    away_section if prefix == "home_" else home_section
                                ).pitchers if b.is_probable_starter
                            ),
                            None,
                        )
                        line = projection_tonight.starter_first_five(
                            _f5_history(settings),
                            getattr(opposing_box, "player_id", None),
                            on=on,
                        )
                        f5_features.update(
                            {f"{prefix}{k}": v for k, v in line.items()}
                        )
                    projection.first_five = projection_tonight.first_five(
                        f5_art, f5_features
                    )
                    if projection.first_five is not None:
                        # Give it the full-game total so the two models can be
                        # checked against each other rather than only against
                        # arithmetic.
                        projection.first_five.full_game_total = (
                            projection.score.get("expected_total")
                        )
            except Exception as exc:  # noqa: BLE001
                print(f"  warning: first-five skipped ({exc})", file=sys.stderr)
    except Exception as exc:
        # A projection is an addition to the report, never a precondition for
        # it. Say what failed and carry on.
        print(f"  warning: projection skipped ({type(exc).__name__}: {exc})",
              file=sys.stderr)

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
        savant_benchmarks=savant_benchmarks,
        savant_populations=savant_populations,
        series_boxes=series_boxes,
        projection=projection,
        series=tc.parse_series(
            api.head_to_head(
                archiver, team_id=sections["home"].team_id,
                opponent_id=sections["away"].team_id, season=season, through=on,
            ).json(),
            home_team_id=sections["home"].team_id,
            today_game_pk=game_pk,
        ),
    )
    bundle.highlights = hl.build(bundle)
    return bundle
