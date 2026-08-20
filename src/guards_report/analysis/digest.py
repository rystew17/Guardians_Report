"""Build the compact, numbers-only view of a subject that the model is given.

This module is the boundary between the deterministic report and the written
analysis. Three properties matter and are enforced here rather than trusted:

1. **The model never sees HTML.** It receives a small JSON object of figures
   that metrics/ already computed. There is nothing to re-derive and nothing
   to re-format.

2. **The model never sees a number the report does not show.** Every value in
   a digest also appears in the rendered page, so a reader can check any claim
   against the box it came from.

3. **Every value is recorded for verification.** `Digest.values` collects the
   formatted form of every figure, which analysis/verify.py uses to confirm the
   model did not invent a number. That check is only as good as this list, so
   values are registered as they are added rather than re-walked afterwards.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from guards_report.analysis.verify import NUMBER


def _rate(value: Any) -> str | None:
    """Format as the report does: .305 rather than 0.305."""
    if value is None:
        return None
    text = f"{float(value):.3f}"
    return text[1:] if text.startswith("0.") else text


def _two(value: Any) -> str | None:
    return None if value is None else f"{float(value):.2f}"


def _pct(value: Any, *, already_pct: bool = False) -> str | None:
    if value is None:
        return None
    number = float(value)
    return f"{number if already_pct else number * 100:.1f}%"


@dataclass
class Digest:
    """One subject's figures, plus the exact strings the report displays.

    `subject_id` is stable across days for the same player, so an analysis can
    be cached and looked up. `fingerprint` changes whenever any figure changes,
    which is what makes the cache safe: same fingerprint means the underlying
    numbers are identical, so a stored analysis is still accurate.
    """

    kind: str          # "matchup" | "pitcher" | "batter"
    subject_id: str
    label: str
    data: dict[str, Any] = field(default_factory=dict)
    values: set[str] = field(default_factory=set)

    def _register(self, value: Any) -> None:
        """Record every figure inside `value`, however deeply nested.

        The invariant this protects: anything the model can read must be
        registered, or verify.py will report it as invented. Two ways that was
        violated before:

        * Groups nest. `form` is {"L5": {"ERA": "6.65", ...}}, so registering
          only the top level left every window figure unregistered while the
          model was shown all of them.
        * Figures are combined into phrases. An arsenal line reads
          "19.8% usage, 28.6% whiff, .268 xwOBA", but prose cites "28.6%" on
          its own, so the phrase must be broken into its numbers as well.

        Registration therefore uses the same tokenizer the verifier uses to
        pull numbers out of prose, so the two cannot drift apart.
        """
        if value is None or value == "":
            return
        if isinstance(value, dict):
            for key, nested in value.items():
                # Keys carry figures too -- "L15" and "vs LHP" are both read by
                # the model and both quotable back in prose.
                self._register(key)
                self._register(nested)
            return
        if isinstance(value, (list, tuple, set)):
            for nested in value:
                self._register(nested)
            return
        if isinstance(value, bool):
            return  # "True" is not a figure anyone verifies
        if isinstance(value, (str, int, float)):
            text = str(value)
            self.values.add(text)
            for token in NUMBER.findall(text):
                self.values.add(token)

    def put(self, key: str, value: Any) -> None:
        """Record a figure and register its displayed form for verification."""
        if value is None or value == "":
            return
        self.data[key] = value
        self._register(value)

    def put_group(self, group: str, values: dict[str, Any]) -> None:
        cleaned = {k: v for k, v in values.items() if v is not None and v != ""}
        if not cleaned:
            return
        self.data[group] = cleaned
        self._register(cleaned)

    @property
    def fingerprint(self) -> str:
        """Hash of the figures. Identical figures reuse a stored analysis."""
        payload = json.dumps(self.data, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_json(self) -> str:
        return json.dumps(self.data, indent=1, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Matchup
# ---------------------------------------------------------------------------


def matchup_digest(bundle: Any) -> Digest:
    d = Digest(kind="matchup", subject_id=f"game-{bundle.game_pk}",
               label=f"{bundle.away.abbreviation} at {bundle.home.abbreviation}")

    d.put("game_date", bundle.game_date.isoformat())
    d.put("venue", bundle.venue_name)
    if bundle.weather.get("condition"):
        d.put_group("weather", {
            "condition": bundle.weather.get("condition"),
            "temp_f": bundle.weather.get("temp"),
            "wind": bundle.weather.get("wind"),
        })

    for side, team in (("away", bundle.away), ("home", bundle.home)):
        profile = team.profile
        if not profile:
            continue
        record = profile.record
        block: dict[str, Any] = {"team": team.name}
        if record:
            block.update({
                "record": f"{record.wins}-{record.losses}",
                "division_rank": record.division_rank,
                "games_back": record.games_back,
                "streak": record.streak,
                "last_10": record.split("lastTen"),
                "home_record": record.split("home"),
                "away_record": record.split("away"),
                "one_run_record": record.split("oneRun"),
                "pythagorean": record.pythagorean,
                "luck_wins_vs_expected": record.luck,
            })
        block["run_differential"] = profile.run_differential
        block["bullpen_arms_with_2plus_days_rest"] = (
            f"{profile.bullpen_available} of {profile.bullpen_total}"
        )
        block["bullpen_pitches_last_3_days"] = profile.bullpen_pitches_last_3

        if profile.series_record and profile.series_record.completed:
            sr = profile.series_record
            block["series_record"] = sr.record
            block["series_completed"] = sr.completed
            block["sweeps_recorded"] = sr.sweeps_for
            block["times_swept"] = sr.sweeps_against

        for group, label in (("hitting", "offense_ranks"), ("pitching", "run_prevention_ranks")):
            ranks = {
                stat.label: f"{stat.rank} of 30"
                for stat in getattr(profile, group)
                if stat.rank is not None
            }
            if ranks:
                block[label] = ranks

        d.put_group(side, block)

    if bundle.series and bundle.series.games_played:
        d.put_group("season_series", {
            "meetings": bundle.series.games_played,
            "home_leads": bundle.series.record_for(home=True),
            "game_number_in_current_set": bundle.series.series_game_number,
            "games_in_current_set": bundle.series.games_in_series,
        })

    for team, side in ((bundle.away, "away"), (bundle.home, "home")):
        starter = next((p for p in team.pitchers if p.is_probable_starter), None)
        if starter:
            d.put_group(f"{side}_probable_starter", {
                "name": starter.name,
                "throws": starter.hand,
                "era": _two(starter.season.get("era")),
                "fip": _two(starter.season.get("fip")),
                "whip": _two(starter.season.get("whip")),
                "innings": starter.season.get("inningsPitchedDisplay"),
                "k_pct": _pct(starter.season.get("kPct")),
                "bb_pct": _pct(starter.season.get("bbPct")),
            })

    d.put_group("league_context", {
        "league_avg": _rate(bundle.league_hitting.get("avg")),
        "league_obp": _rate(bundle.league_hitting.get("obp")),
        "league_slg": _rate(bundle.league_hitting.get("slg")),
        "league_era": _two(bundle.league_pitching.get("era")),
    })

    d.put_group("precomputed_highlights", {
        f"{i + 1}. {h.headline}": h.detail for i, h in enumerate(bundle.highlights)
    })
    return d


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------


def _windows(box: Any, keys: tuple[tuple[str, str, str], ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for window in box.windows:
        line: dict[str, Any] = {"games": window.stats.get("games")}
        for key, label, fmt in keys:
            value = window.stats.get(key)
            line[label] = (
                _rate(value) if fmt == "rate"
                else _two(value) if fmt == "two"
                else _pct(value)
            )
        out[window.label] = {k: v for k, v in line.items() if v is not None}
    return out


PITCHER_KEYS = (
    ("era", "ERA", "two"), ("fip", "FIP", "two"), ("whip", "WHIP", "two"),
    ("kPct", "K%", "pct"), ("bbPct", "BB%", "pct"), ("hrPer9", "HR/9", "two"),
)
BATTER_KEYS = (
    ("avg", "AVG", "rate"), ("obp", "OBP", "rate"), ("slg", "SLG", "rate"),
    ("ops", "OPS", "rate"), ("iso", "ISO", "rate"),
    ("kPct", "K%", "pct"), ("bbPct", "BB%", "pct"),
)

SPLIT_LABELS = {
    "vl": "vs LHP", "vr": "vs RHP", "risp": "runners in scoring position",
    "risp2": "RISP with two out", "lc": "late and close", "2s": "with two strikes",
    "ac": "ahead in count", "bc": "behind in count",
    "pi000": "first 75 pitches", "pi760": "pitch 76 onward",
}


def _situational(box: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for code, label in SPLIT_LABELS.items():
        stat = box.situational.get(code)
        if not stat:
            continue
        pa = stat.get("plateAppearances") or stat.get("battersFaced")
        if not pa:
            continue
        entry = {"sample": f"{pa} PA", "OPS": stat.get("ops"), "AVG": stat.get("avg")}
        if stat.get("era"):
            entry["ERA"] = stat.get("era")
        out[label] = {k: v for k, v in entry.items() if v}
    return out


def _percentiles(box: Any, keys: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    return {
        label: f"{box.percentiles[key]}th percentile"
        for key, label in keys
        if box.percentiles.get(key) not in (None, "")
    }


def pitcher_digest(
    box: Any, bundle: Any, team_label: str, *, kind: str = "pitcher"
) -> Digest:
    # `kind` selects the task prompt; the subject id stays keyed on the player
    # so a cached note follows him if his role changes between runs.
    d = Digest(kind=kind, subject_id=f"pitcher-{box.player_id}",
               label=f"{box.name} ({team_label})")

    d.put("name", box.name)
    d.put("throws", box.hand)
    d.put("role", box.role)
    d.put("is_todays_probable_starter", box.is_probable_starter)

    d.put_group("form", _windows(box, PITCHER_KEYS))
    d.put_group("statcast_percentiles", _percentiles(box, (
        ("xera", "xERA"), ("xwoba", "xwOBA"), ("k_percent", "strikeout rate"),
        ("bb_percent", "walk rate"), ("whiff_percent", "whiff rate"),
        ("chase_percent", "chase rate"), ("brl_percent", "barrel rate allowed"),
        ("hard_hit_percent", "hard-hit rate allowed"), ("fb_velocity", "fastball velocity"),
    )))

    if box.arsenal:
        d.put_group("arsenal", {
            p["pitch"]: (
                f"{_pct(p['usage'], already_pct=True)} usage, "
                f"{_pct(p['whiff'], already_pct=True)} whiff, "
                f"{_rate(p['xwoba'])} xwOBA"
            )
            for p in box.arsenal if p.get("pitch") and p.get("usage")
        })

    d.put_group("situational", _situational(box))

    if box.availability:
        a = box.availability
        d.put_group("workload", {
            "days_rest": a.days_rest,
            "pitches_last_3_days": a.pitches_last_3,
            "appearances_last_7_days": a.appearances_last_7,
            "pitched_back_to_back": a.back_to_back,
        })

    d.put_group("league_average_for_comparison", {
        "ERA": _two(bundle.league_pitching.get("era")),
        "FIP": _two(bundle.league_pitching.get("fip")),
        "WHIP": _two(bundle.league_pitching.get("whip")),
        "K%": _pct(bundle.league_pitching.get("kPct")),
        "BB%": _pct(bundle.league_pitching.get("bbPct")),
    })
    return d


def batter_digest(
    box: Any, bundle: Any, team_label: str, opposing_hand: str | None,
    *, kind: str = "batter",
) -> Digest:
    d = Digest(kind=kind, subject_id=f"batter-{box.player_id}",
               label=f"{box.name} ({team_label})")

    d.put("name", box.name)
    d.put("bats", box.hand)
    d.put("position", box.position)
    d.put("batting_order", box.batting_order)
    d.put("in_todays_lineup", box.batting_order is not None)
    if opposing_hand:
        d.put("faces_starter_throwing", opposing_hand)

    d.put_group("form", _windows(box, BATTER_KEYS))
    if box.sabermetrics.get("wRcPlus") is not None:
        d.put("wRC_plus", round(float(box.sabermetrics["wRcPlus"])))

    d.put_group("statcast_percentiles", _percentiles(box, (
        ("xwoba", "xwOBA"), ("xba", "xBA"), ("xslg", "xSLG"),
        ("brl_percent", "barrel rate"), ("exit_velocity", "exit velocity"),
        ("hard_hit_percent", "hard-hit rate"), ("k_percent", "strikeout rate"),
        ("bb_percent", "walk rate"), ("whiff_percent", "whiff rate"),
        ("chase_percent", "chase rate"), ("sprint_speed", "sprint speed"),
    )))

    if box.batted_ball.get("bbe"):
        d.put_group("batted_ball_profile", {
            "batted_balls": int(box.batted_ball["bbe"]),
            "pull": _pct(box.batted_ball.get("pull_rate"), already_pct=True),
            "straightaway": _pct(box.batted_ball.get("straight_rate"), already_pct=True),
            "opposite_field": _pct(box.batted_ball.get("oppo_rate"), already_pct=True),
            "ground_ball": _pct(box.batted_ball.get("gb_rate"), already_pct=True),
            "line_drive": _pct(box.batted_ball.get("ld_rate"), already_pct=True),
            "fly_ball": _pct(box.batted_ball.get("fb_rate"), already_pct=True),
        })

    if box.arsenal:
        # Weakest and strongest pitch types by xwOBA, which is the part of the
        # arsenal table a reader would actually act on.
        rated = [p for p in box.arsenal if p.get("xwoba") and p.get("pa")]
        rated.sort(key=lambda p: p["xwoba"])
        if rated:
            d.put_group("versus_pitch_type", {
                f"weakest: {rated[0]['pitch']}": f"{_rate(rated[0]['xwoba'])} xwOBA",
                f"strongest: {rated[-1]['pitch']}": f"{_rate(rated[-1]['xwoba'])} xwOBA",
            })

    d.put_group("situational", _situational(box))

    if box.fielding.get("outs_above_average") is not None:
        d.put_group("defense", {
            "outs_above_average": int(box.fielding["outs_above_average"]),
        })

    d.put_group("league_average_for_comparison", {
        "AVG": _rate(bundle.league_hitting.get("avg")),
        "OBP": _rate(bundle.league_hitting.get("obp")),
        "SLG": _rate(bundle.league_hitting.get("slg")),
        "OPS": _rate(bundle.league_hitting.get("ops")),
        "K%": _pct(bundle.league_hitting.get("kPct")),
        "BB%": _pct(bundle.league_hitting.get("bbPct")),
    })
    return d


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_subjects(bundle: Any) -> list[Digest]:
    """The subjects that get a written read-out.

    Everyone who could plausibly appear: the matchup, both probable starters,
    the posted lineups, the relievers likely to be available, and the bench
    bats behind them. A preview is used to prepare for what might happen, and
    the seventh-inning arm and the platoon pinch-hitter are exactly the people
    a reader has the least feel for.

    Pitchers already ruled out by workload are skipped -- there is nothing to
    say about a man who cannot pitch tonight, and it would spend credit to say
    it.
    """
    digests = [matchup_digest(bundle)]

    for team in (bundle.away, bundle.home):
        label = team.abbreviation
        starter = next((p for p in team.pitchers if p.is_probable_starter), None)
        if starter:
            digests.append(pitcher_digest(starter, bundle, label))

        for box in team.pitchers:
            if box.is_probable_starter or not _likely_available(box):
                continue
            digests.append(pitcher_digest(box, bundle, label, kind="reliever"))

        for box in team.batters:
            kind = "batter" if box.batting_order is not None else "bench"
            digests.append(
                batter_digest(box, bundle, label, team.opposing_hand, kind=kind)
            )

    return digests


def _likely_available(box: Any) -> bool:
    """Whether a reliever could reasonably be used tonight.

    Back-to-back appearances with a heavy recent pitch count are how a bullpen
    arm becomes unavailable. This mirrors the availability already shown in his
    box; it decides only whether writing about him is worth the credit.
    """
    availability = getattr(box, "availability", None)
    if availability is None:
        return True
    if getattr(availability, "back_to_back", False):
        return (getattr(availability, "pitches_last_3", 0) or 0) < 40
    return True
