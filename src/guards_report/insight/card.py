"""Analysis for a whole card, computed rather than generated.

The entry point that replaces `analysis/agent.py`. Same shape of output -- one
short piece of prose per subject -- produced by evaluators and templates instead
of a language model, so the same game yields the same words forever and every
figure traces to a computation.

Structured to lose a section rather than the page. An evaluator that raises
takes its own subject down and nothing else, because a report missing one
player's line is still a report and a report that failed to build is not.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from guards_report.insight import batters as batter_evaluators
from guards_report.insight import dossier as dossier_module
from guards_report.insight import evaluators, matchup, profile as profile_module
from guards_report.insight import render, select
from guards_report.insight import voice as voice_module

# How many criteria each subject scans, which sets its significance floor. These
# must track the evaluators actually called below: raising the count without
# raising the bar is how a finding set quietly fills with noise.
PITCHER_CRITERIA = 8
BATTER_CRITERIA = 12

# Columns the evaluators read. Projected at load time because the corpus is 121
# columns wide and no evaluator wants them all.
COLUMNS = [
    "batter", "pitcher", "stand", "p_throws", "pitch_name", "description",
    "events", "zone", "bb_type", "launch_speed", "launch_speed_angle",
    "hc_x", "hc_y", "woba_value", "estimated_woba_using_speedangle",
]


@dataclass
class CardAnalysis:
    """One line per subject, plus what it cost to produce."""

    matchup: str = ""
    matchup_parts: list = field(default_factory=list)
    subjects: dict[int, str] = field(default_factory=dict)
    dossiers: dict[int, Any] = field(default_factory=dict)
    findings: dict[int, int] = field(default_factory=dict)
    seconds: float = 0.0
    covered: int = 0
    attempted: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return self.covered / self.attempted if self.attempted else 0.0


def load_corpus(pitch_dir: Path, season: int) -> pd.DataFrame:
    frames = [
        pd.read_parquet(path, columns=COLUMNS)
        for path in sorted(Path(pitch_dir).glob(f"{season}_*.parquet"))
    ]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True)


def load_references(models_dir: Path, season: int) -> dict:
    """This season's graded populations, or empty frames if not yet built."""
    out = {}
    for kind, name in (("batter", "batter_profiles"), ("pitcher", "pitcher_profiles")):
        path = Path(models_dir) / f"{name}.parquet"
        if not path.exists():
            out[kind] = pd.DataFrame()
            continue
        frame = pd.read_parquet(path)
        out[kind] = frame[frame["season"] == season].reset_index(drop=True)
    return out


def analyse_pitcher(pitcher_id: int, surname: str, *, pitch_corpus, corpus) -> tuple[str, int]:
    found = evaluators.arsenal(
        pitcher_id=pitcher_id, pitches=pitch_corpus, league=pitch_corpus
    )
    seen = int((corpus["pitcher"] == pitcher_id).sum())
    found += evaluators.no_track_record(player_id=pitcher_id, evidence=seen, minimum=150)
    # The extremes are renamed so the sentence says which end it describes.
    found = [
        replace(f, code="pit.arsenal.best" if f.z > 0 else "pit.arsenal.worst")
        if f.code == "pit.arsenal.pitch" else f
        for f in found
    ]
    chosen = select.select(found, criteria=PITCHER_CRITERIA, limit=2)
    return render.sentence(chosen, subject=surname), len(found)


def analyse_batter(
    batter_id: int, surname: str, *, corpus, pitch_corpus, profile
) -> tuple[str, int]:
    empty = pd.Series(dtype=float)
    found = []
    found += batter_evaluators.batted_ball_profile(
        batter_id=batter_id, plate=corpus, league_profile=profile)
    found += batter_evaluators.spray_tendency(
        batter_id=batter_id, plate=corpus, league_pull=profile.get("pull", empty))
    found += batter_evaluators.plate_discipline(
        batter_id=batter_id, plate=corpus,
        league_chase=profile["chase"].dropna(), league_whiff=profile["whiff"].dropna())
    found += batter_evaluators.pitch_type_weakness(
        batter_id=batter_id, plate=pitch_corpus, league=pitch_corpus)
    found += batter_evaluators.platoon_split(
        batter_id=batter_id, plate=corpus,
        league_gap=profile.get("platoon_gap", empty).dropna())
    found += batter_evaluators.results_versus_contact(
        batter_id=batter_id, plate=corpus, league_gap=profile["luck_gap"].dropna())
    chosen = select.select(found, criteria=BATTER_CRITERIA, limit=2)
    return render.sentence(chosen, subject=surname), len(found)


def _surname(name: str) -> str:
    return name.split(" (")[0].split()[-1] if name else ""


def analyse(bundle: Any, *, pitch_dir: Path, on: date | None = None) -> CardAnalysis:
    """Every subject on the card, computed.

    Mirrors what `analysis/digest.select_subjects` chooses, so the two layers can
    be compared line for line and swapped without the page changing shape.
    """
    started = time.time()
    result = CardAnalysis()
    season = (on or date.today()).year

    corpus = load_corpus(pitch_dir, season)
    if corpus.empty:
        result.warnings.append(f"no pitch data for {season}")
        result.seconds = time.time() - started
        return result

    pitch_corpus = corpus[corpus["pitch_name"].notna() & (corpus["pitch_name"] != "")]
    try:
        profile = batter_evaluators.league_profile(corpus)
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"league profile: {type(exc).__name__}: {exc}")
        profile = pd.DataFrame()

    home, away = bundle.home.abbreviation, bundle.away.abbreviation
    projection = getattr(bundle, "projection", None)

    # Graded populations for the profiles, plus the leaderboards that carry the
    # two tools the pitch corpus cannot see.
    references = load_references(Path(pitch_dir).parent / "models", season)
    populations = getattr(bundle, "savant_populations", {}) or {}
    starters = {
        "home": next((p for p in getattr(bundle.home, "pitchers", [])
                      if getattr(p, "is_probable_starter", False)), None),
        "away": next((p for p in getattr(bundle.away, "pitchers", [])
                      if getattr(p, "is_probable_starter", False)), None),
    }

    def _row_for(frame, column: str, player_id: int) -> dict | None:
        if frame is None or frame.empty or column not in frame.columns:
            return None
        hit = frame[frame[column] == int(player_id)]
        return hit.iloc[0].to_dict() if len(hit) else None

    def _profile_for(box, kind: str):
        column = "batter" if kind == "batter" else "pitcher"
        row = _row_for(references.get(kind), column, int(box.player_id))
        if row is None:
            return None
        if kind == "batter":
            return profile_module.build_batter(
                box, row, references["batter"], populations)
        return profile_module.build_pitcher(box, row, references["pitcher"])

    # Each side's opponent starter, profiled once, so a matchup is one profile
    # against another rather than a head-to-head line of nine at-bats.
    # One voice for the whole card, so no two players in a row are
    # described with the same adjective.
    card_voice = voice_module.Voice()

    opposing = {}
    for side, other in (("home", "away"), ("away", "home")):
        box = starters[other]
        opposing[side] = (box, _profile_for(box, "pitcher") if box else None)

    # And the reverse. A batter faces one pitcher; a pitcher faces nine, so the
    # lineup he will see is aggregated into a single opponent with the same
    # tool grades a hitter carries. Graded against the hand he throws: a
    # lineup's profile moves by handedness, and nine hitters is enough sample to
    # make the split usable where one hitter is not.
    lineups: dict[str, Any] = {}
    for side, other in (("home", "away"), ("away", "home")):
        section = bundle.away if other == "away" else bundle.home
        try:
            lineups[side] = profile_module.lineup_profile(
                list(getattr(section, "batters", [])),
                references.get("batter"), populations,
                hand=(getattr(starters[side], "hand", "") or ""),
            )
        except Exception as exc:  # noqa: BLE001
            result.warnings.append(f"lineup profile: {type(exc).__name__}: {exc}")
            lineups[side] = None

    # The bat the projection likes most tonight, so the note can name the man
    # rather than leave the reader to go and find him.
    key_bat = None
    if projection is not None:
        try:
            found = matchup.key_player(projection, home, away)
            if found:
                detail = found[0].detail
                key_bat = {
                    "name": detail.get("name"), "homer": detail.get("homer"),
                    "slot": detail.get("slot"), "team": detail.get("team"),
                }
        except Exception:  # noqa: BLE001 -- naming a hitter is not essential
            key_bat = None

    # -- the game itself -----------------------------------------------------
    try:
        # Three parts in reading order rather than one paragraph: who these two
        # clubs are, who is pitching, then what the models make of it. A win
        # probability lands differently once the first two are established.
        built = matchup.note(
            bundle,
            home_starter=(opposing["away"][1] if opposing.get("away") else None),
            away_starter=(opposing["home"][1] if opposing.get("home") else None),
            # `lineups[side]` is what that side's *pitcher* faces, which is the
            # other club's batters. Passing it as "the home lineup" is what
            # graded each starter against his own team.
            home_faces=lineups.get("home"), away_faces=lineups.get("away"),
            voice=card_voice,
        )
        result.matchup_parts = list(built.parts)
        result.matchup = " ".join(text for _, text in built.parts)
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"matchup: {type(exc).__name__}: {exc}")

    # -- everyone in it ------------------------------------------------------
    for side, section in (("home", bundle.home), ("away", bundle.away)):
        opponent_box, opponent_profile = opposing.get(side, (None, None))

        for box in list(getattr(section, "pitchers", [])):
            result.attempted += 1
            try:
                text, n = analyse_pitcher(
                    int(box.player_id), _surname(box.name),
                    pitch_corpus=pitch_corpus, corpus=corpus,
                )
            except Exception as exc:  # noqa: BLE001
                result.warnings.append(f"{box.name}: {type(exc).__name__}: {exc}")
                continue
            result.findings[int(box.player_id)] = n
            if text:
                result.subjects[int(box.player_id)] = text
                result.covered += 1
            try:
                player = _profile_for(box, "pitcher")
                if player is not None:
                    # The key bat belongs to the *other* club, so it is only
                    # offered to the pitcher who has to face him.
                    theirs = None
                    if key_bat and key_bat.get("team") not in (
                        None, section.abbreviation
                    ):
                        theirs = key_bat
                    result.dossiers[int(box.player_id)] = dossier_module.build(
                        box, player, surname=_surname(box.name),
                        opposing_lineup=lineups.get(side),
                        key_bat=theirs, voice=card_voice,
                        starter=bool(getattr(box, "is_probable_starter", False)))
            except Exception as exc:  # noqa: BLE001
                result.warnings.append(
                    f"{box.name} profile: {type(exc).__name__}: {exc}")

        for box in list(getattr(section, "batters", [])):
            result.attempted += 1
            if profile.empty:
                continue
            try:
                text, n = analyse_batter(
                    int(box.player_id), _surname(box.name),
                    corpus=corpus, pitch_corpus=pitch_corpus, profile=profile,
                )
            except Exception as exc:  # noqa: BLE001
                result.warnings.append(f"{box.name}: {type(exc).__name__}: {exc}")
                continue
            result.findings[int(box.player_id)] = n
            if text:
                result.subjects[int(box.player_id)] = text
                result.covered += 1
            try:
                player = _profile_for(box, "batter")
                if player is not None:
                    result.dossiers[int(box.player_id)] = dossier_module.build(
                        box, player, surname=_surname(box.name),
                        opposing_starter=opponent_box,
                        opposing_profile=opponent_profile,
                        voice=card_voice)
            except Exception as exc:  # noqa: BLE001
                result.warnings.append(
                    f"{box.name} profile: {type(exc).__name__}: {exc}")

    result.seconds = time.time() - started
    return result
