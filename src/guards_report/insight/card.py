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
from guards_report.insight import evaluators, matchup, render, select

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
    subjects: dict[int, str] = field(default_factory=dict)
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

    # -- the game itself -----------------------------------------------------
    try:
        game_findings = (
            matchup.from_projection(projection, home, away)
            + matchup.starter_strikeout_edge(projection, home, away)
            + matchup.key_player(projection, home, away)
        )
        result.matchup = matchup.paragraph(game_findings)
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"matchup: {type(exc).__name__}: {exc}")

    # -- everyone in it ------------------------------------------------------
    for section in (bundle.home, bundle.away):
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

    result.seconds = time.time() - started
    return result
