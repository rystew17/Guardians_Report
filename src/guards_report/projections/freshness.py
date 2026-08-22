"""Bring every corpus current before a report is built.

The projection layer reads four caches, and until this module existed three of
them silently froze. `corpus.build`, `pitchers.build` and `pitches.build` all
skip a season whose file is already on disk, which is exactly right for a
finished season and exactly wrong for the one in progress: the first report of
the year writes a snapshot, and every report after it trains and projects on
that same snapshot while appearing perfectly current.

That failure has now appeared three times in this project in different clothes --
Elo ratings frozen at the previous September, a talent model fitted on completed
seasons and never carried forward, and a pull loop whose range stopped short of
the current year. It never raises. The report renders, the numbers look
plausible, and they describe a league that has moved on.

So refreshing is one explicit step with one report of what it did, rather than a
property each caller has to remember. Everything downstream reads from disk and
can assume the disk is current, and the page states the date each corpus reaches
rather than implying it is live.

Cost is bounded and mostly the pitch corpus: one schedule request for the game
corpus, a handful for the pitcher logs, and thirty for the pitch corpus, which
fetches only the days since it was last written.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from guards_report.projections import corpus, pitchers, pitches


@dataclass
class Freshness:
    """What each corpus reaches, and what it cost to get there."""

    corpus_through: date | None = None
    pitchers_through: date | None = None
    pitches_through: date | None = None
    derived_through: date | None = None
    profiles_through: date | None = None
    fitted_ages: dict[str, int] = field(default_factory=dict)
    requests: int = 0
    seconds: float = 0.0
    refreshed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def stale_days(self, on: date) -> dict[str, int | None]:
        """Days between each corpus and the date being projected."""
        return {
            name: (on - value).days if value else None
            for name, value in (
                ("corpus", self.corpus_through),
                ("pitchers", self.pitchers_through),
                ("pitches", self.pitches_through),
                ("derived", self.derived_through),
                ("profiles", self.profiles_through),
            )
        }

    def is_current(self, on: date, *, tolerance: int = 1) -> bool:
        """Whether everything reaches the day before the game.

        One day of tolerance, because a corpus can only contain games that have
        finished, and the report is built before tonight's has been played.
        """
        return all(
            value is not None and value >= 0 and value <= tolerance
            for value in self.stale_days(on).values()
        )

    def line(self) -> str:
        parts = [
            f"corpus {self.corpus_through}",
            f"pitchers {self.pitchers_through}",
            f"pitches {self.pitches_through}",
            f"derived {self.derived_through}",
            f"profiles {self.profiles_through}",
        ]
        return (
            " | ".join(parts)
            + f"  ({self.requests} requests, {self.seconds:.0f}s)"
            + (f"  WARNINGS: {len(self.warnings)}" if self.warnings else "")
        )


# Fitted artifacts whose age the report should state. Their coefficients are
# deliberately fixed -- a report must never re-estimate a model, or two reports
# of the same game would disagree -- but "deliberately fixed" and "quietly
# months old" look identical from the page, and only one of them is fine.
FITTED_ARTIFACTS = ("game_outcome.json", "first5.json", "props.json")

# Past this the fit is stale enough to say so. Roughly a fortnight of baseball:
# long enough that rosters, rotations and form have all moved, short enough that
# it fires before a pennant race renders the ratings meaningless.
REFIT_AFTER_DAYS = 14


def fitted_ages(models_dir: Path, *, now: date | None = None) -> dict[str, int]:
    """Days since each fitted artifact was written.

    Read from the artifact's own `fitted_at` rather than the file's mtime, which
    a sync, a restore or a copy would reset without the model having changed.
    """
    import json

    today = now or date.today()
    ages: dict[str, int] = {}
    for name in FITTED_ARTIFACTS:
        path = Path(models_dir) / name
        if not path.exists():
            continue
        try:
            stamp = json.loads(path.read_text()).get("fitted_at")
            if stamp:
                fitted = datetime.fromisoformat(stamp).date()
                ages[name.removesuffix(".json")] = (today - fitted).days
        except Exception:  # noqa: BLE001 -- an unreadable stamp is not fatal
            continue
    return ages


def _latest(directory: Path, pattern: str, column: str = "game_date") -> date | None:
    files = sorted(Path(directory).glob(pattern))
    if not files:
        return None
    stamps = []
    for path in files:
        try:
            values = pd.read_parquet(path, columns=[column])[column]
        except Exception:  # noqa: BLE001 -- a missing column is not fatal here
            continue
        if len(values):
            stamps.append(pd.to_datetime(values).max())
    return max(stamps).date() if stamps else None


def _note_stale_fits(result: "Freshness") -> None:
    """Warn about fits that have aged out.

    Called on both paths deliberately. The short-circuit is taken precisely when
    every corpus is current, which is the ordinary case -- so a check that only
    ran on the refreshing path would stay silent about a model that had been
    sitting untouched for two months, on every single day that nothing else was
    wrong.
    """
    for name, age in result.fitted_ages.items():
        if age > REFIT_AFTER_DAYS:
            result.warnings.append(
                f"{name} was fitted {age} days ago; its coefficients predate "
                "roughly a fortnight of baseball and it should be refitted"
            )


def rebuild_derived(root: Path) -> int | None:
    """Recompute the first-five starter table from the pitch corpus.

    A full recompute rather than an append, because each row's value is a
    running mean over that starter's whole career and the stored table keeps the
    mean rather than the totals behind it -- so there is nothing to append onto.
    With the columns projected at read time this is around eight seconds against
    eight million pitches, which is cheaper than the bookkeeping an incremental
    version would need and cannot drift from a full rebuild.

    Returns the number of starts written, or None when there is no corpus yet.
    """
    from guards_report.projections import first5 as first5_module

    directory = Path(root) / "pitches"
    files = sorted(directory.glob("*.parquet"))
    if not files:
        return None

    columns = [
        "game_pk", "game_date", "season", "inning", "pitcher", "batting_team",
        "events", "post_bat_score", "bat_score",
    ]
    pitch = pd.concat(
        [pd.read_parquet(path, columns=columns) for path in files],
        ignore_index=True,
    )
    pitch["game_date"] = pd.to_datetime(pitch["game_date"])

    history = first5_module.starter_history(pitch)
    target = Path(root) / "models" / "f5_starter.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    history.to_parquet(target, index=False)
    return len(history)


def rebuild_profiles(root: Path) -> int | None:
    """Recompute the batter and pitcher reference populations.

    A tool grade is a percentile among this season's qualified players, so the
    population is as perishable as the corpus it comes from -- and staler in
    effect, because a frozen reference silently re-ranks the whole league the
    moment anyone's rate moves. Rebuilt beside the first-five table for exactly
    that reason.
    """
    from guards_report.insight import profile as profile_module

    directory = Path(root) / "pitches"
    files = sorted(directory.glob("*.parquet"))
    if not files:
        return None

    pitch = pd.concat(
        [pd.read_parquet(path,
                         columns=profile_module.PITCH_COLUMNS_FOR_PROFILES)
         for path in files],
        ignore_index=True,
    )
    models = Path(root) / "models"
    models.mkdir(parents=True, exist_ok=True)

    batters = profile_module.build_batter_reference(pitch)
    pitchers_frame = profile_module.build_pitcher_reference(pitch)
    batters.to_parquet(models / "batter_profiles.parquet", index=False)
    pitchers_frame.to_parquet(models / "pitcher_profiles.parquet", index=False)
    return len(batters) + len(pitchers_frame)


def survey(root: Path, season: int) -> Freshness:
    """What is on disk right now, without fetching anything."""
    return Freshness(
        corpus_through=_latest(root / "corpus", f"*{season}*.parquet"),
        pitchers_through=_latest(root / "pitchers", f"*{season}*.parquet"),
        pitches_through=_latest(root / "pitches", f"{season}_*.parquet"),
        derived_through=_latest(root / "models", "f5_starter.parquet"),
        profiles_through=_latest(
            root / "models", "batter_profiles.parquet",
            column="built_through"),
        fitted_ages=fitted_ages(root / "models"),
    )


def refresh_all(
    root: Path,
    *,
    on: date,
    verbose: bool = True,
    skip_pitches: bool = False,
) -> Freshness:
    """Refetch the in-progress season across every corpus.

    Ordered cheapest first, and each step is independently guarded: a failure in
    one corpus leaves the others refreshed and is reported rather than raised. A
    report built on slightly stale pitch data is worth having; one that fails to
    build because Savant was briefly unreachable is not.
    """
    season = on.year
    started = time.time()

    # Generating twice in a day would otherwise refetch thirty team-seasons to
    # discover nothing had changed. The survey is free and answers that.
    standing = survey(root, season)
    if standing.is_current(on):
        standing.seconds = time.time() - started
        standing.refreshed = ["already current"]
        _note_stale_fits(standing)
        if verbose:
            print("  every corpus already current; nothing fetched", flush=True)
        return standing

    result = Freshness()

    # -- game corpus: one schedule request ---------------------------------
    try:
        games = corpus.build(
            range(season, season + 1), cache_dir=root / "corpus", refresh=True
        )
        result.requests += 1
        result.refreshed.append("corpus")
        if verbose:
            print(f"  corpus   {len(games):,} games", flush=True)
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"corpus: {type(exc).__name__}: {exc}")

    # -- pitcher logs: batched by pitcher id -------------------------------
    try:
        full = corpus.build(
            range(corpus.FIRST_SEASON, season + 1), cache_dir=root / "corpus"
        ).query("game_type == 'R'")
        current = full[full["season"] == season].reset_index(drop=True)
        if len(current):
            logs = pitchers.build(
                current, cache_dir=root / "pitchers", refresh=True
            )
            result.requests += 3          # batched; a handful of calls
            result.refreshed.append("pitchers")
            if verbose:
                print(f"  pitchers {len(logs):,} starts", flush=True)
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"pitchers: {type(exc).__name__}: {exc}")

    # -- pitch corpus: incremental, one request per club --------------------
    if not skip_pitches:
        try:
            full = corpus.build(
                range(corpus.FIRST_SEASON, season + 1), cache_dir=root / "corpus"
            ).query("game_type == 'R'")
            teams = pitches.season_teams(full, season)
            added = pitches.refresh_current_season(
                season, teams, cache_dir=root / "pitches",
                through=on, verbose=False,
            )
            result.requests += len(teams)
            result.refreshed.append("pitches")
            if verbose:
                print(f"  pitches  +{sum(added.values()):,} new", flush=True)
        except Exception as exc:  # noqa: BLE001
            result.warnings.append(f"pitches: {type(exc).__name__}: {exc}")

    # -- derived tables: no fetch, but they freeze exactly like a corpus -----
    # `f5_starter` is read at serve time and is derived from the pitch corpus,
    # so refreshing the corpus without rebuilding this leaves the report reading
    # a starter's first-five line from whenever the model was last trained. That
    # is the same failure as the frozen ratings, one layer down, and it is
    # invisible because the table is present and parses cleanly.
    try:
        rebuilt = rebuild_derived(root)
        if rebuilt is not None:
            result.refreshed.append("derived")
            if verbose:
                print(f"  derived  {rebuilt:,} starts", flush=True)
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"derived: {type(exc).__name__}: {exc}")

    try:
        graded = rebuild_profiles(root)
        if graded is not None:
            result.refreshed.append("profiles")
            if verbose:
                print(f"  profiles {graded:,} player-seasons", flush=True)
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"profiles: {type(exc).__name__}: {exc}")

    survey_after = survey(root, season)
    result.corpus_through = survey_after.corpus_through
    result.pitchers_through = survey_after.pitchers_through
    result.pitches_through = survey_after.pitches_through
    result.derived_through = survey_after.derived_through
    result.fitted_ages = survey_after.fitted_ages
    result.seconds = time.time() - started

    # A corpus that reaches past the date being projected has leaked, and that
    # matters more than being behind: it would let a model see the game it is
    # predicting.
    _note_stale_fits(result)

    for name, days in result.stale_days(on).items():
        if days is not None and days < 0:
            result.warnings.append(
                f"{name} contains games on or after {on}; a projection built "
                "from it would see its own outcome"
            )

    return result
