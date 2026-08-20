"""Historical game corpus — the training data every projection is built on.

One row per *game* (not per team-game), carrying only what the source states:
who played, where, when, who started, and what happened. Nothing is derived
here. Feature construction is a separate step precisely so that the raw record
stays auditable and so that a leaking feature cannot hide inside the loader.

The corpus is cached to Parquet. Finished games never change, so a season is
fetched once and re-read from disk thereafter -- model iteration wants
millisecond reads, and 27,000 rows is a rounding error on disk.

Ordering matters more than usual here. Everything downstream depends on being
able to say "these are the games that had already been played on date D", so
rows are sorted by date and the ordering is asserted rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from guards_report.config import COMPETITIVE_GAME_TYPES, MLB_SPORT_ID, STATSAPI_BASE

# Statcast begins in 2015, and the pitch-level features later projections need
# do not exist before it. Starting the corpus there keeps every projection on
# the same footing.
FIRST_SEASON = 2015

# Columns the corpus guarantees. Anything absent from the source is None rather
# than imputed -- a missing value is information, and silently filling it is how
# a model learns from something that was not there.
COLUMNS = (
    "game_pk", "game_date", "season", "game_type",
    "home_team_id", "away_team_id", "home_team", "away_team",
    "venue_id", "venue_name", "day_night",
    "home_runs", "away_runs", "home_hits", "away_hits",
    "home_errors", "away_errors",
    "home_starter_id", "away_starter_id",
    "home_win", "innings", "extra_innings",
)


@dataclass(frozen=True)
class SeasonFetch:
    """What one season's fetch produced, for reporting and auditing."""

    season: int
    games: int
    dropped_no_score: int
    dropped_no_starter: int

    @property
    def kept(self) -> int:
        return self.games


def _schedule_url(season: int) -> str:
    return (
        f"{STATSAPI_BASE}/v1/schedule"
        f"?sportId={MLB_SPORT_ID}&season={season}"
        f"&startDate={season}-03-01&endDate={season}-11-30"
        f"&gameTypes={','.join(COMPETITIVE_GAME_TYPES)}"
        f"&hydrate=team,linescore,decisions,probablePitcher,venue"
    )


def _row(game: dict[str, Any]) -> dict[str, Any] | None:
    """One schedule entry as a corpus row, or None if it cannot be used.

    A game is only usable as training data if it finished and both sides have a
    score. Everything else -- postponed, suspended, in progress -- is dropped
    and counted, never patched.
    """
    status = (game.get("status") or {}).get("abstractGameState")
    if status != "Final":
        return None

    teams = game.get("teams") or {}
    home, away = teams.get("home") or {}, teams.get("away") or {}
    linescore = game.get("linescore") or {}
    ls_teams = linescore.get("teams") or {}
    ls_home, ls_away = ls_teams.get("home") or {}, ls_teams.get("away") or {}

    home_runs, away_runs = ls_home.get("runs"), ls_away.get("runs")
    if home_runs is None or away_runs is None:
        return None
    # A tie is not a decidable outcome. They are vanishingly rare in the modern
    # game but would silently corrupt a binary target.
    if home_runs == away_runs:
        return None

    innings = linescore.get("currentInning") or len(linescore.get("innings") or [])

    return {
        "game_pk": game.get("gamePk"),
        "game_date": date.fromisoformat(game["officialDate"]),
        "season": int(game.get("season") or game["officialDate"][:4]),
        "game_type": game.get("gameType"),
        "home_team_id": (home.get("team") or {}).get("id"),
        "away_team_id": (away.get("team") or {}).get("id"),
        "home_team": (home.get("team") or {}).get("abbreviation"),
        "away_team": (away.get("team") or {}).get("abbreviation"),
        "venue_id": (game.get("venue") or {}).get("id"),
        "venue_name": (game.get("venue") or {}).get("name"),
        "day_night": game.get("dayNight"),
        "home_runs": int(home_runs),
        "away_runs": int(away_runs),
        "home_hits": ls_home.get("hits"),
        "away_hits": ls_away.get("hits"),
        "home_errors": ls_home.get("errors"),
        "away_errors": ls_away.get("errors"),
        "home_starter_id": (home.get("probablePitcher") or {}).get("id"),
        "away_starter_id": (away.get("probablePitcher") or {}).get("id"),
        "home_win": int(home_runs > away_runs),
        "innings": innings,
        "extra_innings": int(bool(innings and innings > 9)),
    }


def fetch_season(season: int, *, archiver: Any = None) -> tuple[list[dict], SeasonFetch]:
    """Every finished game of one season, as corpus rows.

    Uses the archiver when one is supplied so the payload is recorded like every
    other fetch in this project; falls back to a plain request otherwise, which
    keeps the corpus builder usable from a scratch script.
    """
    if archiver is not None:
        from guards_report.sources.http import fetch

        payload = fetch(
            _schedule_url(season), source="mlb-statsapi", archiver=archiver, params={}
        ).json()
    else:
        import json
        import urllib.request

        with urllib.request.urlopen(_schedule_url(season), timeout=120) as response:
            payload = json.load(response)

    rows: list[dict[str, Any]] = []
    no_score = no_starter = 0

    for day in payload.get("dates", []):
        for game in day.get("games", []):
            row = _row(game)
            if row is None:
                if (game.get("status") or {}).get("abstractGameState") == "Final":
                    no_score += 1
                continue
            if row["home_starter_id"] is None or row["away_starter_id"] is None:
                # Kept: the outcome is still valid training data for models that
                # do not use starter features. Counted so the gap is visible.
                no_starter += 1
            rows.append(row)

    rows.sort(key=lambda r: (r["game_date"], r["game_pk"]))
    return rows, SeasonFetch(season, len(rows), no_score, no_starter)


def build(
    seasons: range | list[int],
    *,
    cache_dir: Path,
    archiver: Any = None,
    refresh: bool = False,
) -> "pd.DataFrame":  # noqa: F821 - pandas imported lazily
    """The corpus for the given seasons, fetching only what is not cached.

    Each season is cached separately. A completed season never changes, so it is
    fetched once; only the current season is worth refreshing.
    """
    import pandas as pd

    cache_dir.mkdir(parents=True, exist_ok=True)
    frames, reports = [], []

    for season in seasons:
        path = cache_dir / f"games-{season}.parquet"
        if path.exists() and not refresh:
            frames.append(pd.read_parquet(path))
            continue

        rows, report = fetch_season(season, archiver=archiver)
        frame = pd.DataFrame(rows, columns=list(COLUMNS))
        frame.to_parquet(path, index=False)
        frames.append(frame)
        reports.append(report)

    corpus = pd.concat(frames, ignore_index=True)

    # The schedule endpoint lists a handful of games twice -- 36 of 25,228,
    # suspended games that appear under both their original and their completion
    # entry. The duplicate rows carry identical outcomes, so this is a clean
    # de-duplication rather than a choice between conflicting records.
    #
    # It matters more than the count suggests: a duplicated game fans out
    # through every downstream join, silently double-weighting those games in
    # training and inflating any metric computed over them.
    before = len(corpus)
    corpus = corpus.drop_duplicates(subset="game_pk", keep="first")
    corpus.attrs["duplicates_dropped"] = before - len(corpus)

    corpus = corpus.sort_values(["game_date", "game_pk"]).reset_index(drop=True)

    # The whole feature layer depends on these two properties. Assert them
    # rather than trusting that every future edit preserves them -- a violation
    # of either shows up as a suspiciously good backtest, not as an error.
    assert corpus["game_date"].is_monotonic_increasing, "corpus must be date-ordered"
    assert not corpus["game_pk"].duplicated().any(), "game_pk must be unique"

    corpus.attrs["fetch_reports"] = reports
    return corpus
