"""Plate-appearance corpus — the level where the remaining signal lives.

Every team-level feature tested so far has failed the same way. Recent form,
bullpen quality, offence/defence ratings and a crude defensive proxy were all
built from past game results, and Elo is already an efficient summary of past
game results, so each one arrived collinear with it and contributed nothing.
Measured on 25,192 games, L30 run differential correlates +0.129 with winning
and +0.779 with Elo; strip Elo out and what remains is +0.003, against a
standard error of 0.006.

The way out is not a better summary of the same information. It is finer
information. A season contains roughly 185,000 plate appearances against 2,430
games, so player quality estimated here rests on about seventy times more
evidence than the game model can see -- and, unlike everything above, it is not
a function of which team happened to win.

Only PA-ending rows are kept. Statcast returns every pitch, but the outcome is
recorded once per plate appearance, and holding four times the rows to read one
field would make the cache large enough to be annoying for no gain.

The pull is chunked by team-season because Savant will not answer an unfiltered
date range -- it returns an empty document rather than an error, which is worth
knowing before trusting a date-windowed query. Each chunk is cached on arrival,
so an interrupted pull resumes instead of restarting.
"""

from __future__ import annotations

import csv
import gzip
import io
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

SAVANT_CSV = "https://baseballsavant.mlb.com/statcast_search/csv"

# Statcast's first full season, and the floor the rest of the project uses.
FIRST_SEASON = 2015

# What a plate appearance needs to be worth storing. `woba_value` is the run
# value Statcast assigns the event and `woba_denom` marks whether it counts
# toward the wOBA denominator -- together they are the linear-weights target,
# so no run values have to be hard-coded here and drift out of date.
COLUMNS = [
    "game_date", "game_pk", "at_bat_number", "batter", "pitcher",
    "stand", "p_throws", "events",
    "woba_value", "woba_denom", "estimated_woba_using_speedangle",
    "home_team", "away_team", "inning", "inning_topbot",
    # Base-out state, so run value can later be measured in context as well as
    # context-neutrally. Cheap to carry now, expensive to re-fetch later.
    "outs_when_up", "on_1b", "on_2b", "on_3b", "delta_run_exp",
]

NUMERIC = [
    "game_pk", "at_bat_number", "batter", "pitcher", "inning",
    "woba_value", "woba_denom", "estimated_woba_using_speedangle",
    "outs_when_up", "on_1b", "on_2b", "on_3b", "delta_run_exp",
]

# Savant rejects a request with no entity filter, so a season is pulled one club
# at a time. A club's batting PAs belong to exactly one club, so summing over
# every team covers each plate appearance once with no deduplication needed.
REQUEST_PAUSE = 1.5

# A team-season that comes back far short of a real schedule means the
# abbreviation or the season is wrong, not that the club played 12 games.
MIN_GAMES = 40


def _get(params: dict, *, timeout: int = 600) -> str:
    url = SAVANT_CSV + "?" + urllib.parse.urlencode(params, doseq=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "guards-report/1.0 (+scouting report)",
            "Accept-Encoding": "gzip",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            payload = gzip.decompress(payload)
    return payload.decode("utf-8", "replace")


def fetch_chunk(team: str, season: int, *, retries: int = 3) -> pd.DataFrame:
    """Every plate appearance one club batted in one regular season."""
    params = {
        "all": "true",
        "hfSea": f"{season}|",
        # Regular season only. Spring training and exhibition games would
        # otherwise fold into the talent estimates, and the report already
        # excludes them everywhere else.
        "hfGT": "R|",
        "player_type": "batter",
        "hfTeam": f"{team}|",
        "type": "details",
    }
    last: Exception | None = None
    for attempt in range(retries):
        try:
            text = _get(params)
            break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            time.sleep(4 * (attempt + 1))
    else:
        raise RuntimeError(f"{team} {season}: {last}")

    rows = [row for row in csv.DictReader(io.StringIO(text)) if row.get("events")]
    if not rows:
        return pd.DataFrame(columns=COLUMNS)

    frame = pd.DataFrame(rows)
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise RuntimeError(f"{team} {season}: Savant omitted {missing}")

    frame = frame[COLUMNS].copy()
    for column in NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["batting_team"] = team
    frame["season"] = season
    return frame


def _chunk_path(cache_dir: Path, team: str, season: int) -> Path:
    return cache_dir / f"{season}_{team}.parquet"


def build(
    seasons,
    teams,
    *,
    cache_dir: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """The full PA corpus, fetching only the team-seasons not already cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames, fetched, skipped = [], 0, 0

    for season in seasons:
        for team in teams:
            path = _chunk_path(cache_dir, team, season)
            if path.exists():
                frames.append(pd.read_parquet(path))
                skipped += 1
                continue

            frame = fetch_chunk(team, season)
            games = frame["game_pk"].nunique()
            if len(frame) and games < MIN_GAMES:
                raise RuntimeError(
                    f"{team} {season}: only {games} games returned -- "
                    "check the abbreviation for this season"
                )
            frame.to_parquet(path, index=False)
            frames.append(frame)
            fetched += 1
            if verbose:
                print(
                    f"  {season} {team:4} {len(frame):>6,} PA  {games:>3} games",
                    flush=True,
                )
            time.sleep(REQUEST_PAUSE)

    corpus = pd.concat(frames, ignore_index=True)
    corpus["game_date"] = pd.to_datetime(corpus["game_date"]).dt.date
    corpus = corpus.sort_values(
        ["game_date", "game_pk", "at_bat_number"]
    ).reset_index(drop=True)

    # The same guarantees the game corpus makes, for the same reason: everything
    # downstream asks "what had happened before date D", and that only means
    # something if the ordering is real.
    assert corpus["game_date"].is_monotonic_increasing, "PA corpus must be date-ordered"
    assert not corpus.duplicated(["game_pk", "at_bat_number"]).any(), (
        "a plate appearance must appear once per game"
    )

    corpus.attrs.update(chunks_fetched=fetched, chunks_cached=skipped)
    return corpus


def season_teams(games: pd.DataFrame, season: int) -> list[str]:
    """Club abbreviations that actually played in a season.

    Taken from the game corpus rather than hard-coded, so relocations and
    rebrands follow the source instead of a list that silently goes stale.
    """
    rows = games[games["season"] == season]
    return sorted(set(rows["home_team"]) | set(rows["away_team"]))
