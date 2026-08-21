"""Pitch-level corpus — the finest grain the public data offers.

The project has now failed the same way four times. Recent form, bullpen
quality, offence/defence ratings and a defensive proxy were each built from past
game results, and Elo is already an efficient summary of past game results, so
each arrived collinear and contributed nothing. Measured across 25,192 games,
L30 run differential correlates +0.129 with winning and +0.779 with Elo; remove
Elo and +0.003 survives against a standard error of 0.006.

The pattern is the lesson. A feature helps only when it carries information that
is **not a function of who won**, and at pitch level almost everything qualifies.
Release velocity, spin, movement and location describe what a pitcher did,
independent of whether the ball found a glove. A starter throwing 1.5 mph below
his own baseline is a different pitcher that night, and no outcome-derived
statistic knows it until the runs have already scored.

This is also the level the remaining projections need. Strikeouts are a pitch
outcome before they are a game outcome, and a hit or a home run is a batted-ball
event -- none of the three can be modelled honestly from game-level aggregates.

**Keeping every pitch is nearly free.** Savant answers a team-season query with
the full pitch stream regardless; the earlier version of this loader downloaded
all of it and discarded three quarters before writing. Measured on one
team-season, retaining every pitch costs 0.78 MB compressed -- about 0.26 GB for
the whole history -- against a second multi-hour pull later to recover what was
thrown away.
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

FIRST_SEASON = 2015

# Where the plate appearance happened, and to whom.
CONTEXT = [
    "game_date", "game_pk", "at_bat_number", "pitch_number",
    "inning", "inning_topbot", "batter", "pitcher", "stand", "p_throws",
    "home_team", "away_team",
    "balls", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b",
    # Times through the order. The penalty a starter pays on each pass is one of
    # the better-established effects in baseball and is not recoverable from any
    # season-level line.
    "n_thruorder_pitcher",
]

# What the pitcher actually did, independent of the result. This is the block
# that is not a function of who won, which is the whole reason for descending to
# this grain.
STUFF = [
    "pitch_type", "pitch_name",
    "release_speed", "effective_speed", "release_spin_rate", "release_extension",
    "release_pos_x", "release_pos_z", "arm_angle",
    "pfx_x", "pfx_z", "plate_x", "plate_z", "zone",
]

# What happened. `description` is the per-pitch result and `events` is populated
# only on the pitch that ends the plate appearance.
OUTCOME = [
    "description", "events", "type",
    "bb_type", "launch_speed", "launch_angle", "launch_speed_angle",
    "woba_value", "woba_denom",
    "estimated_woba_using_speedangle", "estimated_ba_using_speedangle",
    "delta_run_exp",
]

COLUMNS = CONTEXT + STUFF + OUTCOME

NUMERIC = [
    "game_pk", "at_bat_number", "pitch_number", "inning", "batter", "pitcher",
    "balls", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b",
    "n_thruorder_pitcher",
    "release_speed", "effective_speed", "release_spin_rate", "release_extension",
    "release_pos_x", "release_pos_z", "arm_angle",
    "pfx_x", "pfx_z", "plate_x", "plate_z", "zone",
    "launch_speed", "launch_angle", "launch_speed_angle",
    "woba_value", "woba_denom",
    "estimated_woba_using_speedangle", "estimated_ba_using_speedangle",
    "delta_run_exp",
]

# Savant refuses an unfiltered date range -- it returns an empty document rather
# than an error -- so a season is pulled one club at a time.
REQUEST_PAUSE = 1.5

MIN_GAMES = 40


def _get(params: dict, *, timeout: int = 900) -> str:
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
    # utf-8-sig, not utf-8: the export carries a byte-order mark, which binds to
    # the first header and made `pitch_type` invisible to a plain column check.
    return payload.decode("utf-8-sig", "replace")


def fetch_chunk(team: str, season: int, *, retries: int = 3) -> pd.DataFrame:
    """Every pitch one club batted against in one regular season."""
    params = {
        "all": "true",
        "hfSea": f"{season}|",
        # Regular season only; spring training would otherwise fold exhibition
        # pitches into the talent estimates.
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

    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return pd.DataFrame(columns=COLUMNS)

    frame = pd.DataFrame(rows)
    # Statcast added fields over time -- bat tracking and arm angle are recent --
    # so a column absent in 2015 is expected rather than an error. Anything the
    # models require is asserted where it is used, not here.
    available = [c for c in COLUMNS if c in frame.columns]
    frame = frame[available].copy()
    for column in NUMERIC:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["batting_team"] = team
    frame["season"] = season
    return frame


def _chunk_path(cache_dir: Path, team: str, season: int) -> Path:
    return cache_dir / f"{season}_{team}.parquet"


def build(seasons, teams, *, cache_dir: Path, verbose: bool = True) -> pd.DataFrame:
    """The pitch corpus, fetching only the team-seasons not already cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames, fetched, cached = [], 0, 0

    for season in seasons:
        for team in teams:
            path = _chunk_path(cache_dir, team, season)
            if path.exists():
                frames.append(pd.read_parquet(path))
                cached += 1
                continue

            frame = fetch_chunk(team, season)
            games = frame["game_pk"].nunique() if len(frame) else 0
            if len(frame) and games < MIN_GAMES:
                raise RuntimeError(
                    f"{team} {season}: only {games} games returned -- "
                    "check the abbreviation for this season"
                )
            frame.to_parquet(path, index=False, compression="zstd")
            frames.append(frame)
            fetched += 1
            if verbose:
                print(
                    f"  {season} {team:4} {len(frame):>7,} pitches  {games:>3} games",
                    flush=True,
                )
            time.sleep(REQUEST_PAUSE)

    corpus = pd.concat(frames, ignore_index=True)
    corpus["game_date"] = pd.to_datetime(corpus["game_date"]).dt.date
    corpus = corpus.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    ).reset_index(drop=True)

    assert corpus["game_date"].is_monotonic_increasing, "corpus must be date-ordered"
    corpus.attrs.update(chunks_fetched=fetched, chunks_cached=cached)
    return corpus


def season_teams(games: pd.DataFrame, season: int) -> list[str]:
    """Club abbreviations that actually played in a season.

    Read from the game corpus rather than hard-coded, so relocations and rebrands
    follow the source instead of a list that quietly goes stale.
    """
    rows = games[games["season"] == season]
    return sorted(set(rows["home_team"]) | set(rows["away_team"]))
