"""Starter game logs, and the as-of statistics built from them.

Every starter's season, start by start. This is the atom the starter features
are derived from, and it exists as raw per-start rows rather than season
aggregates for one reason: a feature for a game on date D may only use starts
*before* D. Season totals cannot give that, so they are never used.

The distinction that motivates the metric set here is that we are predicting a
*team* outcome, not a pitcher's true talent. FIP is built to strip defense out
in order to project a pitcher; but the run that scores on a botched double play
counts on the scoreboard exactly like the one that scores on a home run. So runs
allowed is carried alongside the fielding-independent measures, and the gap
between them is kept as a feature in its own right -- it is a measurement of the
defense playing behind this pitcher.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from guards_report.metrics import formulas as f

BATCH = 25          # pitcher ids per request; the endpoint takes a list
TIMEOUT = 120

# Counting stats summed over a window. Rates are computed from these sums, never
# averaged from per-game rates -- averaging rates weights a 1-inning relief
# outing the same as a complete game.
COUNTS = (
    "outs", "earnedRuns", "runs", "hits", "baseOnBalls", "strikeOuts",
    "homeRuns", "battersFaced", "hitByPitch", "gamesStarted",
)


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _outs(innings: Any) -> int:
    """MLB innings notation ("6.1" = six and one third) to whole outs."""
    try:
        whole, _, frac = str(innings).partition(".")
        return int(whole) * 3 + (int(frac) if frac else 0)
    except (TypeError, ValueError):
        return 0


def _url(ids: Iterable[int], season: int) -> str:
    joined = ",".join(str(i) for i in ids)
    return (
        "https://statsapi.mlb.com/api/v1/people"
        f"?personIds={joined}"
        f"&hydrate=stats(group=[pitching],type=[gameLog],season={season})"
    )


def fetch_season(pitcher_ids: list[int], season: int) -> list[dict[str, Any]]:
    """Per-start rows for the given pitchers in one season."""
    rows: list[dict[str, Any]] = []

    for start in range(0, len(pitcher_ids), BATCH):
        chunk = pitcher_ids[start:start + BATCH]
        with urllib.request.urlopen(_url(chunk, season), timeout=TIMEOUT) as response:
            payload = json.load(response)

        for person in payload.get("people", []):
            pid = person.get("id")
            for block in person.get("stats", []):
                if (block.get("type") or {}).get("displayName") != "gameLog":
                    continue
                for split in block.get("splits", []):
                    stat = split.get("stat") or {}
                    game = split.get("game") or {}
                    rows.append({
                        "pitcher_id": pid,
                        "season": season,
                        "game_date": split.get("date"),
                        "game_pk": game.get("gamePk"),
                        "is_home": bool(split.get("isHome", False)),
                        "outs": _outs(stat.get("inningsPitched")),
                        "earnedRuns": _int(stat.get("earnedRuns")),
                        "runs": _int(stat.get("runs")),
                        "hits": _int(stat.get("hits")),
                        "baseOnBalls": _int(stat.get("baseOnBalls")),
                        "strikeOuts": _int(stat.get("strikeOuts")),
                        "homeRuns": _int(stat.get("homeRuns")),
                        "battersFaced": _int(stat.get("battersFaced")),
                        "hitByPitch": _int(stat.get("hitByPitch")),
                        "gamesStarted": _int(stat.get("gamesStarted")),
                    })
    return rows


def build(corpus_frame, *, cache_dir: Path, refresh: bool = False):
    """Per-start logs for every pitcher who started a game in the corpus."""
    import pandas as pd

    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []

    for season, group in corpus_frame.groupby("season"):
        path = cache_dir / f"starts-{season}.parquet"
        if path.exists() and not refresh:
            frames.append(pd.read_parquet(path))
            continue

        ids = pd.unique(
            pd.concat([group["home_starter_id"], group["away_starter_id"]])
            .dropna().astype(int)
        )
        rows = fetch_season(sorted(int(i) for i in ids), int(season))
        frame = pd.DataFrame(rows)
        frame.to_parquet(path, index=False)
        frames.append(frame)

    logs = pd.concat(frames, ignore_index=True)
    logs["game_date"] = pd.to_datetime(logs["game_date"]).dt.date
    return logs.sort_values(["pitcher_id", "game_date"]).reset_index(drop=True)


def as_of_table(logs, *, league_hr_fb: float = 0.135):
    """Cumulative season-to-date statistics *before* each start.

    The shift is the whole point: row i carries the totals from starts 0..i-1,
    so a feature built from it cannot contain the game it is predicting. A
    pitcher's first start of a season has no prior data and is left null rather
    than filled -- "we do not know yet" is information, and imputing a league
    average would quietly tell the model something it could not have known.

    Returns one row per (pitcher, game), carrying both fielding-independent and
    runs-allowed measures plus the gap between them.
    """
    import numpy as np
    import pandas as pd

    logs = logs.sort_values(["pitcher_id", "season", "game_date"]).copy()
    grouped = logs.groupby(["pitcher_id", "season"], sort=False)

    prior = grouped[list(COUNTS)].cumsum() - logs[list(COUNTS)]
    prior["starts_prior"] = grouped.cumcount()

    innings = prior["outs"] / 3.0
    bf = prior["battersFaced"].replace(0, np.nan)
    ip = innings.replace(0, np.nan)

    out = pd.DataFrame({
        "pitcher_id": logs["pitcher_id"].to_numpy(),
        "season": logs["season"].to_numpy(),
        "game_date": logs["game_date"].to_numpy(),
        "game_pk": logs["game_pk"].to_numpy(),
        "starts_prior": prior["starts_prior"].to_numpy(),
        "ip_prior": innings.to_numpy(),
        # Runs actually allowed -- what the scoreboard uses.
        "era": (prior["earnedRuns"] * 9.0 / ip).to_numpy(),
        "ra9": (prior["runs"] * 9.0 / ip).to_numpy(),
        # Fielding-independent -- what projects the pitcher.
        "fip": (
            (13 * prior["homeRuns"] + 3 * (prior["baseOnBalls"] + prior["hitByPitch"])
             - 2 * prior["strikeOuts"]) / ip
        ).to_numpy(),
        "k_pct": (prior["strikeOuts"] / bf).to_numpy(),
        "bb_pct": (prior["baseOnBalls"] / bf).to_numpy(),
        "hr9": (prior["homeRuns"] * 9.0 / ip).to_numpy(),
        "whip": ((prior["hits"] + prior["baseOnBalls"]) / ip).to_numpy(),
        "ip_per_start": (innings / prior["starts_prior"].replace(0, np.nan)).to_numpy(),
    })

    # FIP needs its league constant to sit on the ERA scale; it is added per
    # season so the two are directly comparable and their gap is meaningful.
    for season, block in out.groupby("season"):
        mask = out["season"] == season
        constant = (
            out.loc[mask, "era"].mean(skipna=True)
            - out.loc[mask, "fip"].mean(skipna=True)
        )
        out.loc[mask, "fip"] = out.loc[mask, "fip"] + constant

    # The defense playing behind him. Positive means more runs are scoring than
    # his fielding-independent line accounts for.
    out["era_minus_fip"] = out["era"] - out["fip"]
    out["ra9_minus_fip"] = out["ra9"] - out["fip"]

    # Too few innings to mean anything. Kept as null rather than dropped so the
    # caller decides what to do with an unknown.
    thin = out["ip_prior"] < 10
    for column in ("era", "ra9", "fip", "k_pct", "bb_pct", "hr9", "whip",
                   "era_minus_fip", "ra9_minus_fip", "ip_per_start"):
        out.loc[thin, column] = np.nan

    return out


# ---------------------------------------------------------------------------
# All pitchers, not just starters -- required for bullpen availability
# ---------------------------------------------------------------------------


def season_pitcher_ids(season: int) -> list[int]:
    """Every pitcher on a roster in a season.

    The starter-only corpus cannot answer bullpen questions: the arms that
    matter most on a given night are the ones who never start.
    """
    url = f"https://statsapi.mlb.com/api/v1/sports/1/players?season={season}"
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        payload = json.load(response)

    return sorted(
        int(person["id"])
        for person in payload.get("people", [])
        if (person.get("primaryPosition") or {}).get("abbreviation") == "P"
    )


def build_all(seasons: Iterable[int], *, cache_dir: Path, refresh: bool = False):
    """Per-appearance logs for every pitcher, cached per season."""
    import pandas as pd

    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []

    for season in seasons:
        path = cache_dir / f"all-{season}.parquet"
        if path.exists() and not refresh:
            frames.append(pd.read_parquet(path))
            continue
        rows = fetch_season(season_pitcher_ids(int(season)), int(season))
        frame = pd.DataFrame(rows)
        frame.to_parquet(path, index=False)
        frames.append(frame)

    logs = pd.concat(frames, ignore_index=True)
    logs["game_date"] = pd.to_datetime(logs["game_date"]).dt.date
    return logs.sort_values(["pitcher_id", "game_date"]).reset_index(drop=True)
