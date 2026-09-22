"""Temperature and wind at first pitch -- the one game-day input that moved runs.

Seven kinds of game-specific information were tested against the models in one
pass: starter velocity and recent form, travel and time zones, the home-plate
umpire's zone, catcher framing, hot lineups, look-alike games, and weather.
Weather was the only one to clear the bar set before any result was seen
(Bonferroni for seven tests, |z| > 2.69). On 23,928 held-out team-games,
2022-26, paired per game on the runs model's log-likelihood:

    temperature + wind    +0.000868 per team-game, z = +3.53, better in 5/5 seasons

Fitted on everything, a 20-degree swing is worth about 4.7% of a club's runs
(z = +8.78 on the coefficient), which is the direction the physics says -- warm
air is thin and the ball carries. Wind is the weaker half (z = +2.68) because it
carries no direction here: fifteen miles an hour blowing out and blowing in are
the same number. Temperature alone scored marginally better (z = +3.95); the
pair is what was fixed in advance, so the pair is what ships.

Sources, both free and neither typed in by hand:

  MLB statsapi venues    coordinates, roof type and time zone per park
  Open-Meteo             hourly temperature and wind: the historical archive for
                         games already played, the forecast for tonight's

That split is a train/serve difference and is recorded as one. The model is
fitted on measured conditions and served a same-day forecast. A temperature
forecast a few hours out typically misses by a degree or two; at 0.23% of runs
per degree that is well inside the noise, but it is not zero.

Roofs: a dome's outdoor weather is not the weather the game is played in, so
domes are held at DOME_TEMP_F and calm. That constant is an assumption, not a
measurement -- domes are climate-controlled to about room temperature -- and it
touches roughly 3% of games. A retractable roof's state on a given night is in
no source available here, so those games keep the outdoor reading.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from guards_report.sources.http import get_json

STATSAPI = "https://statsapi.mlb.com/api/v1"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"

WEATHER_COLUMNS = ["temp_f", "wind_mph"]

# A stated assumption, not a measurement: see the module docstring.
DOME_TEMP_F = 72.0

# Open-Meteo weighs a request by its volume, not as one call, and a multi-year
# hourly pull spends hundreds of calls against an hourly quota. One season per
# request, a pause between them, and a long wait on 429 keep a backfill inside
# it; the first attempt at this without pacing lost fifteen parks to the limit.
REQUEST_PAUSE_SECONDS = 4.0


def _dir(root: Path) -> Path:
    d = Path(root) / "weather"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get(url: str, params: dict[str, Any], *, attempts: int = 6) -> Any:
    """A governed request, with the patience a volume-metered quota needs.

    Every call goes through `sources.http.get_json`, which owns the
    process-wide spacing and the short retries -- the project's rule, enforced
    by a test, after ungoverned requests stopped a refresh with nothing raised.
    What it cannot know is that Open-Meteo refuses by quota for minutes at a
    time, so a refusal here waits the quota out rather than spending its
    retries inside one window.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return get_json(url, params=params, timeout=120)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < attempts - 1:
                time.sleep(300 if attempt < 2 else 900)
    raise RuntimeError(f"gave up on {url}: {last}")


# ---------------------------------------------------------------- venues ---

def venue_facts(root: Path, venue_ids=None, *, refresh: bool = False) -> pd.DataFrame:
    """Coordinates, roof type and time zone for each park, from MLB's own API.

    Cached. A venue not yet in the cache is fetched on demand, so a new park or
    a neutral-site game does not need a manual step.
    """
    path = _dir(root) / "venues.parquet"
    have = pd.read_parquet(path) if path.exists() and not refresh else pd.DataFrame(
        columns=["venue_id", "venue_name", "lat", "lon", "roof", "tz_id", "tz_offset"])
    # Not `venue_ids or []`: callers pass numpy arrays, and an array's truth
    # value is ambiguous -- that raises instead of meaning "none given".
    want = {int(v) for v in (list(venue_ids) if venue_ids is not None else [])
            if v == v}
    missing = sorted(want - set(have["venue_id"].dropna().astype(int)))
    if not missing and len(have):
        return have

    payload = _get(f"{STATSAPI}/venues", {
        "venueIds": ",".join(map(str, missing)) if missing else "",
        "hydrate": "location,fieldInfo,timezone",
    })
    rows = []
    for v in payload.get("venues", []):
        coord = ((v.get("location") or {}).get("defaultCoordinates") or {})
        rows.append({
            "venue_id": v.get("id"),
            "venue_name": v.get("name"),
            "lat": coord.get("latitude"),
            "lon": coord.get("longitude"),
            "roof": (v.get("fieldInfo") or {}).get("roofType"),
            "tz_id": (v.get("timeZone") or {}).get("id"),
            "tz_offset": (v.get("timeZone") or {}).get("offset"),
        })
    merged = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
    merged = merged.drop_duplicates("venue_id", keep="last")
    merged.to_parquet(path, index=False)
    return merged


# ---------------------------------------------------------------- roofs ----

def apply_roof(temp_f, wind_mph, roof):
    """Conditions the game is played in, given what the roof does.

    Vectorised over arrays or applied to scalars alike.
    """
    temp = np.asarray(temp_f, dtype=float).copy()
    wind = np.asarray(wind_mph, dtype=float).copy()
    dome = np.asarray(roof, dtype=object) == "Dome"
    temp[dome] = DOME_TEMP_F
    wind[dome] = 0.0
    return temp, wind


# ---------------------------------------------------------------- history --

def _start_times(season: int) -> pd.DataFrame:
    """First-pitch time for every regular-season game in a season."""
    sched = _get(f"{STATSAPI}/schedule", {"sportId": 1, "season": int(season),
                                          "gameType": "R"})
    rows = []
    for day in sched.get("dates", []):
        for g in day.get("games", []):
            rows.append({"game_pk": g.get("gamePk"), "start_utc": g.get("gameDate")})
    return pd.DataFrame(rows).drop_duplicates("game_pk", keep="last")


def history(root: Path, games: pd.DataFrame, *, fetch: bool = True,
            verbose: bool = False, attempts: int = 6) -> pd.DataFrame:
    """Measured conditions at first pitch for every game in `games`.

    Cached per game. With `fetch`, anything missing is pulled one venue-season
    at a time; without it, only the cache is read, which is what a fit on
    another machine wants when it must not reach the network.
    """
    path = _dir(root) / "history.parquet"
    columns = ["game_pk", "temp_f", "wind_mph", "venue_id", "roof"]
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=columns)
    # An empty cache written from an empty frame reads back with no columns at
    # all, and every lookup below would raise. Missing weather must degrade to
    # an average night, never to a failed fit.
    have = have.reindex(columns=columns)
    known = set(have.loc[have["temp_f"].notna(), "game_pk"])
    todo = games[~games["game_pk"].isin(known)][["game_pk", "season", "game_date",
                                                  "venue_id"]]

    if fetch and len(todo):
        venues = venue_facts(root, todo["venue_id"].dropna().unique())
        vmap = venues.dropna(subset=["lat", "lon"]).set_index("venue_id")
        # A park with no coordinates can never be filled -- neutral sites
        # abroad, mostly. Dropping those first keeps every refit from spending
        # a schedule request per season to rediscover it.
        todo = todo[todo["venue_id"].isin(vmap.index)]
    if fetch and len(todo):
        starts = pd.concat([_start_times(s) for s in sorted(todo["season"].unique())],
                           ignore_index=True)
        todo = todo.merge(starts, on="game_pk", how="left")
        todo["start_utc"] = pd.to_datetime(todo["start_utc"], utc=True)
        rows = []
        for (venue_id, season), blk in todo.groupby(["venue_id", "season"]):
            if venue_id not in vmap.index:
                continue
            v = vmap.loc[venue_id]
            dates = pd.to_datetime(blk["game_date"])
            try:
                data = _get(ARCHIVE, {
                    "latitude": v["lat"], "longitude": v["lon"],
                    "start_date": dates.min().date().isoformat(),
                    "end_date": dates.max().date().isoformat(),
                    "hourly": "temperature_2m,wind_speed_10m",
                    "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
                    "timezone": "UTC",
                }, attempts=attempts)
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"  weather {v['venue_name']} {season}: {exc}")
                continue
            hourly = data.get("hourly") or {}
            wx = pd.DataFrame({
                "hour": pd.to_datetime(hourly.get("time", []), utc=True),
                "temp_f": hourly.get("temperature_2m", []),
                "wind_mph": hourly.get("wind_speed_10m", []),
            }).set_index("hour")
            b = blk.dropna(subset=["start_utc"]).copy()
            b["hour"] = b["start_utc"].dt.floor("h")
            joined = b.join(wx, on="hour", how="left")
            joined["roof"] = v["roof"]
            rows.append(joined[["game_pk", "temp_f", "wind_mph", "venue_id", "roof"]])
            time.sleep(REQUEST_PAUSE_SECONDS)
        if rows:
            have = pd.concat([have, *rows], ignore_index=True)
            have = have.sort_values("temp_f", na_position="first") \
                .drop_duplicates("game_pk", keep="last")
            have.to_parquet(path, index=False)

    return have[have["game_pk"].isin(set(games["game_pk"]))]


def attach(data: pd.DataFrame, root: Path, games: pd.DataFrame, *,
           fetch: bool = True, verbose: bool = False) -> pd.DataFrame:
    """Add the weather columns to a per-team-game frame, roofs applied.

    Both sides of a game share one reading. A game with no reading keeps NaN,
    which the design matrix fills with the column mean -- no weather is treated
    as average weather, not as a cold night.
    """
    wx = history(root, games, fetch=fetch, verbose=verbose)
    out = data.merge(wx[["game_pk", "temp_f", "wind_mph", "roof"]],
                     on="game_pk", how="left")
    temp, wind = apply_roof(out["temp_f"], out["wind_mph"], out["roof"])
    out["temp_f"], out["wind_mph"] = temp, wind
    return out.drop(columns=["roof"])


# ---------------------------------------------------------------- tonight --

def forecast(root: Path, venue_id: int | None,
             start: datetime | None) -> dict[str, Any] | None:
    """Tonight's conditions at first pitch, roof applied, or None.

    None is the honest answer when any part is missing -- no park, no first
    pitch time, or no forecast -- and the model reads it as average weather.
    """
    if venue_id is None or start is None:
        return None
    try:
        venues = venue_facts(root, [venue_id])
        row = venues[venues["venue_id"] == int(venue_id)]
        if row.empty:
            return None
        v = row.iloc[0]
        if v["roof"] == "Dome":
            return {"temp_f": DOME_TEMP_F, "wind_mph": 0.0, "roof": "Dome",
                    "source": "dome"}
        if pd.isna(v["lat"]) or pd.isna(v["lon"]):
            return None
        at = start.astimezone(timezone.utc)
        day = at.date().isoformat()
        data = _get(FORECAST, {
            "latitude": v["lat"], "longitude": v["lon"],
            "start_date": day, "end_date": day,
            "hourly": "temperature_2m,wind_speed_10m",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "timezone": "UTC",
        }, attempts=2)
        hourly = data.get("hourly") or {}
        hours = pd.to_datetime(hourly.get("time", []), utc=True)
        want = pd.Timestamp(at).floor("h")
        match = np.where(hours == want)[0]
        if not len(match):
            return None
        i = int(match[0])
        return {"temp_f": float(hourly["temperature_2m"][i]),
                "wind_mph": float(hourly["wind_speed_10m"][i]),
                "roof": v["roof"], "source": "forecast"}
    except Exception:  # noqa: BLE001
        # Weather is an input, not a dependency: a forecast outage leaves the
        # projection at average conditions rather than failing the build.
        return None
