"""Pulling posted prices from The Odds API.

One game a night keeps this well inside the free tier. Current odds are billed
at one credit per market per region, so a nightly pull of the moneyline, the
total and a handful of prop markets costs roughly two hundred credits a month
against an allowance of five hundred.

Two things this deliberately does not do.

It does not silently fall back to anything. If the key is missing or the request
fails, it says so and the caller keeps whatever was typed by hand -- a quiet
fallback to stale prices is how a page ends up comparing tonight's projection
against last week's line.

It does not spend credits it does not need. The response carries the remaining
allowance in its headers, and that figure is passed back so the app can show it
rather than discovering the limit by hitting it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

import requests

# Imported for the side effect: it loads .env, which is where the key lives.
# Reading os.environ without it returns nothing and the client reports itself
# as unconfigured while the key sits on disk two directories up.
from guards_report import config as _config  # noqa: F401
from guards_report.odds import types

BASE = "https://api.the-odds-api.com/v4"
SPORT = "baseball_mlb"
REGION = "us"
TIMEOUT = 20

# Their market names, mapped to ours. The prop markets live on a different
# endpoint and are only fetched for the game being reported on.
FEATURED = {
    "h2h": types.MONEYLINE,
    "totals": types.TOTAL,
    "spreads": types.RUNLINE,
}
# Markets where the player's name arrives in `description` and the side in
# `name`, rather than the name being the selection itself.
PLAYER_MARKETS = frozenset({
    types.STRIKEOUTS, types.HITS, types.HOME_RUNS, types.TOTAL_BASES})

PROPS = {
    "pitcher_strikeouts": types.STRIKEOUTS,
    "batter_hits": types.HITS,
    "batter_home_runs": types.HOME_RUNS,
    "batter_total_bases": types.TOTAL_BASES,
}


class OddsAPIError(RuntimeError):
    """Raised with something a person can act on, not a status code alone."""


@dataclass
class Pull:
    """What came back, and what it cost."""

    markets: list[types.Market] = field(default_factory=list)
    credits_remaining: int | None = None
    credits_used: int | None = None
    books: list[str] = field(default_factory=list)
    note: str = ""


def api_key() -> str:
    return os.environ.get("ODDS_API_KEY", "").strip()


def configured() -> bool:
    return bool(api_key())


def _get(path: str, params: dict[str, Any]) -> tuple[Any, dict]:
    key = api_key()
    if not key:
        raise OddsAPIError(
            "No ODDS_API_KEY set. Get a free key at the-odds-api.com and put "
            "ODDS_API_KEY=... in .env"
        )
    try:
        response = requests.get(
            f"{BASE}{path}", params={**params, "apiKey": key}, timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise OddsAPIError(f"Could not reach the odds API: {exc}") from exc

    if response.status_code == 401:
        raise OddsAPIError("The odds API rejected the key. Check ODDS_API_KEY.")
    if response.status_code == 429:
        raise OddsAPIError(
            "Out of credits for this month. The free tier is 500; the usage "
            "page at the-odds-api.com shows when it resets.")
    if not response.ok:
        raise OddsAPIError(
            f"The odds API returned {response.status_code}: "
            f"{response.text[:160]}")

    return response.json(), response.headers


def _as_int(headers: dict, name: str) -> int | None:
    try:
        return int(headers.get(name, ""))
    except (TypeError, ValueError):
        return None


def fetch(
    *,
    game_date: date,
    home_team: str,
    away_team: str,
    book: str = "",
    include_props: bool = True,
) -> Pull:
    """Tonight's prices for one game.

    `home_team` and `away_team` are full club names as the API spells them
    ("Cleveland Guardians"). Matching is done on those rather than on
    abbreviations because the API does not publish abbreviations, and guessing
    one would silently return the wrong game.
    """
    events, headers = _get(
        f"/sports/{SPORT}/odds",
        {
            "regions": REGION,
            "markets": ",".join(FEATURED),
            "oddsFormat": "american",
            "dateFormat": "iso",
        },
    )

    pull = Pull(
        credits_remaining=_as_int(headers, "x-requests-remaining"),
        credits_used=_as_int(headers, "x-requests-used"),
    )

    event = _match(events, game_date, home_team, away_team)
    if event is not None and _underway(event):
        # In-play prices move with the score and mean nothing against a pregame
        # projection. Left in, a game Cleveland was winning drifted the Angels
        # to +1600, our unchanged 41.9% cleared that by twenty-nine points, and
        # the page recommended staking almost eight percent of bankroll on it.
        pull.note = (
            "First pitch has passed, so the board is live in-play pricing. "
            "Nothing is compared against it -- those prices move with the score "
            "and our projection does not.")
        return pull
    if event is None:
        pull.note = (
            f"No {away_team} at {home_team} found on {game_date}. The board may "
            f"not be posted yet -- lines usually appear the evening before.")
        return pull

    quotes, books = _quotes_from(event, game_date, book)

    if include_props and event.get("id"):
        try:
            prop_event, prop_headers = _get(
                f"/sports/{SPORT}/events/{event['id']}/odds",
                {
                    "regions": REGION,
                    "markets": ",".join(PROPS),
                    "oddsFormat": "american",
                    "dateFormat": "iso",
                },
            )
            pull.credits_remaining = _as_int(
                prop_headers, "x-requests-remaining") or pull.credits_remaining
            extra, more_books = _quotes_from(prop_event, game_date, book)
            quotes.extend(extra)
            books.update(more_books)
        except OddsAPIError as exc:
            # Props are a bonus; losing them must not cost the sides and totals.
            pull.note = f"Sides and totals only -- props unavailable ({exc})"

    grouped = types.group(quotes)
    complete = [m for m in grouped if m.complete]
    pull.books = sorted(books)
    pull.markets = complete if book else _one_book(complete)

    # A market posted only one way -- home runs usually are, quoted to happen
    # and not to not happen -- cannot have its margin removed, so it is skipped.
    # Saying so beats a market silently missing from the page.
    one_sided = sorted({
        m.name for m in grouped if not m.complete
    } - {m.name for m in complete})
    if one_sided:
        note = (f"{', '.join(one_sided)} quoted one way only, so the margin "
                f"cannot be removed and they are not priced")
        pull.note = f"{pull.note} · {note}" if pull.note else note
    return pull


def _one_book(markets: list[types.Market]) -> list[types.Market]:
    """Keep one book per market, chosen by agreement with the others.

    Every book is its own market, so leaving them all in gives six rows per
    selection. Mixing books *inside* a market is worse: the best price on each
    side across two books can sum below 100%, which reads as free money and is
    really two markets stapled together. But the constraint only binds within a
    market -- choosing one book for the whole night silently drops every market
    it does not post, and books differ a lot, with five posting strikeouts and
    one posting home runs.

    Choosing the tightest margin was the first rule and it is not enough. On one
    night two books priced the same hitter to record a hit at -189 and +340,
    which is 65% against 23% -- irreconcilable, so one of them is not the bet it
    claims to be. Both had a margin near six percent, so tightness could not
    tell them apart, and the odd one won.

    The median across books can. A single mislabeled market cannot move it, so
    the book nearest it is the one describing the bet everyone else is
    describing. Margin breaks ties, which is what it was always good for.
    """
    from guards_report.betting import prices as price_math

    grouped: dict[tuple, list[types.Market]] = {}
    for market in markets:
        grouped.setdefault(
            (market.name, market.line, market.subject), []).append(market)

    def first_side(market: types.Market) -> float | None:
        decimals = market.decimal()
        if not decimals:
            return None
        fair = price_math.fair_probabilities(decimals, "multiplicative")
        return fair[0] if fair else None

    def margin(market: types.Market) -> float:
        decimals = market.decimal()
        if not decimals:
            return 9.0
        value = price_math.overround(decimals)
        return value if value is not None and value > 0 else 9.0

    kept: list[types.Market] = []
    for found in grouped.values():
        # Compare on the same selection across books, not on whichever side each
        # happened to list first.
        ordered = [
            types.Market(name=m.name,
                         quotes=sorted(m.quotes, key=lambda q: q.selection.lower()))
            for m in found
        ]
        priced = [(m, first_side(m)) for m in ordered]
        usable = [(m, p) for m, p in priced if p is not None]
        if not usable:
            continue
        if len(usable) == 1:
            kept.append(usable[0][0])
            continue

        values = sorted(p for _, p in usable)
        middle = len(values) // 2
        median = (values[middle] if len(values) % 2
                  else (values[middle - 1] + values[middle]) / 2.0)
        kept.append(
            sorted(usable, key=lambda pair: (abs(pair[1] - median),
                                             margin(pair[0]),
                                             pair[0].book))[0][0])
    return kept


def _underway(event: dict) -> bool:
    """Whether first pitch has already passed."""
    stamp = str(event.get("commence_time") or "")
    if not stamp:
        return False
    try:
        started = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return False
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) >= started


def _match(events: list[dict], game_date: date, home: str, away: str) -> dict | None:
    """The one game on this date, matched on the date baseball uses.

    Start times arrive in UTC, where a 9pm Eastern first pitch is stamped the
    following day. Accepting both the requested date and the day after was the
    first attempt and it is too loose: last night's game and tomorrow night's
    game then both match a request for today, and the earlier one wins -- so a
    build for tomorrow priced a game that had already been played, and every
    price on it was live in-play.

    Converting to Eastern and comparing the calendar date is exact. Every North
    American schedule is published on that clock, which is why the stamps look
    wrong in UTC in the first place.
    """
    home_key, away_key = home.strip().lower(), away.strip().lower()
    for event in events or []:
        if (str(event.get("home_team", "")).lower() != home_key
                or str(event.get("away_team", "")).lower() != away_key):
            continue
        if _local_date(event) == game_date:
            return event
    return None


def _local_date(event: dict) -> date | None:
    """The event's date on the clock the schedule is published in."""
    stamp = str(event.get("commence_time") or "")
    if not stamp:
        return None
    try:
        started = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    try:
        from zoneinfo import ZoneInfo

        return started.astimezone(ZoneInfo("America/New_York")).date()
    except Exception:  # noqa: BLE001 -- no tz database is not fatal
        # Four hours is the summer offset and this only has to land on the
        # right calendar day, not the right minute.
        return (started - timedelta(hours=4)).date()


def _quotes_from(
    event: dict, game_date: date, preferred: str,
) -> tuple[list[types.Quote], set[str]]:
    """Flatten one event's bookmakers into quotes.

    One book at a time. Mixing books inside a market would produce an overround
    that describes no bet anyone can place -- the best price on each side across
    two books can even sum below 100%, which reads as free money and is really
    two different markets stapled together.
    """
    captured = datetime.now(timezone.utc)
    quotes: list[types.Quote] = []
    books: set[str] = set()

    for bookmaker in event.get("bookmakers") or []:
        title = str(bookmaker.get("title") or bookmaker.get("key") or "book")
        books.add(title)
        if preferred and title.lower() != preferred.lower():
            continue

        for market in bookmaker.get("markets") or []:
            name = FEATURED.get(market.get("key")) or PROPS.get(market.get("key"))
            if name is None:
                continue
            for outcome in market.get("outcomes") or []:
                price = outcome.get("price")
                if price is None:
                    continue
                quotes.append(types.Quote(
                    game_date=game_date,
                    market=name,
                    selection=_selection(name, outcome),
                    american=float(price),
                    book=title,
                    captured_at=captured,
                    line=_point(outcome),
                ))

    return quotes, books


def _selection(market: str, outcome: dict) -> str:
    """The label our own beliefs are keyed on.

    Totals come back as "Over"/"Under"; props as the player's name with the
    side in `description` or `name` depending on the market. Both are
    normalized here so nothing downstream has to know which.
    """
    name = str(outcome.get("name") or "").strip()
    description = str(outcome.get("description") or "").strip()

    if market in (types.TOTAL, types.F5_TOTAL):
        return name.lower()
    if market in PLAYER_MARKETS and description:
        return f"{description} {name}".strip().lower()
    return name


def _point(outcome: dict) -> float | None:
    value = outcome.get("point")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
