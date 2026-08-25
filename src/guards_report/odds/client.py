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
from datetime import date, datetime, timezone
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
PROPS = {
    "pitcher_strikeouts": types.STRIKEOUTS,
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

    complete = [m for m in types.group(quotes) if m.complete]
    pull.books = sorted(books)
    pull.markets = complete if book else _one_book(complete)
    return pull


def _one_book(markets: list[types.Market]) -> list[types.Market]:
    """Keep a single book's prices rather than all of them.

    Every book on the board is its own market, so leaving them all in gives the
    page six rows per selection and an unreadable table. Mixing them is worse:
    the best price on each side across two books can sum below 100%, which reads
    as free money and is really two different markets stapled together.

    The book kept is the one pricing tightest -- lowest average margin across
    the markets it covers, among those covering the most. That is the closest
    thing to a sharp price on a US retail board, and it is the price we would
    actually want to have taken.
    """
    from guards_report.betting import prices as price_math

    by_book: dict[str, list[types.Market]] = {}
    for market in markets:
        by_book.setdefault(market.book, []).append(market)
    if not by_book:
        return []

    def rank(entry) -> tuple:
        name, found = entry
        margins = []
        for market in found:
            decimals = market.decimal()
            if not decimals:
                continue
            margin = price_math.overround(decimals)
            if margin is not None:
                margins.append(margin)
        average = sum(margins) / len(margins) if margins else 9.0
        # Most markets first, then tightest pricing.
        return (-len(found), average, name)

    return sorted(by_book.items(), key=rank)[0][1]


def _match(events: list[dict], game_date: date, home: str, away: str) -> dict | None:
    home_key, away_key = home.strip().lower(), away.strip().lower()
    for event in events or []:
        if (str(event.get("home_team", "")).lower() != home_key
                or str(event.get("away_team", "")).lower() != away_key):
            continue
        stamp = str(event.get("commence_time", ""))[:10]
        # A 7pm Eastern start is stamped the *next* day in UTC, so matching the
        # game date alone would miss every night game -- which is most of them.
        if stamp in _acceptable_stamps(game_date):
            return event
    return None


def _acceptable_stamps(game_date: date) -> set[str]:
    from datetime import timedelta

    return {
        game_date.isoformat(),
        (game_date + timedelta(days=1)).isoformat(),
    }


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
    if market == types.STRIKEOUTS and description:
        return f"{description} {name}".strip().lower()
    return name


def _point(outcome: dict) -> float | None:
    value = outcome.get("point")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
