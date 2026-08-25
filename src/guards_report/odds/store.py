"""Keeping every price we ever saw.

Tonight's prices are tomorrow's evidence. A quote captured before first pitch
and thrown away is a closing-line comparison that can never be made, and the
closing line is the only scoreboard this project has that converges inside a
season.

So nothing is overwritten. Each capture appends, keyed by market, selection and
the moment it was taken, and the same market priced twice on the same night
keeps both rows. That is what makes "where was this line when we flagged it,
and where did it close" answerable at all.

Written through `projections.atomic` for the same reason the model artifacts
are: a half-written parquet read by the next build is a file that loads as a
shorter history rather than as an error.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from guards_report.odds import types
from guards_report.projections import atomic

COLUMNS = [
    "game_date", "game_pk", "market", "selection", "line",
    "american", "book", "captured_at",
]


def path_for(root: Path) -> Path:
    return Path(root) / "odds" / "quotes.parquet"


def load(root: Path) -> pd.DataFrame:
    """Every quote on record, or an empty frame with the right columns.

    The empty case has to carry its columns. A column-less frame merges into
    nothing downstream and the failure surfaces a long way from here -- which
    this project has already shipped once, on an empty bullpen history.
    """
    target = path_for(root)
    if not target.is_file():
        return pd.DataFrame(columns=COLUMNS)
    try:
        return pd.read_parquet(target)
    except (OSError, ValueError):
        return pd.DataFrame(columns=COLUMNS)


def append(root: Path, quotes: list[types.Quote]) -> int:
    """Add quotes to the record. Returns how many were new.

    Exact duplicates are dropped -- the same price, for the same selection, at
    the same instant -- because pressing the button twice should not double the
    history. Anything differing in price or timestamp is kept, since that is the
    line moving, which is the entire point of storing it.
    """
    if not quotes:
        return 0

    fresh = pd.DataFrame([q.as_row() for q in quotes], columns=COLUMNS)
    existing = load(root)

    combined = pd.concat([existing, fresh], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["game_date", "market", "selection", "line", "american",
                "book", "captured_at"],
        keep="first")

    atomic.write_frame(combined, path_for(root), index=False)
    return len(combined) - len(existing) if before else len(fresh)


def for_game(root: Path, game_date: date) -> list[types.Quote]:
    """Every quote captured for one date, oldest first."""
    frame = load(root)
    if not len(frame):
        return []
    stamps = pd.to_datetime(frame["game_date"]).dt.date
    frame = frame[stamps == game_date].sort_values("captured_at")
    return [_as_quote(row) for row in frame.itertuples()]


def latest_markets(root: Path, game_date: date) -> list[types.Market]:
    """The most recent price for each selection, grouped into markets.

    A night may be captured several times as lines move. Pricing against a
    stale row would compare our number to a price nobody can take any more, so
    the newest quote per selection wins.

    The key deliberately excludes the line. A total re-entered at 9 after being
    entered at 8.5 is the same market having moved, not a second market -- and
    keying on the line kept both, so the page showed each strikeout prop twice,
    both rows labelled with whichever line happened to be written last. Books do
    post alternate lines, but a report covering one game a night takes them one
    at a time, and the newest entry is the live one.
    """
    quotes = for_game(root, game_date)
    newest: dict[tuple, types.Quote] = {}
    for quote in quotes:                    # already oldest first
        newest[(quote.market, quote.selection, quote.book)] = quote

    # The record keeps every book, which is what makes a price history. Pricing
    # wants one of them per market -- otherwise the page shows the same bet once
    # per book, which is what it did.
    from guards_report.odds import client

    return client._one_book(
        [m for m in types.group(list(newest.values())) if m.complete])


def _as_quote(row) -> types.Quote:
    captured = getattr(row, "captured_at", None)
    if isinstance(captured, pd.Timestamp):
        captured = captured.to_pydatetime()
    game_date = getattr(row, "game_date")
    if isinstance(game_date, (pd.Timestamp, datetime)):
        game_date = game_date.date()

    line = getattr(row, "line", None)
    game_pk = getattr(row, "game_pk", None)
    return types.Quote(
        game_date=game_date,
        market=str(getattr(row, "market")),
        selection=str(getattr(row, "selection")),
        american=float(getattr(row, "american")),
        book=str(getattr(row, "book") or "manual"),
        captured_at=captured,
        game_pk=int(game_pk) if pd.notna(game_pk) else None,
        line=float(line) if pd.notna(line) else None,
    )


def record(
    root: Path,
    text: str,
    *,
    game_date: date,
    book: str = "manual",
    game_pk: int | None = None,
) -> list[types.Market]:
    """Parse typed prices, store them, and hand back the markets.

    The one entry point the app and the CLI both use, so a price cannot reach
    the page without also reaching the record. A play shown but never stored is
    a play whose closing line can never be checked.
    """
    from guards_report.odds import manual

    markets = manual.parse_markets(
        text, game_date=game_date, book=book, game_pk=game_pk,
        captured_at=datetime.now(timezone.utc))
    append(root, [q for m in markets for q in m.quotes])
    return markets
