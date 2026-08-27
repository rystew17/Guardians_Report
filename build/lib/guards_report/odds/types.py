"""What a posted price is, and what makes a set of them a market.

The distinction that earns its own module: a `Quote` is one price on one
outcome, and a `Market` is the complete set of outcomes that partition the
possibilities. Only a complete market can be de-vigged, because removing the
bookmaker's margin means normalizing across every outcome -- and normalizing
across an incomplete set silently invents probability.

That failure is quiet enough to be worth guarding. Given only the favorite's
-135 and nothing else, proportional scaling happily returns 1.0 for it, which
reads as a certainty rather than as a missing counterpart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

# Canonical market names. The manual parser accepts several spellings for each
# because they are typed by hand at speed, but everything downstream sees these.
MONEYLINE = "moneyline"
TOTAL = "total"
RUNLINE = "runline"
F5_MONEYLINE = "f5_moneyline"
F5_TOTAL = "f5_total"
STRIKEOUTS = "strikeouts"
HITS = "hits"
HOME_RUNS = "home_runs"
TOTAL_BASES = "total_bases"

TWO_WAY = frozenset({MONEYLINE, TOTAL, RUNLINE, F5_MONEYLINE, F5_TOTAL,
                     STRIKEOUTS, HITS, HOME_RUNS, TOTAL_BASES})

# Markets whose two sides carry the same number with opposite signs.
SPREAD_MARKETS = frozenset({RUNLINE})

# Markets books commonly post one way only -- "to hit a home run" is offered to
# happen and not to not happen. The bet is real and the price is real; what is
# missing is the counterpart that would let the margin be measured.
ONE_WAY_ALLOWED = frozenset({HOME_RUNS})


@dataclass(frozen=True)
class Quote:
    """One price, on one outcome, from one book, at one moment."""

    game_date: date
    market: str
    selection: str
    american: float
    book: str = "manual"
    captured_at: datetime | None = None
    game_pk: int | None = None
    line: float | None = None     # the number for totals, run lines and props

    @property
    def subject(self) -> str:
        """Whose bet this is, for markets quoted per player.

        A market is the complete set of outcomes that partition one bet, and on
        a player prop that means one player -- "Mike Trout over 0.5 hits" and
        "Mike Trout under 0.5" are a market; Trout's over and Jose Ramirez's
        under are two halves of two different ones.

        Without this every hitter on the card collapsed into a single market of
        thirty-four quotes, which is never complete, so the entire hits board
        was dropped as "quoted one way only".
        """
        parts = self.selection.strip().lower().rsplit(" ", 1)
        if len(parts) == 2 and parts[1] in ("over", "under"):
            return parts[0]
        return ""

    def as_row(self) -> dict:
        return {
            "game_date": self.game_date,
            "game_pk": self.game_pk,
            "market": self.market,
            "selection": self.selection,
            "line": self.line,
            "american": self.american,
            "book": self.book,
            "captured_at": self.captured_at,
        }


@dataclass(frozen=True)
class Market:
    """Every outcome of one bet, priced together.

    `complete` is what the de-vig gates on. A two-way market with one side
    missing is not a market yet, however reasonable the single price looks.
    """

    name: str
    quotes: list[Quote] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        if self.name in ONE_WAY_ALLOWED:
            return len(self.quotes) >= 1
        if self.name in TWO_WAY:
            return len(self.quotes) == 2
        return len(self.quotes) >= 2

    @property
    def one_way(self) -> bool:
        """Whether the margin has to be assumed rather than measured."""
        return len(self.quotes) == 1

    @property
    def book(self) -> str:
        books = {q.book for q in self.quotes}
        return books.pop() if len(books) == 1 else "mixed"

    @property
    def subject(self) -> str:
        """The player this market is about, empty for team and game markets."""
        found = {q.subject for q in self.quotes if q.subject}
        return found.pop() if len(found) == 1 else ""

    @property
    def line(self) -> float | None:
        lines = {q.line for q in self.quotes if q.line is not None}
        if len(lines) == 1:
            return lines.pop()
        # A spread's sides differ only in sign; report the magnitude rather than
        # nothing, which is what the page and the belief lookup need.
        if self.name in SPREAD_MARKETS and lines:
            magnitudes = {abs(v) for v in lines}
            if len(magnitudes) == 1:
                return magnitudes.pop()
        return None

    def line_for(self, selection: str) -> float | None:
        """The number as posted to one side, sign included."""
        wanted = selection.strip().lower()
        for quote in self.quotes:
            if quote.selection.strip().lower() == wanted:
                return quote.line
        return None

    def american(self) -> list[float]:
        return [q.american for q in self.quotes]

    def decimal(self) -> list[float] | None:
        """Prices as decimal odds, or None if any of them will not convert.

        Exists so the units cannot be confused at a call site. Everything in
        `betting.prices` works in decimals; quotes are stored American because
        that is how they are posted and typed. Handing American odds to a
        function expecting decimals is not an error that announces itself --
        -135 simply reads as a price below evens and the whole market comes
        back unpriceable, which looks like a quiet night.
        """
        from guards_report.betting import prices

        out = []
        for quote in self.quotes:
            converted = prices.american_to_decimal(quote.american)
            if converted is None:
                return None
            out.append(converted)
        return out

    def selections(self) -> list[str]:
        return [q.selection for q in self.quotes]


def group(quotes: list[Quote]) -> list[Market]:
    """Collect quotes into markets, keyed by market name and line.

    The line is part of the key because a total of 8.5 and a total of 9 are
    different bets. Grouping them together would pair the over on one number
    with the under on another, and the resulting overround would look plausible
    while describing nothing that exists.
    """
    buckets: dict[tuple, list[Quote]] = {}
    for quote in quotes:
        # A spread is posted as -1.5 to one side and +1.5 to the other. They are
        # one bet, so the magnitude keys them; keying on the signed number split
        # every run line into two one-sided halves, neither de-viggable, both
        # reported as "quoted one way only".
        number = (abs(quote.line) if quote.line is not None
                  and quote.market in SPREAD_MARKETS else quote.line)
        key = (quote.market, number, quote.book, quote.game_date,
               quote.subject)
        buckets.setdefault(key, []).append(quote)
    return [Market(name=key[0], quotes=found) for key, found in buckets.items()]
