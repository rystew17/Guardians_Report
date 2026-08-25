"""Reading prices typed by hand.

The report covers one game a night, so the whole input is two moneylines, a
total and perhaps a strikeout number -- fifteen seconds of typing. That makes
the manual path competitive with an API rather than a fallback to it, and it
has two genuine advantages: no vendor to depend on, and the price recorded is
the one actually available to whoever typed it.

The format is forgiving because it is typed at speed, in a hurry, probably on a
phone. Case, whitespace, the leading plus, and the several names every market
goes by are all absorbed. What is *not* absorbed is an incomplete market: one
side of a two-way bet is rejected rather than stored, because a market missing
its counterpart cannot be de-vigged and would otherwise sail through as a
certainty.

Accepted, all equivalent:

    ml CLE -135 DET +115
    moneyline CLE -135
    moneyline DET +115
    ML  cle  -135   det  +115

Totals and props carry their number:

    total o8.5 -105 u8.5 -115
    k Bibee o5.5 -120 u5.5 +100
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone

from guards_report.odds import types

# Every spelling a market goes by, mapped to the canonical name. Typed by hand,
# so the aliases matter more than they look.
ALIASES = {
    "ml": types.MONEYLINE, "moneyline": types.MONEYLINE, "money": types.MONEYLINE,
    "h2h": types.MONEYLINE, "line": types.MONEYLINE,
    "total": types.TOTAL, "tot": types.TOTAL, "ou": types.TOTAL,
    "o/u": types.TOTAL, "overunder": types.TOTAL,
    "rl": types.RUNLINE, "runline": types.RUNLINE, "spread": types.RUNLINE,
    "f5": types.F5_MONEYLINE, "f5ml": types.F5_MONEYLINE,
    "first5": types.F5_MONEYLINE, "f5line": types.F5_MONEYLINE,
    "f5total": types.F5_TOTAL, "f5tot": types.F5_TOTAL, "f5ou": types.F5_TOTAL,
    "k": types.STRIKEOUTS, "ks": types.STRIKEOUTS, "so": types.STRIKEOUTS,
    "strikeouts": types.STRIKEOUTS, "punchouts": types.STRIKEOUTS,
    "h": types.HITS, "hit": types.HITS, "hits": types.HITS,
    "hr": types.HOME_RUNS, "homer": types.HOME_RUNS,
    "homers": types.HOME_RUNS, "homeruns": types.HOME_RUNS,
    "tb": types.TOTAL_BASES, "bases": types.TOTAL_BASES,
    "totalbases": types.TOTAL_BASES,
}

# -135, +115, 135 all read as prices. A bare number without a sign is taken as
# negative only below 100, where no positive price exists -- above it the
# convention is genuinely ambiguous and guessing would flip the whole market.
_PRICE = re.compile(r"^[+-]?\d{3,5}$")
_LINE = re.compile(r"^(?:([ou]))?(\d+(?:\.\d+)?)$", re.IGNORECASE)


class ParseError(ValueError):
    """Raised with the offending line, so a typo can be found by eye."""


def _price(token: str) -> float | None:
    if not _PRICE.match(token):
        return None
    value = float(token.lstrip("+"))
    if not token.startswith(("+", "-")) and abs(value) < 100:
        return None
    return value


def parse(
    text: str,
    *,
    game_date: date,
    book: str = "manual",
    captured_at: datetime | None = None,
    game_pk: int | None = None,
) -> list[types.Quote]:
    """Read typed prices into quotes. Raises ParseError naming the bad line."""
    captured_at = captured_at or datetime.now(timezone.utc)
    quotes: list[types.Quote] = []

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue

        tokens = line.replace("/", " ").split()
        market = ALIASES.get(tokens[0].lower().replace("-", ""))
        if market is None:
            raise ParseError(f"unknown market in: {raw.strip()!r}")

        rest = tokens[1:]
        if not rest:
            raise ParseError(f"no prices in: {raw.strip()!r}")

        # Walk the remainder as (selection..., price) runs. A price closes the
        # selection that preceded it, which is what lets one line carry both
        # sides without needing a separator.
        on_this_line: list[types.Quote] = []
        pending: list[str] = []
        found = 0
        for token in rest:
            price = _price(token)
            if price is None:
                pending.append(token)
                continue
            if not pending:
                raise ParseError(f"price with no selection in: {raw.strip()!r}")

            selection, number = _split_selection(pending)
            on_this_line.append(types.Quote(
                game_date=game_date,
                market=market,
                selection=selection,
                american=price,
                book=book,
                captured_at=captured_at,
                game_pk=game_pk,
                line=number,
            ))
            pending = []
            found += 1

        if pending:
            raise ParseError(f"selection with no price in: {raw.strip()!r}")
        if not found:
            raise ParseError(f"no prices in: {raw.strip()!r}")

        quotes.extend(_fill_subjects(on_this_line))

    return quotes


def _fill_subjects(quotes: list[types.Quote]) -> list[types.Quote]:
    """Give the bare side of an over/under the subject typed beside it.

    Nobody types the name twice -- `k Bibee o5.5 -120 u5.5 +100` leaves the
    under with nothing to be under. Scoped to the one line it was typed on,
    because a line is where the intent lives: two hitters in the same market
    would otherwise have the second inherit the first's name, which is how a
    board of twenty hit props collapsed into one unpriceable market.
    """
    subject = ""
    for quote in quotes:
        if quote.subject:
            subject = quote.selection.strip().rsplit(" ", 1)[0]
            break
    if not subject:
        return quotes

    out = []
    for quote in quotes:
        side = quote.selection.strip().lower()
        if side in ("over", "under"):
            out.append(types.Quote(
                **{**vars(quote), "selection": f"{subject} {side}"}))
        else:
            out.append(quote)
    return out


def _split_selection(tokens: list[str]) -> tuple[str, float | None]:
    """Pull the number out of a selection, if it carries one.

    `o8.5` is over eight and a half; `Bibee o5.5` is Bibee over five and a
    half. The side is normalized to the words `over` and `under` so nothing
    downstream has to know about the shorthand.
    """
    words = list(tokens)
    number: float | None = None
    side: str | None = None

    for index, token in enumerate(words):
        match = _LINE.match(token)
        if match and (match.group(1) or "." in token or len(token) <= 4):
            prefix, value = match.groups()
            # A bare integer that is really a team name would be unusual, but a
            # bare integer with no over/under prefix and no decimal is more
            # likely a line than anything else in this position.
            number = float(value)
            if prefix:
                side = "over" if prefix.lower() == "o" else "under"
            words.pop(index)
            break

    label = " ".join(words).strip()
    if side and label:
        return f"{label} {side}", number
    if side:
        return side, number
    return label, number


def parse_markets(text: str, **kwargs) -> list[types.Market]:
    """Parse, group, and refuse anything that is not a complete market.

    An incomplete two-way market is the one input that must not pass. Only a
    full set of outcomes can be de-vigged, and de-vigging a single price
    normalizes it to 1.0 -- which reads as a certainty rather than as a missing
    other side.
    """
    # Subjects are filled in before grouping, not after. Markets are keyed by
    # player, so a bare "under" carries no subject yet and would land in its own
    # market -- leaving both halves of one bet looking one-sided.
    markets = types.group(parse(text, **kwargs))
    incomplete = [m for m in markets if not m.complete]
    if incomplete:
        described = ", ".join(
            f"{m.name}" + (f" {m.line:g}" if m.line is not None else "")
            for m in incomplete
        )
        raise ParseError(
            f"incomplete market (need both sides to remove the vig): {described}")
    return markets
