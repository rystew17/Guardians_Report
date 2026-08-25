"""Closing line value: the only scoreboard that reports back inside a season.

Profit and loss cannot settle whether this works. Detecting a 2% return edge at
80% power needs roughly 17,800 bets, which at a handful of plays a night is
about nineteen seasons -- and at one play a night, ninety-six. A losing year and
a genuine edge are statistically the same object on any horizon anyone will
wait for.

Closing line value asks a different and much cheaper question: after we flagged
a selection, did the market move toward us? It is near-continuous rather than
win-or-lose, so its variance is a fraction of a bet's, and it converges in the
low hundreds of observations. It also needs no outcome at all -- only the price
at first pitch.

Measured on de-vigged probabilities rather than on raw prices, because the
margin itself moves. A book that widens from -110 to -115 on both sides has not
changed its opinion, and reading that as line movement would credit us with
value the market never conceded.

The sign convention, since it reads correctly in both directions and would
never announce an inversion: **positive means the market came to us.** We took
a side at a fair probability of 0.52, it closed at 0.55, the market ended up
agreeing with us more than when we started, and CLV is +0.03.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from guards_report.betting import prices
from guards_report.betting.edge import Play
from guards_report.odds import types
from guards_report.projections import atomic

COLUMNS = [
    "game_date", "market", "selection", "line", "book",
    "american_taken", "p_market_taken", "p_model", "sigma",
    "edge", "z", "stake", "action", "flagged_at",
]


def path_for(root: Path) -> Path:
    return Path(root) / "odds" / "plays.parquet"


def load(root: Path) -> pd.DataFrame:
    target = path_for(root)
    if not target.is_file():
        return pd.DataFrame(columns=COLUMNS)
    try:
        return pd.read_parquet(target)
    except (OSError, ValueError):
        return pd.DataFrame(columns=COLUMNS)


def log(
    root: Path,
    plays: list[Play],
    *,
    game_date: date,
    actions: dict[str, str] | None = None,
    book: str = "manual",
    flagged_at: datetime | None = None,
) -> int:
    """Record what we thought, at the price available when we thought it.

    Everything considered is logged, not only what was recommended. A record of
    bets alone cannot answer whether the threshold is set correctly, because it
    contains no examples of what was turned down -- and "would the passes have
    beaten the close too" is exactly the question that decides whether the bar
    is too high.
    """
    if not plays:
        return 0

    flagged_at = flagged_at or datetime.now(timezone.utc)
    actions = actions or {}
    rows = [{
        "game_date": game_date,
        "market": play.market,
        "selection": play.selection,
        "line": play.line,
        "book": book,
        "american_taken": play.american,
        "p_market_taken": play.p_market,
        "p_model": play.p_model,
        "sigma": play.sigma,
        "edge": play.edge,
        "z": play.z,
        "stake": play.stake,
        "action": actions.get(f"{play.market}:{play.selection}", "considered"),
        "flagged_at": flagged_at,
    } for play in plays]

    fresh = pd.DataFrame(rows, columns=COLUMNS)
    combined = pd.concat([load(root), fresh], ignore_index=True)
    # The line is part of the key. Without it a total flagged at 8.5 and again
    # at 9 collapses to one row, and the record loses the bet it did not keep.
    combined = combined.drop_duplicates(
        subset=["game_date", "market", "selection", "line", "flagged_at"],
        keep="last")
    atomic.write_frame(combined, path_for(root), index=False)
    return len(fresh)


@dataclass(frozen=True)
class Movement:
    """One flagged selection, and where its price finished."""

    game_date: date
    market: str
    selection: str
    action: str
    p_taken: float
    p_closed: float
    edge: float

    @property
    def ours(self) -> bool:
        """Whether this is the side we leaned, rather than its counterpart.

        Both sides of every market are logged, which is deliberate -- a record
        containing only what was bet cannot answer whether the threshold is set
        right, because it holds no examples of what was turned down. But the two
        sides are exact complements, so pooling both is guaranteed to average to
        precisely zero and would report a dead flat scoreboard forever.
        """
        return self.edge > 0

    @property
    def points(self) -> float:
        """Positive means the market moved toward the side we flagged."""
        return self.p_closed - self.p_taken

    @property
    def beat_close(self) -> bool:
        return self.points > 0


def closing_probabilities(
    markets: list[types.Market], *, devig: str = "shin",
) -> dict[tuple[str, str], float]:
    """De-vigged probability per selection, from the last complete markets.

    Incomplete markets are skipped rather than approximated. A closing price
    with one side missing cannot be de-vigged, and pairing it against an
    opening price that could would compare two different quantities.
    """
    out: dict[tuple[str, str], float] = {}
    for market in markets:
        if not market.complete:
            continue
        decimals = market.decimal()
        if decimals is None:
            continue
        fair = prices.fair_probabilities(decimals, devig)
        if fair is None:
            continue
        for quote, probability in zip(market.quotes, fair):
            out[(market.name, quote.selection.lower())] = probability
    return out


def measure(
    root: Path,
    closing: dict[tuple[str, str], float],
    *,
    game_date: date,
) -> list[Movement]:
    """Compare what we flagged against where the market closed."""
    frame = load(root)
    if not len(frame):
        return []
    stamps = pd.to_datetime(frame["game_date"]).dt.date
    frame = frame[stamps == game_date]

    movements = []
    for row in frame.itertuples():
        key = (str(row.market), str(row.selection).lower())
        if key not in closing:
            continue
        movements.append(Movement(
            game_date=game_date,
            market=str(row.market),
            selection=str(row.selection),
            action=str(getattr(row, "action", "considered")),
            p_taken=float(row.p_market_taken),
            p_closed=float(closing[key]),
            edge=float(getattr(row, "edge", 0.0)),
        ))
    return movements


@dataclass(frozen=True)
class Record:
    """The running scoreboard, over every night measured so far."""

    n: int
    mean_points: float
    beat_rate: float
    n_bets: int
    mean_points_bets: float

    @property
    def enough(self) -> bool:
        """Whether the record is long enough to read as anything.

        A few hundred observations is where closing line value starts to
        separate from noise. Below that the mean is real but its error bar
        covers everything, and a page that reports it without saying so invites
        exactly the over-reading this project keeps guarding against.
        """
        return self.n >= 200


def summarize(movements: list[Movement]) -> Record | None:
    """Pool the movements. None when nothing has been measured yet."""
    ours = [m for m in movements if m.ours]
    if not ours:
        return None

    points = [m.points for m in ours]
    bets = [m.points for m in ours if m.action == "bet"]
    return Record(
        n=len(points),
        mean_points=sum(points) / len(points),
        beat_rate=sum(1 for p in points if p > 0) / len(points),
        n_bets=len(bets),
        mean_points_bets=(sum(bets) / len(bets)) if bets else 0.0,
    )
