"""Career honours, kept strictly apart from the season analysis.

The report grades a player on the current season and says so, because that is
what predicts tonight. It is also why a first-ballot career could read as "an
average bat" -- true of four months, ridiculous about the man.

This is the other half of that fix. Rather than let a career prior contaminate
a season measurement -- which would carry a declining thirty-four-year-old at
the level he held at twenty-seven -- the career is stated separately, as fact,
where it cannot be mistaken for a projection.

**The filter is an allowlist, not a keyword match.** The awards feed carries 151
distinct names for twenty players, and most are not career honours: Player of
the Week, Rookie of the Month, Home Run Derby Participant, Heart and Hustle, and
a long tail of minor-league selections -- `MiLB.com Organization All-Star`,
`PCL Mid-Season All-Star`, `Baseball America Double-A All-Star`. Substring
matching on "All-Star" pulls all of those in and turns a Triple-A journeyman
into a decorated veteran. Only the exact names below count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Canonical honour -> the exact feed names that mean it, most prestigious first.
# Order is the display order: a reader wants the MVPs before the All-Star
# selections, and three honours is a credential where nine is a list.
MAJOR_AWARDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("MVP", ("AL MVP", "NL MVP")),
    ("Cy Young", ("AL Cy Young", "NL Cy Young")),
    ("Rookie of the Year", ("Jackie Robinson AL Rookie of the Year",
                            "Jackie Robinson NL Rookie of the Year")),
    ("All-MLB First Team", ("All-MLB First Team",)),
    ("Hank Aaron Award", ("AL Hank Aaron Award", "NL Hank Aaron Award")),
    ("Platinum Glove", ("Rawlings AL Platinum Glove", "Rawlings NL Platinum Glove")),
    ("Reliever of the Year", ("Mariano Rivera AL Reliever of the Year",
                              "Trevor Hoffman NL Reliever of the Year")),
    ("Comeback Player of the Year", ("AL Comeback Player of the Year",
                                     "NL Comeback Player of the Year")),
    ("Gold Glove", ("Rawlings AL Gold Glove", "Rawlings NL Gold Glove")),
    ("Silver Slugger", ("AL Silver Slugger", "NL Silver Slugger")),
    ("Outstanding DH", ("Edgar Martinez Outstanding DH of the Year",)),
    ("All-MLB Second Team", ("All-MLB Second Team",)),
    ("All-Star", ("AL All-Star", "NL All-Star")),
)

_BY_NAME: dict[str, str] = {
    feed_name: honour
    for honour, feed_names in MAJOR_AWARDS
    for feed_name in feed_names
}
_RANK: dict[str, int] = {honour: i for i, (honour, _) in enumerate(MAJOR_AWARDS)}


@dataclass
class Honour:
    """One award, and every season it was won."""

    name: str
    seasons: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.seasons)


@dataclass
class Career:
    """What a player has done before tonight, as fact rather than forecast."""

    honours: list[Honour] = field(default_factory=list)
    games: int = 0
    seasons: int = 0
    totals: dict[str, Any] = field(default_factory=dict)

    @property
    def decorated(self) -> bool:
        return bool(self.honours)


def honours_from(awards: Any) -> list[Honour]:
    """The career honours in a player's award feed, best first.

    Everything not on the allowlist is dropped, which is most of it.
    """
    found: dict[str, Honour] = {}
    for award in awards or []:
        honour = _BY_NAME.get((award.get("name") or "").strip())
        if honour is None:
            continue
        season = str(award.get("season") or "").strip()
        entry = found.setdefault(honour, Honour(name=honour))
        if season and season not in entry.seasons:
            entry.seasons.append(season)

    for entry in found.values():
        entry.seasons.sort()
    return sorted(found.values(), key=lambda h: (_RANK[h.name], -h.count))


def summarize(career: Career, *, limit: int = 3) -> str:
    """The honours line, or empty when there is nothing worth stating.

    Empty is the normal answer. Most players on a card have never won any of
    these, and "no major awards" is not a fact about a player worth printing --
    it describes the large majority of major leaguers.
    """
    if not career.honours:
        return ""
    parts = []
    for honour in career.honours[:limit]:
        if honour.count > 1:
            parts.append(f"{honour.count}x {honour.name}")
        else:
            only = f" ({honour.seasons[0]})" if honour.seasons else ""
            parts.append(f"{honour.name}{only}")
    return ", ".join(parts)


def build(box: Any) -> Career:
    """A player's career block from what the preview already carries."""
    totals = getattr(box, "career", None) or {}
    return Career(
        honours=honours_from(getattr(box, "awards", None)),
        games=int(totals.get("gamesPlayed") or 0),
        totals=totals,
    )
