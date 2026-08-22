"""The vocabulary the whole analysis speaks in.

Fifty-two players each described as "above average" or "below average" is
technically correct and reads like a form letter, and a reader stops seeing the
words after the fourth box. This module holds every phrasing the report can
reach for, so the same finding can be said many ways and a page of players reads
like a page of players.

Three rules keep variety from becoming imprecision.

**Variety is in the wording, never in the claim.** Every phrasing inside one
band means the same thing. "Replacement level" and "the roster's last man" are
interchangeable; "flawed" and "replacement level" are not, and never sit in the
same bucket. If two phrasings would leave a reader with different beliefs, they
belong to different bands.

**A player reads the same way every time.** The choice is a stable hash of the
subject and the thing being described, not a random draw and not a rotation. So
one player's page is consistent with itself across sections, and consistent
between builds -- the same game yields the same words forever, which is half the
argument for computing this at all. The builtin `hash` is unusable here: Python
salts string hashing per process, and this package already shipped that bug once.

**Neighbouring players differ.** A stable hash alone will still hand the same
adjective to two men in a row often enough to notice, so the caller can pass
what it has already used and the picker steps to its next-best option instead.
That is the difference between variety that is designed and variety that is
merely possible.
"""

from __future__ import annotations

import zlib
from typing import Any, Sequence

# --------------------------------------------------------------------------
# Value bands
# --------------------------------------------------------------------------
# Keyed by runs above average per 150 games. The bands are the claim; the lists
# are only how it can be said. Every entry completes "he is ..." so the caller
# never has to know which article a phrase wants.

VALUE_BANDS: tuple[tuple[float, str, tuple[str, ...]], ...] = (
    (38.0, "superstar", (
        "an MVP-caliber player", "one of the best players in the league",
        "a genuine superstar", "the kind of player a team is built around",
        "a perennial All-Star", "as good as this gets",
    )),
    (25.0, "excellent", (
        "an excellent player", "a star", "one of the better players in the game",
        "an All-Star-caliber player", "a difference-maker",
        "a player who decides games",
    )),
    (12.0, "good", (
        "a good player", "a solid regular", "a quality everyday player",
        "an above-average regular", "a player who helps a team win",
        "comfortably above average",
    )),
    (2.0, "useful", (
        "a useful regular", "a serviceable everyday player",
        "a little better than average", "a player who holds his own",
        "an honest regular", "average with a bit left over",
    )),
    (-2.0, "average", (
        "a roughly average player", "an average big-leaguer",
        "the definition of a league-average player", "squarely average",
        "the middle of the league", "neither an asset nor a problem",
    )),
    (-10.0, "limited", (
        "a limited player", "a flawed player", "a second-division regular",
        "a player with real holes", "below average and playing anyway",
        "a stopgap",
    )),
    (-20.0, "poor", (
        "a poor player", "a bad big-leaguer", "a liability most nights",
        "someone a contender would not start", "well below average",
        "a weak link",
    )),
    (float("-inf"), "replacement", (
        "a replacement-level player", "the roster's last man",
        "freely available talent", "a player on borrowed time",
        "replacement level in every sense", "the bottom of a major-league roster",
    )),
)


# Tool grades, by percentile.
#
# Adjectives only, and that is a grammatical constraint rather than a stylistic
# one. These fill the slot in "<phrase> command (82nd percentile)", so a noun
# phrase produces "a carrying tool command" and "as bad as it gets ground game"
# -- which is how the first version of this read. Every entry below has been
# checked to complete that sentence.
TOOL_BANDS: tuple[tuple[float, str, tuple[str, ...]], ...] = (
    (92.0, "elite", ("elite", "outstanding", "exceptional", "extraordinary",
                     "top-tier", "premier")),
    (80.0, "plus-plus", ("excellent", "superb", "formidable", "high-end",
                         "well above-average", "standout")),
    (65.0, "plus", ("plus", "above-average", "solid", "strong", "reliable",
                    "dependable")),
    (45.0, "average", ("average", "fair", "unremarkable", "ordinary",
                       "middling", "workmanlike")),
    (30.0, "fringe", ("fringy", "shaky", "underwhelming", "suspect", "modest",
                      "below-average")),
    (15.0, "poor", ("poor", "weak", "deficient", "substandard", "lacking",
                    "problematic")),
    (float("-inf"), "bottom", ("unplayable", "dismal", "abysmal", "wretched",
                               "bottom-of-the-scale", "nonexistent")),
)


# How to open the form section, by direction.
FORM_OPENERS = {
    "up": ("he is trending up", "he has been swinging it lately",
           "he has picked it up", "he is going well", "he is heating up",
           "the last few weeks have been his best"),
    "down": ("he is trending down", "he has gone cold", "he is in a rough patch",
             "he has slipped", "he is scuffling", "the last few weeks have been lean"),
    "flat": ("the last fifteen games look like the rest of his season",
             "nothing has changed lately", "he is doing what he always does",
             "recent form matches the season", "no real movement either way",
             "steady as he has been all year"),
}


def _index(*key: Any) -> int:
    """A stable choice, identical in every process and every build.

    Not the builtin `hash`: Python salts string hashing per process, so the same
    player drew a different adjective on every build. That bug already shipped
    once in this package and was invisible from inside a single run.
    """
    return zlib.crc32(":".join(str(part) for part in key).encode())


def choose(
    options: Sequence[str], *key: Any, avoid: Sequence[str] | None = None
) -> str:
    """One phrasing from a set, stable for this subject and unlike its neighbours.

    `avoid` carries what has already been said nearby. A stable hash on its own
    still repeats an adjective for two consecutive players often enough to
    notice, and a page where three men in a row are "solid" reads as a template
    however varied the vocabulary behind it is.
    """
    if not options:
        return ""
    start = _index(*key) % len(options)
    blocked = set(avoid or ())
    for step in range(len(options)):
        candidate = options[(start + step) % len(options)]
        if candidate not in blocked:
            return candidate
    return options[start]


def _band(bands, value: float):
    for cut, name, phrasings in bands:
        if value >= cut:
            return name, phrasings
    return bands[-1][1], bands[-1][2]


def value_phrase(
    runs_per_150: float, *key: Any, avoid: Sequence[str] | None = None
) -> tuple[str, str]:
    """How to say what a player is worth. Returns (phrase, band name)."""
    name, phrasings = _band(VALUE_BANDS, float(runs_per_150))
    return choose(phrasings, name, *key, avoid=avoid), name


def tool_phrase(
    grade: float, *key: Any, avoid: Sequence[str] | None = None
) -> tuple[str, str]:
    """How to say how good a tool is. Returns (phrase, band name)."""
    name, phrasings = _band(TOOL_BANDS, float(grade))
    return choose(phrasings, name, *key, avoid=avoid), name


def form_phrase(
    direction: str, *key: Any, avoid: Sequence[str] | None = None
) -> str:
    return choose(FORM_OPENERS.get(direction, FORM_OPENERS["flat"]),
                  direction, *key, avoid=avoid)


class Voice:
    """Tracks what has been said on this page, so the next player differs.

    Held for the length of one card. The memory is deliberately short -- the
    last few phrasings, not all of them -- because avoiding everything already
    used would exhaust the vocabulary and force the picker back to whatever it
    started with, which is the repetition it was built to prevent.
    """

    def __init__(self, memory: int = 3) -> None:
        self._memory = memory
        self._recent: dict[str, list[str]] = {}

    def _remember(self, kind: str, phrase: str) -> None:
        seen = self._recent.setdefault(kind, [])
        seen.append(phrase)
        del seen[:-self._memory]

    def value(self, runs_per_150: float, *key: Any) -> str:
        phrase, _ = value_phrase(
            runs_per_150, *key, avoid=self._recent.get("value", []))
        self._remember("value", phrase)
        return phrase

    def tool(self, grade: float, *key: Any) -> str:
        phrase, _ = tool_phrase(
            grade, *key, avoid=self._recent.get("tool", []))
        self._remember("tool", phrase)
        return phrase

    def blurb(self, options: Sequence[str], *key: Any) -> str:
        phrase = choose(options, *key, avoid=self._recent.get("blurb", []))
        self._remember("blurb", phrase)
        return phrase

    def form(self, direction: str, *key: Any) -> str:
        phrase = form_phrase(
            direction, *key, avoid=self._recent.get("form", []))
        self._remember("form", phrase)
        return phrase
