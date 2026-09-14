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
    (45.0, "mvp", (
        "an MVP-caliber player", "one of the two or three best in the league",
        "as good as this gets", "the best player on most fields he walks onto",
        "in the MVP conversation", "carrying a franchise",
    )),
    (33.0, "superstar", (
        "a genuine superstar", "the kind of player a team is built around",
        "one of the game's best", "a star by any measure",
        "one of the best players in the league", "a headliner",
    )),
    (24.0, "allstar", (
        "an All-Star", "a difference-maker", "a player who decides games",
        "one of the better players in the game", "a legitimate star",
        "the best player in most lineups",
    )),
    (17.0, "verygood", (
        "a very good player", "well above an everyday standard",
        "a clear plus regular", "the sort of player contenders want",
        "good enough to build around", "a genuine asset",
    )),
    (11.0, "solid", (
        "a solid regular", "a quality everyday player",
        "comfortably above average", "a player who helps a team win",
        "a dependable starter", "better than most who play his position",
    )),
    (5.0, "aboveaverage", (
        "an above-average regular", "a little better than average",
        "an honest regular with something extra", "a useful everyday player",
        "a shade above the everyday line", "worth a lineup spot on merit",
    )),
    (-2.0, "average", (
        "an average regular", "about what an everyday player should be",
        "right at the everyday standard", "a league-average bat and body",
        "neither a strength nor a hole", "the middle of the distribution",
    )),
    (-8.0, "fringe", (
        "a fringe regular", "playable but stretched as an everyday piece",
        "a second-division starter", "better suited to a smaller role",
        "hanging on to a lineup spot", "more of a platoon piece",
    )),
    (-16.0, "bench", (
        "a bench player", "a reserve rather than a regular",
        "someone a good team carries rather than starts",
        "a depth piece", "playing above his level as a starter",
        "a bench bat pressed into service",
    )),
    (-9e9, "replacement", (
        "a replacement-level player", "the sort of player freely available",
        "costing his team runs every time he plays",
        "below what a call-up would give", "a genuine hole",
        "not an everyday player at this level",
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


# How a season verdict is framed. Every one of these scopes the claim to this
# year, which is the whole point of them -- but a card carries twenty-six
# players and one fixed frame would open twenty-six paragraphs identically,
# which reads as a template however varied the verdict inside it is.
#
# All are perfect tense or explicitly dated. None can be read as a claim about
# the player rather than the season.
# Every frame is built on "has been", and that is a grammatical constraint
# rather than a lack of imagination.
#
# The tier that substitutes in comes from VALUE_BANDS, which holds three
# different kinds of phrase: noun ("a serious bat"), participial ("carrying a
# franchise") and prepositional ("in the MVP conversation"). "Has been" is a
# copula and takes all three. Almost nothing else does -- tried and discarded:
# "has played like carrying a franchise", "has graded out as in the MVP
# conversation", "has given his team in the MVP conversation".
#
# Variety therefore comes from where the time marker sits, not from the verb.
# Adding a frame with a different verb means first tagging all sixty tier
# phrasings by kind, which is a larger change than it looks.
SEASON_FRAMES = (
    "{name} has been {tier} this season",
    "This season {name} has been {tier}",
    "On the year {name} has been {tier}",
    "Through this season {name} has been {tier}",
    "So far this season {name} has been {tier}",
    "{name} has been {tier} through this year",
)


def season_frame(name: str, tier: str, *key: Any,
                 avoid: Sequence[str] | None = None) -> str:
    """One season-scoped opening for a profile."""
    template = choose(SEASON_FRAMES, *key, avoid=avoid)
    return template.format(name=name, tier=tier)


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

    def frame(self, name: str, tier: str, *key: Any) -> str:
        template = choose(SEASON_FRAMES, *key, avoid=self._recent.get("frame", []))
        self._remember("frame", template)
        return template.format(name=name, tier=tier)

    def team(self, code: str, *key: Any, **slots: Any) -> str:
        phrase = team_phrase(
            code, *key, avoid=self._recent.get("team", []), **slots)
        if phrase:
            self._remember("team", phrase)
        return phrase

    def matchup(self, code: str, *key: Any, **slots: Any) -> str:
        phrase = matchup_phrase(
            code, *key, avoid=self._recent.get("matchup", []), **slots)
        if phrase:
            self._remember("matchup", phrase)
        return phrase

    def form(self, direction: str, *key: Any) -> str:
        phrase = form_phrase(
            direction, *key, avoid=self._recent.get("form", []))
        self._remember("form", phrase)
        return phrase


# --------------------------------------------------------------------------
# Matchup phrasings
# --------------------------------------------------------------------------
# The same discipline as the value and tool bands: several ways to say one
# thing, never several things. Each entry is a format string over the same
# slots, so swapping one for another changes the wording and nothing else.
#
# `{p}` is the pitcher's surname, `{b}` the batter's, `{hand}` an optional
# handedness clause. A phrasing that used a slot the caller does not supply
# would raise at render time, so every variant in a set takes the same slots.

MATCHUP_PHRASES: dict[str, tuple[str, ...]] = {
    # -- batter against a starter ---------------------------------------
    "bat.overmatched": (
        "{p} misses bats and {b} does not make much contact, which is the worst "
        "version of this for him",
        "{b} swings through pitches and {p} is the man to make him pay for it",
        "this is the bad one for {b}: a swing-and-miss arm against a hitter who "
        "already misses",
    ),
    "bat.contact_vs_stuff": (
        "{p}'s swing-and-miss against one of the harder men in the league to "
        "strike out",
        "{p} gets his strikeouts, but {b} is a difficult place to look for one",
        "the strikeout is {p}'s business and {b} does not give many away",
    ),
    "bat.contact_vs_soft": (
        "a contact hitter against a pitcher who does not miss bats, so the ball "
        "is going to be in play",
        "neither man is trying to avoid contact here; expect the ball put in "
        "play",
        "{b} makes contact and {p} allows it, which puts this on the defense",
    ),
    "bat.both_weak": (
        "{b} swings through a lot, but {p} is not the man to punish it",
        "a hitter who misses against a pitcher who cannot make him",
        "{b}'s swing-and-miss goes unpunished by an arm that does not chase it",
    ),
    "bat.grounder_vs_grounder": (
        "{p} keeps it on the ground and {b} already hits it there",
        "a sinkerballer against a hitter who beats it into the dirt anyway",
        "both men are working toward the same ground ball",
    ),
    "bat.grounder_vs_loft": (
        "{b} wants it in the air and {p} will not let him have it",
        "{p} keeps the ball down, which is the one place {b} cannot use it",
        "a hitter who lifts against a pitcher who does not let him",
    ),
    "bat.air_vs_loft": (
        "{p} lets the ball get airborne, which is exactly where {b} wants it",
        "{b} hits it in the air and {p} gives that up",
        "the ball gets up against {p}, and up is where {b} does his damage",
    ),
    "bat.power_vs_soft": (
        "{b}'s power against a pitcher who gives up hard contact is the danger "
        "here",
        "{p} has been squared up all year and {b} is the man to do it",
        "hard contact is available against {p}, and {b} is who takes it",
    ),
    "bat.power_vs_suppress": (
        "{p} has kept the barrel off the ball all year, which is the one thing "
        "{b} needs",
        "{b} needs to square one up and {p} has not allowed many",
        "power against a pitcher who has not given any up",
    ),
    "bat.zone_vs_chase": (
        "{p} pounds the zone and {b} chases, so the free pass is unlikely to "
        "arrive",
        "{b} will not be walked here; {p} is around the plate too often",
        "a chaser against a strike-thrower, which is no way to reach first for "
        "free",
    ),
    "bat.wild_vs_patient": (
        "{p} is around the zone less than most and {b} will make him prove it",
        "{b} takes his walks and {p} hands them out",
        "patience against a pitcher who struggles to find the plate",
    ),
    "bat.both_wild": (
        "neither man is disciplined here — {p} misses the zone and {b} swings "
        "at it anyway",
        "{p} cannot find the plate and {b} will not make him",
        "a wild arm against a free swinger, which usually resolves itself",
    ),

    # -- pitcher against a lineup ---------------------------------------
    "pit.stuff_vs_weak_contact": (
        "{p} misses bats and this lineup does not make much contact{hand}",
        "a swing-and-miss arm against a lineup that swings and misses{hand}",
        "this is a good place for {p} to look for strikeouts{hand}",
    ),
    "pit.soft_vs_contact": (
        "a contact lineup against a pitcher who does not miss bats — the ball "
        "is going to be in play all night",
        "neither side is avoiding contact; this one goes to the defense",
        "{p} does not miss bats and this lineup does not miss pitches",
    ),
    "pit.stuff_vs_contact": (
        "{p}'s swing-and-miss against a lineup that puts the bat on it",
        "a strikeout arm against nine men who are hard to strike out",
        "{p} will have to work harder than usual for his whiffs",
    ),
    "pit.grounder_vs_loft": (
        "they want the ball in the air and {p} keeps it down",
        "a lineup built to lift against a pitcher who will not let it",
        "{p}'s ground game against a lineup trying to get underneath it",
    ),
    "pit.air_vs_loft": (
        "a lineup that elevates against a pitcher who lets the ball get "
        "airborne",
        "they hit it in the air and {p} gives that up, which is the risk here",
        "{p} lets the ball get up, and this lineup is built to use that",
    ),
    "pit.power_vs_soft": (
        "this lineup has real power and {p} has been hit hard all year",
        "{p} gives up loud contact to a lineup that can punish it",
        "the barrels are there for the taking against {p}",
    ),
    "pit.weak_power_vs_suppress": (
        "little power here to trouble a pitcher who keeps the barrel off the "
        "ball",
        "{p} suppresses hard contact and this lineup does not generate much",
        "neither the lineup nor {p} is likely to produce much loud contact",
    ),
    "pit.wild_vs_patient": (
        "a patient lineup against a pitcher who has trouble finding the zone, "
        "which is how short outings start",
        "{p} misses the zone and this lineup is content to watch him do it",
        "walks are the risk: a patient lineup against a pitcher without command",
    ),
    "pit.zone_vs_chase": (
        "{p} throws strikes and they chase, so the walks are unlikely to come",
        "a strike-thrower against a lineup that expands, which is a quick night",
        "{p} is around the plate and this lineup does not make pitchers work",
    ),
}


def matchup_phrase(code: str, *key: Any, avoid: Sequence[str] | None = None,
                   **slots: Any) -> str:
    """One phrasing of a matchup, filled in and stable for this pairing."""
    options = MATCHUP_PHRASES.get(code)
    if not options:
        return ""
    chosen = choose(options, code, *key, avoid=avoid)
    try:
        return chosen.format(**slots)
    except (KeyError, IndexError):
        return ""


# --------------------------------------------------------------------------
# Team-level phrasings
# --------------------------------------------------------------------------
# The matchup note is written once a day for the same two clubs through a
# three-game series, so it repeats harder than any player block does. Same rule
# as everywhere else: several ways to say one thing, never several things.
#
# Slots: {a} and {h} are the two club abbreviations, {fav} the better side,
# {dog} the other, and the numeric slots are pre-formatted by the caller.

TEAM_PHRASES: dict[str, tuple[str, ...]] = {
    # -- who is better on paper -----------------------------------------
    "rec.gap": (
        "{fav} are the better team on record, {favrec} against {dogrec}",
        "the standings are not close: {favrec} for {fav}, {dogrec} for {dog}",
        "{favrec} against {dogrec} — {fav} have been the better side all year",
    ),
    "rec.close": (
        "these two are near enough level on record, {favrec} against {dogrec}",
        "little between them in the standings — {favrec} and {dogrec}",
        "{favrec} against {dogrec}, which is close to a coin flip on paper",
    ),
    "rundiff.gap": (
        "{fav} have outscored their opposition by {favdiff} on the year against "
        "{dogdiff} for {dog}",
        "the run differential says the same thing: {favdiff} for {fav}, "
        "{dogdiff} for {dog}",
        "{fav} sit at {favdiff} on run differential, {dog} at {dogdiff}",
    ),
    "luck.flattered": (
        "{team}'s record flatters them — {luck} wins above what their scoring "
        "implies",
        "{team} have won {luck} more than their runs deserve, which tends not "
        "to hold",
        "{luck} of {team}'s wins are not supported by their run scoring",
    ),
    "luck.unlucky": (
        "{team} have been unlucky, {luck} wins short of what their scoring "
        "implies",
        "{team}'s record understates them by {luck} wins",
        "the runs say {team} should have {luck} more wins than they do",
    ),
    "form.gap": (
        "recent form points the same way: {fav} are {favten} in their last ten, "
        "{dog} {dogten}",
        "{fav} have gone {favten} over ten games to {dog}'s {dogten}",
        "over the last ten it is {favten} for {fav} against {dogten} for {dog}",
    ),
    "form.against": (
        "recent form cuts against that — {hot} are {hotten} in their last ten "
        "while {cold} have gone {coldten}",
        "the last ten flip it: {hotten} for {hot}, {coldten} for {cold}",
        "{hot} are the hotter side right now at {hotten} to {coldten}",
    ),
    "streak": (
        "{team} arrive on {streak}",
        "{team} come in having {streakverb}",
        "{team} are riding {streak}",
    ),
    "series.led": (
        "{leader} lead this series {lead}",
        "{leader} are up {lead} through {played}",
        "{leader} took the opener and lead {lead}",
    ),
    "series.level": (
        "the series is level at {lead}",
        "one apiece so far",
        "nothing between them in the series at {lead}",
    ),
    "series.opener": (
        "this is the opener",
        "first of {total}",
        "game one of {total}",
    ),
    "elo.gap": (
        "the ratings agree, and by more than the records do",
        "the model's own team rating separates them further than the standings",
        "on rating rather than record the gap is wider still",
    ),
    "elo.narrow": (
        "the ratings see them closer than the records do",
        "the model's team rating is less impressed by the gap than the "
        "standings are",
        "on rating the two are nearer than the win column suggests",
    ),

    # -- the pitching matchup -------------------------------------------
    "sp.mismatch": (
        "{better} has been the better pitcher this year by a clear margin",
        "this is a mismatch on the mound in {better}'s favour",
        "{better} is the more accomplished of the two starters by some way",
    ),
    "sp.even": (
        "the two starters are hard to separate on the season",
        "little between the starters on their year's work",
        "a fairly even matchup on the mound",
    ),
    "sp.contrast": (
        "two different kinds of pitcher: {a_desc} against {h_desc}",
        "a contrast in style — {a_desc} opposite {h_desc}",
        "{a_desc} and {h_desc}, which is as different as two starters get",
    ),
    "sp.form": (
        "{who} has been the sharper of the two lately",
        "recent work favours {who}",
        "{who} is the one arriving in better touch",
    ),
    "bullpen.gap": (
        "the bullpens are not equal either — {better} carry the deeper one",
        "{better} have the better relief corps behind their starter",
        "past the starters, {better} hold the advantage",
    ),
    "bullpen.tired": (
        "{team}'s pen has been worked hard, {pitches} pitches across three days",
        "{team} come in with a tired bullpen — {pitches} pitches in three days",
        "{pitches} pitches over three days leaves {team}'s relief thin",
    ),
}

def team_phrase(code: str, *key, avoid=None, **slots) -> str:
    """One phrasing of a team-level observation, filled in and stable."""
    options = TEAM_PHRASES.get(code)
    if not options:
        return ""
    chosen = choose(options, code, *key, avoid=avoid)
    try:
        return chosen.format(**slots)
    except (KeyError, IndexError):
        return ""
