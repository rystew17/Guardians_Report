"""What kind of player this is, and whether he is any good.

The evaluators in this package answer "what is unusual about him". This module
answers the two questions a reader actually opens a scouting report with -- how
good is he, and what kind of player is he -- which no amount of correct isolated
facts adds up to. ".324 expected against the curveball" is true and inert; "a
command artist who lives in the zone and will not beat himself" is the same
pitcher described in a way you can act on.

**The axes are measured, not assumed.** Principal components on 3,449 qualified
batter-seasons give four axes carrying 84% of the variance: power (39%),
aggression (21%), loft (15%) and contact quality (9%). Three of the four are
strong traits -- a player's position on them correlates 0.80-0.83 with his own
next season -- so they describe the player rather than the year. Pitchers give
five axes carrying 79%, with stuff the most stable at 0.82.

Loft is genuinely separate from power. That was not obvious and it is why "hits
it in the air without the exit velocity to support it" is a real profile rather
than a slight on a slugger.

**The grades are composites, not component scores.** The PCA establishes how
many real dimensions exist; it makes a poor vocabulary, because a reader needs
"power" to mean barrels and home runs rather than a loading vector that shifts
when the population does. Each tool is the mean z-score of the measures that
define it, graded against the current season and rebuilt every year.

**Rate, not accumulation.** Whether a player is good is a question about quality,
so every grade and the tier itself are rates -- runs above average per 150 games,
steals per time on first, outs above average per 150. Six outs above average in
forty games is elite, and a counting stat would call it ordinary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# League constants, measured from the pitch corpus (2015-2026, 2.03m plate
# appearances). Recomputed by `scripts/measure_profile_constants.py`.
# --------------------------------------------------------------------------

LEAGUE_XWOBA = 0.3170
WOBA_SCALE = 1.20

# Runs per stolen base and per time caught, from run expectancy. Catching
# costs roughly twice what stealing gains, which is why a low-volume thief with
# a poor success rate grades below a player who never runs at all.
RUN_SB = 0.200
RUN_CS = -0.400

# League stolen-base rate per time on first, for centring the baserunning term.
LEAGUE_SB_RATE = 0.048

# Runs per 150 games, relative to an average player at that position. Outs
# above average is already position-relative -- it compares a shortstop with
# other shortstops -- so without this a good defensive first baseman and a good
# defensive catcher grade identically, and they are not worth the same.
POSITION_ADJUSTMENT = {
    "C": 11.6, "SS": 6.9, "2B": 2.3, "3B": 2.3, "CF": 2.3,
    "LF": -6.9, "RF": -6.9, "1B": -11.6, "DH": -16.2,
}

# Percentile bands the profile rules are written in.
ELITE, HIGH, MID_HI, MID_LO, LOW, POOR = 85, 65, 55, 45, 35, 15


@dataclass
class Tool:
    """One graded ability, with the measures that produced it."""

    name: str
    label: str
    grade: float                      # 0-100 percentile in the reference season
    z: float
    support: dict[str, float] = field(default_factory=dict)
    # Signed z per measure. Without this the note quotes a fixed measure for
    # each tool and can say "elite contact quality, 89.1 mph" for one player
    # and "poor, 87.2 mph" for another -- both true, jointly incoherent,
    # because the grade was driven by line-drive rate and the sentence was not.
    contributions: dict[str, float] = field(default_factory=dict)

    @property
    def driver(self) -> str:
        """The measure that earned this grade, for quoting as evidence."""
        if not self.contributions:
            return ""
        return max(self.contributions.items(), key=lambda kv: abs(kv[1]))[0]

    @property
    def band(self) -> str:
        for cut, word in (
            (ELITE, "elite"), (HIGH, "plus"), (MID_LO, "average"),
            (POOR, "below average"),
        ):
            if self.grade >= cut:
                return word
        return "poor"


@dataclass
class Match:
    """A profile this player fits, and how well."""

    code: str
    label: str
    # Every phrasing of this archetype. A tuple rather than a string because two
    # Soft-Contact Managers on one card otherwise read word for word alike.
    blurb: tuple[str, ...] | str
    strength: float                   # 0-1, how squarely he sits in it


@dataclass
class PlayerProfile:
    """Everything the written profile needs, already decided."""

    player_id: int
    kind: str                         # batter | pitcher
    tools: dict[str, Tool] = field(default_factory=dict)
    matches: list[Match] = field(default_factory=list)
    runs_per_150: float | None = None
    tier: str = ""
    tier_grade: float | None = None
    sample: int = 0
    thin: bool = False

    def tool(self, name: str) -> Tool | None:
        return self.tools.get(name)

    @property
    def carrying(self) -> list[Tool]:
        """What he is actually good at, best first."""
        return sorted(
            (t for t in self.tools.values() if t.grade >= HIGH),
            key=lambda t: -t.grade,
        )

    @property
    def weaknesses(self) -> list[Tool]:
        return sorted(
            (t for t in self.tools.values() if t.grade <= LOW),
            key=lambda t: t.grade,
        )


# --------------------------------------------------------------------------
# Tool definitions
# --------------------------------------------------------------------------
# Each tool is the mean of its measures' z-scores, with a sign so that positive
# always means better. Signs are the whole content here: `k_rate` and
# `chase_rate` are negated because a high one is a weakness, and a tool that
# silently averaged a strength with an unnegated weakness would grade an
# undisciplined slugger as balanced.

BATTER_TOOLS: dict[str, tuple[str, tuple[tuple[str, int], ...]]] = {
    "power":      ("power",        (("barrel_rate", 1), ("hr_rate", 1), ("ev", 1))),
    "contact":    ("bat-to-ball",  (("whiff_per_swing", -1), ("k_rate", -1))),
    "discipline": ("plate discipline", (("bb_rate", 1), ("chase_rate", -1))),
    "loft":       ("launch",       (("la", 1), ("gb_rate", -1))),
    "hard":       ("contact quality", (("ev", 1), ("ld_rate", 1))),
}

PITCHER_TOOLS: dict[str, tuple[str, tuple[tuple[str, int], ...]]] = {
    "stuff":      ("stuff",        (("k_rate", 1), ("whiff_per_swing", 1))),
    "command":    ("command",      (("bb_rate", -1), ("zone_rate", 1))),
    "grounders":  ("ground game",  (("gb_rate", 1), ("fb_rate", -1))),
    "suppress":   ("contact suppression", (("barrel_allowed", -1), ("ev_allowed", -1))),
    "arsenal":    ("arsenal",      (("mix", 1), ("primary_share", -1))),
}


def _percentile(value: float, population: np.ndarray) -> float:
    """Where a value sits in its reference population, 0-100."""
    population = population[np.isfinite(population)]
    if not len(population) or not np.isfinite(value):
        return float("nan")
    return float((population < value).mean() * 100.0)


def grade_tools(
    row: dict[str, float],
    reference,
    definitions: dict[str, tuple[str, tuple[tuple[str, int], ...]]],
) -> dict[str, Tool]:
    """Grade every tool for one player against a reference population.

    `reference` is a frame of the same measures for everyone who qualified this
    season. Grading against the current season rather than a stored constant is
    deliberate: the strikeout rate that was average in 2015 is well below
    average now, and a fixed threshold would quietly re-rank the league every
    year it was left alone.
    """
    tools: dict[str, Tool] = {}
    for name, (label, measures) in definitions.items():
        zs, support, contributions = [], {}, {}
        for measure, sign in measures:
            if measure not in reference.columns:
                continue
            column = reference[measure].to_numpy(dtype=float)
            value = row.get(measure)
            if value is None or not np.isfinite(value):
                continue
            mean, sd = np.nanmean(column), np.nanstd(column)
            if not sd:
                continue
            contribution = sign * (value - mean) / sd
            zs.append(contribution)
            support[measure] = float(value)
            contributions[measure] = float(contribution)
        if not zs:
            continue
        z = float(np.mean(zs))
        # The composite's own reference, so the grade is a percentile among
        # players rather than a normal-curve assumption about the composite.
        composite = _composite_population(reference, measures)
        tools[name] = Tool(
            name=name, label=label, z=z,
            grade=_percentile(z, composite), support=support,
            contributions=contributions,
        )
    return tools


def _composite_population(reference, measures) -> np.ndarray:
    """The same composite computed for everyone, to grade one player against."""
    parts = []
    for measure, sign in measures:
        if measure not in reference.columns:
            continue
        column = reference[measure].to_numpy(dtype=float)
        mean, sd = np.nanmean(column), np.nanstd(column)
        if sd:
            parts.append(sign * (column - mean) / sd)
    return np.nanmean(np.vstack(parts), axis=0) if parts else np.array([])


# --------------------------------------------------------------------------
# Value — is he a good player
# --------------------------------------------------------------------------

def batting_runs(xwoba: float, plate_appearances: int) -> float:
    """Runs above average from hitting.

    The standard wRAA form on expected rather than actual wOBA, so a hitter is
    graded on the contact he made rather than on where it happened to land.
    """
    if xwoba is None or not np.isfinite(xwoba):
        return 0.0
    return float((xwoba - LEAGUE_XWOBA) / WOBA_SCALE * plate_appearances)


def baserunning_runs(season: dict, measured: float | None = None) -> float:
    """Runs above average on the bases.

    Prefers Savant's own figure, which is the sum of what a runner gained
    taking extra bases on batted balls and what he gained stealing -- the
    Baserunning Run Value on his player page. That is a measurement.

    The fallback below is an estimate from stolen bases and times caught, used
    only when the board has no row for him. It is worse in a specific way: it
    cannot see the half of baserunning that happens without a throw, so a
    runner who never steals but goes first to third all year reads as neutral.
    """
    if measured is not None and np.isfinite(measured):
        return float(measured)

    singles = float(season.get("singles") or 0)
    walks = float(season.get("baseOnBalls") or 0)
    hbp = float(season.get("hitByPitch") or 0)
    on_first = singles + walks + hbp
    if on_first <= 0:
        return 0.0

    sb = float(season.get("stolenBases") or 0)
    cs = float(season.get("caughtStealing") or 0)
    gained = RUN_SB * sb + RUN_CS * cs
    # Centered on what an average player would have produced with the same
    # number of chances, so this is runs *above average* like the other terms.
    expected = RUN_SB * LEAGUE_SB_RATE * on_first
    return float(gained - expected)


def fielding_runs(box: Any, games: int) -> float:
    """Runs above average in the field, including the positional adjustment.

    Outs above average is position-relative -- it compares a shortstop with
    other shortstops -- so a good defensive first baseman and a good defensive
    catcher carry the same OAA and are not worth the same. The adjustment is
    what makes a glove-first catcher legible as valuable rather than as an
    average fielder who cannot hit.
    """
    prevented = (getattr(box, "fielding", {}) or {}).get("fielding_runs_prevented")
    runs = float(prevented) if prevented is not None else 0.0
    position = (getattr(box, "position", "") or "").upper()
    adjustment = POSITION_ADJUSTMENT.get(position, 0.0) * (games / 150.0)
    return runs + adjustment


TIERS = (
    (25.0, "an excellent player"),
    (12.0, "a good player"),
    (-2.0, "a roughly average player"),
    (-14.0, "a below-average player"),
)


def tier_for(runs_per_150: float | None) -> str:
    if runs_per_150 is None or not np.isfinite(runs_per_150):
        return ""
    for cut, label in TIERS:
        if runs_per_150 >= cut:
            return label
    return "a replacement-level player"


# --------------------------------------------------------------------------
# The profiles
# --------------------------------------------------------------------------
# Each is a rule over graded tools plus, where the archetype needs it, a raw
# rate. A player may satisfy several and that is not a defect: a slugger who
# also walks is genuinely both a classic slugger and a three-outcomes hitter,
# and forcing a single label would throw away the half that made him unusual.
#
# `strength` is how squarely he sits inside the rule, so a player who clears
# every clause comfortably leads the ones who scrape in.

@dataclass(frozen=True)
class Definition:
    code: str
    label: str
    blurb: str
    kind: str
    rule: Any                          # (tools, extra) -> float | None


def _g(tools: dict[str, Tool], name: str) -> float:
    tool = tools.get(name)
    return tool.grade if tool and np.isfinite(tool.grade) else float("nan")


def _clears(*conditions: tuple[float, float, bool]) -> float | None:
    """Average margin by which a set of clauses is satisfied, or None.

    Each clause is (grade, threshold, above). Returning the margin rather than
    a boolean is what lets two players who both match be ranked, instead of
    printing whichever the tuple happened to list first.
    """
    margins = []
    for grade, threshold, above in conditions:
        if not np.isfinite(grade):
            return None
        margin = (grade - threshold) if above else (threshold - grade)
        if margin < 0:
            return None
        margins.append(min(margin / 35.0, 1.0))
    return float(np.mean(margins)) if margins else None


def _has_plus(tools: dict[str, Tool]) -> bool:
    """Whether anything about this player plays above average.

    Used to disqualify the two profiles that assert an absence. Without it a
    player can be told he has nothing that plays above average in the same
    sentence that credits him with plus baserunning, which is two true grades
    and one false claim.
    """
    return any(
        np.isfinite(tool.grade) and tool.grade >= HIGH for tool in tools.values()
    )


BATTER_PROFILES: tuple[Definition, ...] = (
    Definition(
        "tto", "Three True Outcomes",
        (
            "walks, strikes out and does damage -- the ball rarely goes in play",
            "three outcomes and not much else, which is a living",
            "he will walk, whiff or hit it out, and rarely anything between",
        ),
        "batter",
        lambda t, x: _clears((_g(t, "power"), HIGH, True),
                             (_g(t, "discipline"), MID_HI, True),
                             (_g(t, "contact"), LOW, False)),
    ),
    Definition(
        "swing_miss_slug", "Swing-and-Miss Slugger",
        (
            "enormous damage when he connects, and he does not connect often",
            "all or nothing, and the all is worth waiting for",
            "he will miss a lot and make one hurt",
        ),
        "batter",
        lambda t, x: _clears((_g(t, "power"), HIGH, True),
                             (_g(t, "contact"), LOW, False),
                             (_g(t, "discipline"), MID_LO, False)),
    ),
    Definition(
        "complete", "Complete Hitter",
        (
            "power without the strikeouts that usually pay for it",
            "damage and contact in the same hitter, which is rare",
            "he hits it hard and he hits it often",
        ),
        "batter",
        lambda t, x: _clears((_g(t, "power"), HIGH, True),
                             (_g(t, "contact"), HIGH, True)),
    ),
    Definition(
        "slugger", "Classic Slugger",
        (
            "power and patience, and pitchers work around him",
            "real thump, and the discipline to wait for something to hit",
            "he does not chase, and he punishes anything in reach",
        ),
        "batter",
        lambda t, x: _clears((_g(t, "power"), HIGH, True),
                             (_g(t, "discipline"), HIGH, True),
                             (_g(t, "contact"), POOR, True)),
    ),
    Definition(
        "on_base", "On-Base Grinder",
        (
            "takes his walks and makes pitchers work, without the power",
            "a professional at-bat every time, if not a loud one",
            "he gets on base without ever threatening the fence",
        ),
        "batter",
        lambda t, x: _clears((_g(t, "discipline"), HIGH, True),
                             (_g(t, "contact"), MID_LO, True),
                             (_g(t, "power"), MID_LO, False)),
    ),
    Definition(
        "bat_to_ball", "Bat-to-Ball Wizard",
        (
            "puts everything in play; striking him out is a project",
            "the bat finds the ball almost every time",
            "getting a swing and miss past him is genuinely hard",
        ),
        "batter",
        lambda t, x: _clears((_g(t, "contact"), ELITE, True),
                             (_g(t, "power"), MID_HI, False)),
    ),
    Definition(
        "line_drive", "Line-Drive Doubles Machine",
        (
            "hits it hard on a line, gap to gap",
            "line drives, doubles, and not much lift",
            "he squares the ball up and lets it find a gap",
        ),
        "batter",
        lambda t, x: _clears((_g(t, "hard"), HIGH, True),
                             (_g(t, "contact"), MID_LO, True),
                             (_g(t, "loft"), ELITE, False)),
    ),
    Definition(
        "free_swinger", "Free Swinger",
        (
            "expands the zone and will chase himself out of an at-bat",
            "he swings, and the strike zone is a suggestion",
            "pitchers do not have to throw him strikes",
        ),
        "batter",
        lambda t, x: _clears((_g(t, "discipline"), POOR, False),),
    ),
    Definition(
        "speed_glove", "Speed and Glove",
        (
            "earns his place with his legs and his glove rather than his bat",
            "the value is in the field and on the bases",
            "he is here for defense and speed, and both play",
        ),
        "batter",
        lambda t, x: _clears((x.get("speed", float("nan")), HIGH, True),
                             (x.get("defense", float("nan")), HIGH, True),
                             (_g(t, "power"), MID_HI, False)),
    ),
    Definition(
        "glove_first", "Glove First",
        (
            "the glove is the whole case, and it is a strong one",
            "he catches everything, and that is enough",
            "the defense carries him, and it carries a long way",
        ),
        "batter",
        lambda t, x: _clears((x.get("defense", float("nan")), ELITE, True),
                             (_g(t, "power"), MID_LO, False)),
    ),
    Definition(
        "basepath", "Basepath Terror",
        (
            "a genuine threat the moment he reaches",
            "once he is on, the pitcher has a second problem",
            "he changes the inning by reaching first",
        ),
        "batter",
        lambda t, x: _clears((x.get("speed", float("nan")), HIGH, True),
                             (x.get("baserunning", float("nan")), HIGH, True)),
    ),
    Definition(
        "five_tool", "Five-Tool Player",
        (
            "hits, hits for power, runs and fields, with nothing to hide",
            "there is no part of this game he does badly",
            "everything plays, which is the rarest profile there is",
        ),
        "batter",
        lambda t, x: _clears((_g(t, "power"), MID_HI, True),
                             (_g(t, "contact"), MID_HI, True),
                             (x.get("speed", float("nan")), MID_HI, True),
                             (x.get("defense", float("nan")), MID_HI, True)),
    ),
    Definition(
        "air_ball", "Air-Ball Chaser",
        (
            "sells out for loft without the exit velocity to reward it",
            "he hits it in the air and it does not go far enough",
            "the launch angle is there and the contact is not",
        ),
        "batter",
        lambda t, x: _clears((_g(t, "loft"), HIGH, True),
                             (_g(t, "power"), LOW, False)),
    ),
    Definition(
        "wasted_contact", "Wasted Contact",
        (
            "hits the ball hard and hits it into the ground",
            "the exit velocity is real and it is going to a shortstop",
            "good contact, wrong angle, and the results follow the angle",
        ),
        "batter",
        lambda t, x: _clears((_g(t, "hard"), MID_HI, True),
                             (_g(t, "loft"), POOR, False)),
    ),
    Definition(
        "fringe", "Fringe Bat",
        (
            "nothing here plays above average, and nothing carries him",
            "no tool stands out, and none of them hides",
            "there is no part of this profile that beats you",
        ),
        "batter",
        lambda t, x: None if _has_plus(t) else _clears(
            (_g(t, "power"), MID_LO, False),
            (_g(t, "contact"), MID_LO, False),
            (_g(t, "discipline"), MID_LO, False)),
    ),
)


PITCHER_PROFILES: tuple[Definition, ...] = (
    Definition(
        "power_arm", "Power Strikeout Arm",
        (
            "misses bats, and does not need the defense behind him",
            "he gets his own outs",
            "swing and miss is the whole plan, and it works",
        ),
        "pitcher",
        lambda t, x: _clears((_g(t, "stuff"), HIGH, True),
                             (_g(t, "command"), POOR, True)),
    ),
    Definition(
        "effectively_wild", "Effectively Wild",
        (
            "overpowering and unpredictable, to both sides",
            "hard to hit and hard to catch, sometimes in the same inning",
            "the stuff is real and so is the walk rate",
        ),
        "pitcher",
        lambda t, x: _clears((_g(t, "stuff"), HIGH, True),
                             (_g(t, "command"), LOW, False)),
    ),
    Definition(
        "command_artist", "Command Artist",
        (
            "lives in the zone and will not beat himself",
            "he throws strikes and makes hitters earn everything",
            "nothing loud, nothing free, and a lot of quick innings",
        ),
        "pitcher",
        lambda t, x: _clears((_g(t, "command"), HIGH, True),
                             (_g(t, "stuff"), MID_HI, False)),
    ),
    Definition(
        "groundball", "Groundball Machine",
        (
            "keeps it on the floor and out of the seats",
            "he pitches to a shortstop, and it works",
            "the ball stays down, which is a fine way to survive",
        ),
        "pitcher",
        lambda t, x: _clears((_g(t, "grounders"), HIGH, True),
                             (_g(t, "suppress"), MID_LO, True)),
    ),
    Definition(
        "kitchen_sink", "Crafty Kitchen-Sink",
        (
            "five pitches, none overpowering, and a plan for each hitter",
            "he beats you with variety rather than velocity",
            "no single pitch scares anyone; the sequence does",
        ),
        "pitcher",
        lambda t, x: _clears((_g(t, "arsenal"), HIGH, True),
                             (_g(t, "stuff"), MID_HI, False)),
    ),
    Definition(
        "soft_contact", "Soft-Contact Manager",
        (
            "few strikeouts, few walks, and nothing hit squarely",
            "he does not miss bats and he does not need to",
            "contact happens, and almost none of it is loud",
        ),
        "pitcher",
        lambda t, x: _clears((_g(t, "suppress"), HIGH, True),
                             (_g(t, "command"), MID_LO, True),
                             (_g(t, "stuff"), MID_HI, False)),
    ),
    Definition(
        "flyball", "Flyball and Homer-Prone",
        (
            "everything goes in the air, and some of it keeps going",
            "he lives with fly balls, which is a risky lease",
            "the ball gets up, and in the wrong park it leaves",
        ),
        "pitcher",
        lambda t, x: _clears((_g(t, "grounders"), LOW, False),
                             (_g(t, "suppress"), LOW, False)),
    ),
    Definition(
        "two_pitch", "Two-Pitch Power Reliever",
        (
            "two offerings, thrown hard, for one turn through",
            "one time through the order is the whole design",
            "two pitches and enough velocity to make them play",
        ),
        "pitcher",
        lambda t, x: _clears((_g(t, "arsenal"), POOR, False),
                             (_g(t, "stuff"), HIGH, True)),
    ),
    Definition(
        "breaking_ball", "Breaking-Ball Specialist",
        (
            "builds everything off the bender",
            "the breaking ball is the pitch, and everything sets it up",
            "he throws his breaking stuff more than his fastball",
        ),
        "pitcher",
        lambda t, x: _clears((x.get("breaking_share", float("nan")), 40.0, True),
                             (_g(t, "stuff"), MID_LO, True)),
    ),
    Definition(
        "innings_eater", "Innings Eater",
        (
            "average most places, and takes the ball every fifth day",
            "nothing special, and he will get you through six",
            "he is here for length rather than dominance",
        ),
        "pitcher",
        lambda t, x: _clears((_g(t, "stuff"), MID_HI, False),
                             (_g(t, "command"), MID_LO, True),
                             (x.get("bf_per_game", float("nan")), 20.0, True)),
    ),
    Definition(
        "struggling", "Struggling",
        (
            "nothing is playing above average at the moment",
            "no part of this is working right now",
            "everything is a little short, and it shows in the results",
        ),
        "pitcher",
        lambda t, x: None if _has_plus(t) else _clears(
            (_g(t, "stuff"), MID_LO, False),
            (_g(t, "command"), MID_LO, False),
            (_g(t, "suppress"), MID_LO, False)),
    ),
)


def match_profiles(
    tools: dict[str, Tool], extra: dict[str, float], *, kind: str, limit: int = 3
) -> list[Match]:
    """Every profile this player fits, most squarely first.

    Multiple matches are the expected case rather than a failure to decide, and
    no match is a valid answer -- a player who sits mid-table on every axis has
    no archetype, and inventing the nearest one would be the same manufactured
    observation the significance floor exists to prevent.
    """
    definitions = BATTER_PROFILES if kind == "batter" else PITCHER_PROFILES
    found = []
    for definition in definitions:
        try:
            strength = definition.rule(tools, extra)
        except Exception:  # noqa: BLE001 -- a missing measure is not a match
            strength = None
        if strength is not None:
            found.append(Match(definition.code, definition.label,
                               definition.blurb, float(strength)))
    found.sort(key=lambda m: -m.strength)
    return found[:limit]


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def _pct(value, population) -> float:
    if value is None or not population:
        return float("nan")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return _percentile(value, np.asarray(population, dtype=float))


def _rate_per_150(total, games: float) -> float:
    """A counting stat put on a common footing.

    Six outs above average in forty games is elite and the same six across a
    full season is ordinary, so nothing here is graded on an accumulated total.
    """
    if total is None or not games:
        return float("nan")
    try:
        return float(total) * 150.0 / float(games)
    except (TypeError, ValueError):
        return float("nan")


def external_grades(box: Any, populations: dict[str, list[float]]) -> dict[str, float]:
    """Speed, defense and baserunning, graded against the whole league.

    These three cannot come from the pitch corpus, and they cannot be graded
    against the twenty-six players on tonight's card either -- a card is not a
    population. The leaderboards travel with the bundle for exactly this.
    """
    running = getattr(box, "running", {}) or {}
    fielding = getattr(box, "fielding", {}) or {}
    season = getattr(box, "season", {}) or {}
    games = float(season.get("gamesPlayed") or season.get("games") or 0)

    grades: dict[str, float] = {}
    grades["speed"] = _pct(running.get("sprint_speed"), populations.get("sprint_speed"))

    # Defense is rated per 150 games, and the league population is rescaled the
    # same way so the comparison is like for like.
    oaa = fielding.get("outs_above_average")
    if oaa is not None and games:
        pool = populations.get("outs_above_average") or []
        grades["defense"] = _pct(_rate_per_150(oaa, games), pool)
    else:
        grades["defense"] = float("nan")

    # Steals per time on first. A burner who rarely reaches should not be
    # marked down for chances he never had, and a part-timer who steals ten in
    # ten games should not be filed alongside a regular who stole ten in a year.
    measured = getattr(box, "baserunning_runs", None)
    pool = populations.get("baserunning_runs") or []
    if measured is not None and pool and games:
        grades["baserunning"] = _pct(_rate_per_150(measured, games), pool)
    else:
        on_first = sum(float(season.get(k) or 0)
                       for k in ("singles", "baseOnBalls", "hitByPitch"))
        if on_first >= 20:
            attempts = float(season.get("stolenBases") or 0)
            grades["baserunning"] = min(
                100.0, (attempts / on_first) / max(LEAGUE_SB_RATE, 1e-9) * 50.0
            )
        else:
            grades["baserunning"] = float("nan")
    return grades


# Stealing is zero-inflated: most of the league never runs, so a linear map
# from rate to percentile files the majority at zero and the note then reports
# "poor baserunning" for a catcher who was never going to steal. Nobody
# describes a player's weakness as not attempting steals. So the grade is a
# plus tool or it is nothing -- it can carry a player, it cannot condemn one.
BASERUNNING_FLOOR = 60.0


def value_runs(box: Any, xwoba: float | None, plate_appearances: int) -> tuple[float, float]:
    """Runs above average, total and per 150 games.

    Three terms, each already in runs: the bat, the legs and the glove, the
    last carrying the positional adjustment so a shortstop's glove is not
    weighed against a first baseman's. The per-150 figure is the one that
    answers "is he good"; the total answers "how much has he given you", and
    they are different questions.
    """
    season = getattr(box, "season", {}) or {}
    games = float(season.get("gamesPlayed") or season.get("games") or 0)

    runs = batting_runs(xwoba, plate_appearances) if xwoba is not None else 0.0
    runs += baserunning_runs(
        season, getattr(box, "baserunning_runs", None))
    runs += fielding_runs(box, games)
    per150 = runs * 150.0 / games if games else float("nan")
    return float(runs), float(per150)


def build_batter(
    box: Any, row: dict[str, float], reference, populations: dict[str, list[float]],
    *, minimum_pa: int = 120,
) -> PlayerProfile:
    """One batter's profile, or a thin one that says so.

    A profile built on forty plate appearances is a description of forty plate
    appearances. It is marked rather than withheld, because the tools still
    describe what he has done -- but nothing downstream should call him a
    Classic Slugger on that evidence.
    """
    plate_appearances = int(row.get("pa") or 0)
    tools = grade_tools(row, reference, BATTER_TOOLS)
    extra = external_grades(box, populations)

    for name, grade in extra.items():
        if name == "baserunning" and (
            not np.isfinite(grade) or grade < BASERUNNING_FLOOR
        ):
            continue
        if np.isfinite(grade):
            tools[name] = Tool(
                name=name, label={"speed": "speed", "defense": "defense",
                                  "baserunning": "baserunning"}[name],
                grade=float(grade), z=(grade - 50.0) / 25.0, support={},
            )

    total, per150 = value_runs(box, row.get("xwoba"), plate_appearances)
    thin = plate_appearances < minimum_pa
    return PlayerProfile(
        player_id=int(getattr(box, "player_id", 0) or 0),
        kind="batter", tools=tools,
        matches=[] if thin else match_profiles(tools, extra, kind="batter"),
        runs_per_150=None if thin else per150,
        tier="" if thin else tier_for(per150),
        tier_grade=per150, sample=plate_appearances, thin=thin,
    )


def build_pitcher(
    box: Any, row: dict[str, float], reference, *, minimum_bf: int = 120,
) -> PlayerProfile:
    """One pitcher's profile.

    No external layer: everything a pitcher is judged on here -- stuff, command,
    what happens on contact, how many pitches he has -- is in the pitch corpus
    already.
    """
    batters_faced = int(row.get("bf") or 0)
    tools = grade_tools(row, reference, PITCHER_TOOLS)
    extra = {
        "breaking_share": float(row.get("breaking_share") or float("nan")),
        "bf_per_game": float(row.get("bf_per_game") or float("nan")),
    }
    thin = batters_faced < minimum_bf
    return PlayerProfile(
        player_id=int(getattr(box, "player_id", 0) or 0),
        kind="pitcher", tools=tools,
        matches=[] if thin else match_profiles(tools, extra, kind="pitcher"),
        sample=batters_faced, thin=thin,
    )


# --------------------------------------------------------------------------
# The reference populations
# --------------------------------------------------------------------------
# Derived from the pitch corpus, and therefore able to freeze exactly like every
# other derived table in this project. They are rebuilt by the same refresh that
# rebuilds the first-five table, for the same reason: a grade is a percentile
# among this season's players, and a stale population silently re-ranks
# everybody the moment the league moves.

PITCH_COLUMNS_FOR_PROFILES = [
    "batter", "pitcher", "season", "events", "description", "zone",
    "launch_speed", "launch_angle", "launch_speed_angle", "bb_type",
    "release_speed", "pitch_name", "estimated_woba_using_speedangle",
    "game_pk",
]

SWING_DESCRIPTIONS = {
    "hit_into_play", "foul", "swinging_strike", "swinging_strike_blocked",
    "foul_tip", "foul_bunt", "missed_bunt", "bunt_foul_tip",
}
WHIFF_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "missed_bunt",
}
STRIKEOUTS = ("strikeout", "strikeout_double_play")
WALKS = ("walk", "intent_walk")
BREAKING = {"Slider", "Curveball", "Knuckle Curve", "Sweeper", "Slurve", "Slow Curve"}

# The plate appearances that never become batted balls still have a wOBA, and
# leaving them out would grade a hitter on his contact alone -- flattering the
# free swinger who never walks and punishing nobody for striking out.
UNBATTED_WOBA = {
    "strikeout": 0.0, "strikeout_double_play": 0.0,
    "walk": 0.690, "intent_walk": 0.690, "hit_by_pitch": 0.720,
}


def _prepare(pitch):
    import pandas as pd

    pitch = pitch.copy()
    pitch["is_swing"] = pitch["description"].isin(SWING_DESCRIPTIONS)
    pitch["is_whiff"] = pitch["description"].isin(WHIFF_DESCRIPTIONS)
    pitch["in_zone"] = pitch["zone"].between(1, 9)
    pitch["chased"] = pitch["is_swing"] & ~pitch["in_zone"]
    pitch["barrel"] = pitch["launch_speed_angle"] == 6
    return pitch


def build_batter_reference(pitch, *, minimum_pa: int = 120):
    """One row per batter-season, with every measure the tools read."""
    import numpy as np
    import pandas as pd

    pitch = _prepare(pitch)
    ends = pitch[pitch["events"].notna() & (pitch["events"] != "")]

    grouped = pitch.groupby(["batter", "season"])
    frame = pd.DataFrame({
        "swing_rate": grouped["is_swing"].mean(),
        "whiff_per_swing": grouped.apply(
            lambda d: d["is_whiff"].sum() / max(d["is_swing"].sum(), 1),
            include_groups=False),
        "chase_rate": grouped.apply(
            lambda d: d["chased"].sum() / max((~d["in_zone"]).sum(), 1),
            include_groups=False),
    })

    ended = ends.groupby(["batter", "season"])
    frame["pa"] = ended.size()
    frame["k_rate"] = ended["events"].apply(lambda s: s.isin(STRIKEOUTS).mean())
    frame["bb_rate"] = ended["events"].apply(lambda s: s.isin(WALKS).mean())
    frame["hr_rate"] = ended["events"].apply(lambda s: (s == "home_run").mean())

    expected = ends["estimated_woba_using_speedangle"].astype(float)
    expected = expected.where(
        ~ends["events"].isin(UNBATTED_WOBA), ends["events"].map(UNBATTED_WOBA))
    frame["xwoba"] = ends.assign(x=expected).groupby(["batter", "season"])["x"].mean()

    in_play = ends[ends["bb_type"].notna() & (ends["bb_type"] != "")]
    balls = in_play.groupby(["batter", "season"])
    frame["barrel_rate"] = balls["barrel"].mean()
    frame["ev"] = balls["launch_speed"].mean()
    frame["la"] = balls["launch_angle"].mean()
    for name, kind in (("gb_rate", "ground_ball"), ("ld_rate", "line_drive"),
                       ("fb_rate", "fly_ball")):
        frame[name] = balls["bb_type"].apply(lambda s, k=kind: (s == k).mean())

    return frame[frame["pa"] >= minimum_pa].reset_index()


def build_pitcher_reference(pitch, *, minimum_bf: int = 120):
    """One row per pitcher-season, with every measure the tools read."""
    import numpy as np
    import pandas as pd

    pitch = _prepare(pitch)
    ends = pitch[pitch["events"].notna() & (pitch["events"] != "")]

    grouped = pitch.groupby(["pitcher", "season"])
    frame = pd.DataFrame({
        "whiff_per_swing": grouped.apply(
            lambda d: d["is_whiff"].sum() / max(d["is_swing"].sum(), 1),
            include_groups=False),
        "zone_rate": grouped["in_zone"].mean(),
        "velo": grouped["release_speed"].mean(),
        "mix": grouped["pitch_name"].nunique(),
        "primary_share": grouped["pitch_name"].apply(
            lambda s: s.value_counts(normalize=True).max() if len(s) else np.nan),
        "breaking_share": grouped["pitch_name"].apply(
            lambda s: s.isin(BREAKING).mean() * 100.0 if len(s) else np.nan),
    })

    ended = ends.groupby(["pitcher", "season"])
    frame["bf"] = ended.size()
    frame["k_rate"] = ended["events"].apply(lambda s: s.isin(STRIKEOUTS).mean())
    frame["bb_rate"] = ended["events"].apply(lambda s: s.isin(WALKS).mean())
    frame["bf_per_game"] = frame["bf"] / ended["game_pk"].nunique()

    in_play = ends[ends["bb_type"].notna() & (ends["bb_type"] != "")]
    balls = in_play.groupby(["pitcher", "season"])
    frame["barrel_allowed"] = balls["barrel"].mean()
    frame["ev_allowed"] = balls["launch_speed"].mean()
    for name, kind in (("gb_rate", "ground_ball"), ("fb_rate", "fly_ball")):
        frame[name] = balls["bb_type"].apply(lambda s, k=kind: (s == k).mean())

    return frame[frame["bf"] >= minimum_bf].reset_index()
