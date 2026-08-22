"""The three-part player note: who he is, how he is going, how tonight sets up.

The evaluators produce true isolated facts. This assembles them into the shape a
reader actually uses, which is fixed and in this order:

  1. **Profile** -- what kind of player he is, with the numbers behind it
  2. **Form** -- how he is going lately, and in this series
  3. **Matchup** -- how tonight sets up against this specific opponent

The order is not negotiable and is not sorted by significance. A scouting note
has a natural sequence: you establish who someone is before you say he is in a
slump, because "hitting .190 over ten games" means something different about a
Classic Slugger than about a Fringe Bat. Ranking these three by z-score would
scramble exactly the context that makes the second two legible.

Each part may be empty. A player with forty plate appearances has no profile
worth stating, a player who has been available all week has no form story, and
a reliever who may not appear has no matchup. Silence in one part is normal and
the other parts still stand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from guards_report.insight import profile as prof


@dataclass
class Dossier:
    """One player's three parts, each already written or deliberately empty."""

    player_id: int
    kind: str
    profile: str = ""
    form: str = ""
    matchup: str = ""
    tier: str = ""
    labels: list[str] = field(default_factory=list)
    thin: bool = False

    @property
    def parts(self) -> list[tuple[str, str]]:
        return [(name, text) for name, text in
                (("Profile", self.profile), ("Form", self.form),
                 ("Matchup", self.matchup)) if text]

    @property
    def empty(self) -> bool:
        return not self.parts


def _grade_word(grade: float) -> str:
    for cut, word in ((prof.ELITE, "elite"), (prof.HIGH, "plus"),
                      (prof.MID_LO, "average"), (prof.POOR, "below average")):
        if grade >= cut:
            return word
    return "poor"


def _pct_ordinal(grade: float) -> str:
    n = int(round(grade))
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# How to say each raw measure. Keyed by measure rather than by tool, because
# the sentence quotes whichever measure actually drove the grade -- a composite
# that reads "elite contact quality, 89.1 mph" for one player and "poor, 87.2
# mph" for another is two true statements that cannot both be believed.
MEASURE_PHRASE = {
    "barrel_rate":     "{:.1%} barrel rate",
    "hr_rate":         "a home run every {inv:.0f} plate appearances",
    "ev":              "{:.1f} mph average exit velocity",
    "k_rate":          "{:.1%} strikeout rate",
    "whiff_per_swing": "{:.1%} whiff rate",
    "bb_rate":         "{:.1%} walk rate",
    "chase_rate":      "{:.1%} chase rate",
    "la":              "{:.0f} degree average launch angle",
    "gb_rate":         "{:.1%} ground-ball rate",
    "ld_rate":         "{:.1%} line-drive rate",
    "fb_rate":         "{:.1%} fly-ball rate",
    "zone_rate":       "{:.1%} of his pitches in the zone",
    "barrel_allowed":  "{:.1%} barrel rate against",
    "ev_allowed":      "{:.1f} mph average exit velocity against",
    "mix":             "{:.0f} distinct pitches",
    "primary_share":   "{:.0%} of his pitches on one offering",
}


def _evidence(tool: prof.Tool) -> str:
    """The measure that earned the grade, phrased for a reader."""
    measure = tool.driver
    template = MEASURE_PHRASE.get(measure)
    if not template or measure not in tool.support:
        return ""
    value = tool.support[measure]
    try:
        if "{inv:" in template:
            return template.format(inv=1.0 / value) if value else ""
        return template.format(value)
    except (ValueError, TypeError, ZeroDivisionError):
        return ""


def write_profile(player: prof.PlayerProfile, *, surname: str) -> str:
    """Part one: what kind of player he is, and whether he is any good.

    Leads with the tier because it is the question the reader came with, then
    the archetype, then the evidence. A profile that opened with a barrel rate
    would be another correct isolated fact.
    """
    if player.thin:
        return (f"{player.sample} plate appearances is too little to profile "
                f"{surname} on." if player.kind == "batter" else
                f"{player.sample} batters faced is too little to profile "
                f"{surname} on.")

    pieces: list[str] = []

    labels = [m.label for m in player.matches]
    if labels and player.tier:
        joined = labels[0] if len(labels) == 1 else (
            f"{labels[0]} and {labels[1]}" if len(labels) == 2 else
            f"{labels[0]}, {labels[1]} and {labels[2]}")
        pieces.append(f"{surname} is {player.tier} — {joined.lower()}")
    elif labels:
        pieces.append(f"{surname} profiles as a {labels[0].lower()}")
    elif player.tier:
        pieces.append(f"{surname} is {player.tier} without a clear archetype")
    else:
        pieces.append(f"{surname} sits mid-table across the board")

    if player.matches:
        pieces[-1] += f", {player.matches[0].blurb}"

    # The evidence. Two tools at most: the one carrying him and the one that
    # costs him, because a list of five grades is a table, not a read.
    best = player.carrying[:1]
    worst = [t for t in player.weaknesses if t.name not in {b.name for b in best}][:1]
    support = []
    for tool in best + worst:
        note = _evidence(tool)
        band = _grade_word(tool.grade)
        support.append(
            f"{band} {tool.label} ({_pct_ordinal(tool.grade)} percentile"
            + (f", {note}" if note else "") + ")"
        )
    if support:
        pieces.append(" and ".join(support))

    if player.runs_per_150 is not None and np.isfinite(player.runs_per_150):
        pieces.append(
            f"worth {player.runs_per_150:+.0f} runs per 150 games once his bat, "
            "legs and glove are counted together"
        )

    return ". ".join(p[0].upper() + p[1:] for p in pieces) + "."


# --------------------------------------------------------------------------
# Part two: form
# --------------------------------------------------------------------------
# Two questions, and they are not the same question. "How is he going" is about
# the last few weeks against his own baseline. "How has he gone in this series"
# is three or four games -- far too few to claim a trend, and exactly what a
# reader wants to know anyway. So the first is tested and the second is
# reported: a rolling number gets a significance check before it may say
# "trending up", and a series line is stated as the fact it is.

# A recent window has to clear this before the word "trend" is used. Derived
# the same way as everywhere else in this package -- roughly two criteria
# scanned per player, so a two-sigma departure is the bar.
TREND_FLOOR = 2.0


def _window(box: Any, label: str) -> dict | None:
    """One of the L5 / L15 / L30 rolling lines the box already carries."""
    for line in (getattr(box, "windows", None) or []):
        if getattr(line, "label", "") == label:
            return getattr(line, "stats", None) or None
    return None


def _two_proportion_z(hits: float, at_bats: float,
                      base_rate: float) -> float | None:
    """How far a recent rate sits from the player's own season rate.

    The standard two-proportion form against his own baseline rather than the
    league's -- the question is whether *he* has changed, which is not the same
    as whether he is currently better than average. A slugger hitting .250 over
    a fortnight may be slumping badly and still be above the league.
    """
    if at_bats <= 0 or not 0 < base_rate < 1:
        return None
    se = (base_rate * (1 - base_rate) / at_bats) ** 0.5
    if se <= 0:
        return None
    return float((hits / at_bats - base_rate) / se)


def write_form(box: Any, player: prof.PlayerProfile, *, surname: str) -> str:
    """Part two: how he is going lately, and in this series.

    The trend claim is gated; the series line is not. A hot fortnight is a claim
    about a change and has to clear a floor before it is stated as one. Three
    games in a series is a fact about three games, and "2 for 11 in the series"
    asserts nothing that needs testing.

    Fifteen games is the window. Five is too few for any test to fire and thirty
    is long enough that it stops being news.
    """
    pieces: list[str] = []

    recent = _window(box, "L15")
    season = getattr(box, "season", {}) or {}
    if recent and season:
        at_bats = float(recent.get("atBats") or 0)
        hits = float(recent.get("hits") or 0)
        season_ab = float(season.get("atBats") or 0)
        season_hits = float(season.get("hits") or 0)
        if at_bats >= 25 and season_ab >= 100:
            base = season_hits / season_ab
            z = _two_proportion_z(hits, at_bats, base)
            ops = recent.get("ops")
            season_ops = season.get("ops")
            if z is not None and abs(z) >= TREND_FLOOR:
                way = "up" if z > 0 else "down"
                pieces.append(
                    f"he is trending {way} — hitting {hits / at_bats:.3f} over "
                    f"the last fifteen games against {base:.3f} for the season"
                )
            else:
                # The sample is stated because without it "inside normal
                # variation" beside a ninety-point gap reads as an error rather
                # than as what it is: thirty at-bats cannot separate those.
                pieces.append(
                    f"the last fifteen games look like the rest of his season — "
                    f"{hits / at_bats:.3f} against {base:.3f}, which "
                    f"{int(at_bats)} at-bats cannot separate"
                )

            # Power moves independently of average, and the test above cannot
            # see it: a hitter can keep his average while slugging 200 points
            # more, and calling that unchanged is false in the way that
            # matters. Reported separately rather than folded in, because it is
            # a different claim resting on a different measure.
            iso, season_iso = recent.get("iso"), season.get("iso")
            if iso is not None and season_iso is not None and season_iso > 0:
                gap = float(iso) - float(season_iso)
                if abs(gap) >= 0.075:
                    direction = "more" if gap > 0 else "less"
                    pieces.append(
                        f"with {direction} power than usual — an isolated "
                        f"slugging of {float(iso):.3f} against "
                        f"{float(season_iso):.3f}"
                    )

    series = getattr(box, "series_line", None)
    stat = getattr(series, "stat", None) if series is not None else None
    if stat:
        at_bats = stat.get("atBats")
        hits = stat.get("hits")
        if at_bats:
            extra = []
            for key, word in (("homeRuns", "home run"), ("rbi", "RBI"),
                              ("baseOnBalls", "walk"), ("strikeOuts", "strikeout")):
                count = stat.get(key)
                if count:
                    extra.append(
                        f"{int(count)} {word}" + ("s" if int(count) != 1 else "")
                    )
            tail = f" with {' and '.join(extra[:2])}" if extra else ""
            pieces.append(f"{int(hits or 0)} for {int(at_bats)} in this series{tail}")
        elif stat.get("plateAppearances"):
            walks = int(stat.get("baseOnBalls") or 0)
            if walks:
                pieces.append(
                    f"{walks} walk{'s' if walks != 1 else ''} in this series "
                    "without an official at-bat"
                )

    if not pieces:
        return ""
    return ". ".join(p[0].upper() + p[1:] for p in pieces) + "."


# --------------------------------------------------------------------------
# Part three: matchup
# --------------------------------------------------------------------------

def write_matchup(
    box: Any, player: prof.PlayerProfile, *, surname: str,
    opposing_starter: Any = None, opposing_profile: prof.PlayerProfile | None = None,
    projection_line: dict | None = None,
) -> str:
    """Part three: how tonight sets up against this specific opponent.

    Built from the two profiles rather than from a head-to-head line. Nine
    career at-bats against a pitcher is noise dressed as history, and it is the
    single most common way a scouting note says something confident and untrue.
    What a groundball pitcher does to a hitter who beats the ball into the
    ground is a real claim, and it rests on hundreds of plate appearances on
    both sides.
    """
    pieces: list[str] = []

    if opposing_starter is not None and opposing_profile is not None:
        their_name = (getattr(opposing_starter, "name", "") or "").split(" (")[0]
        their_surname = their_name.split()[-1] if their_name else "the starter"

        # Where the two profiles actually interact. Each pairing is a claim
        # about how one man's strength meets another's weakness, which is what
        # a matchup is; a list of both players' grades is not.
        ours, theirs = player.tools, opposing_profile.tools

        def grade(tools, name):
            tool = tools.get(name)
            return tool.grade if tool and np.isfinite(tool.grade) else float("nan")

        contact, loft = grade(ours, "contact"), grade(ours, "loft")
        power = grade(ours, "power")
        stuff, grounders = grade(theirs, "stuff"), grade(theirs, "grounders")
        command, suppress = grade(theirs, "command"), grade(theirs, "suppress")

        # Every pairing is a claim about how one man's strength meets
        # another's weakness. Written as pairings rather than as a list of both
        # players' grades, because two columns of percentiles is a table and
        # the reader still has to do the work.
        notes = []
        chase = grade(ours, "discipline")

        if np.isfinite(contact) and np.isfinite(stuff):
            if stuff >= prof.HIGH and contact <= prof.LOW:
                notes.append(
                    f"{their_surname} misses bats and {surname} does not make "
                    "much contact, which is the worst version of this for him")
            elif contact >= prof.HIGH and stuff >= prof.HIGH:
                notes.append(
                    f"{their_surname}'s swing-and-miss against one of the "
                    "harder men in the league to strike out")
            elif contact >= prof.HIGH and stuff <= prof.MID_LO:
                notes.append(
                    f"a contact hitter against a pitcher who does not miss "
                    "bats, so the ball is going to be in play")
            elif contact <= prof.LOW and stuff <= prof.MID_LO:
                notes.append(
                    f"{surname} swings through a lot, but {their_surname} is "
                    "not the man to punish it")

        if np.isfinite(loft) and np.isfinite(grounders):
            if grounders >= prof.HIGH and loft <= prof.LOW:
                notes.append(
                    f"{their_surname} keeps it on the ground and {surname} "
                    "already hits it there")
            elif grounders >= prof.HIGH and loft >= prof.HIGH:
                notes.append(
                    f"{surname} wants it in the air and {their_surname} will "
                    "not let him have it")
            elif grounders <= prof.LOW and loft >= prof.HIGH:
                notes.append(
                    f"{their_surname} lets the ball get airborne, which is "
                    f"exactly where {surname} wants it")

        if np.isfinite(power) and np.isfinite(suppress):
            if power >= prof.HIGH and suppress <= prof.LOW:
                notes.append(
                    f"{surname}'s power against a pitcher who gives up hard "
                    "contact is the danger here")
            elif power >= prof.HIGH and suppress >= prof.HIGH:
                notes.append(
                    f"{their_surname} has kept the barrel off the ball all "
                    f"year, which is the one thing {surname} needs")

        if np.isfinite(command) and np.isfinite(chase):
            if command >= prof.HIGH and chase <= prof.LOW:
                notes.append(
                    f"{their_surname} pounds the zone and {surname} chases, so "
                    "the free pass is unlikely to arrive")
            elif command <= prof.LOW and chase >= prof.HIGH:
                notes.append(
                    f"{their_surname} is around the zone less than most and "
                    f"{surname} will make him prove it")
            elif command <= prof.LOW and chase <= prof.LOW:
                notes.append(
                    f"neither man is disciplined here — {their_surname} misses "
                    f"the zone and {surname} swings at it anyway")

        pieces.extend(notes[:2])

    # The model's own number for tonight, which is the one thing here that
    # accounts for the park and where he bats as well as who is pitching.
    if projection_line:
        hits = projection_line.get("hits_expected")
        homer = projection_line.get("homer_chance")
        slot = projection_line.get("slot")
        if hits is not None:
            bits = [f"the model projects {hits:.1f} hits"]
            if homer:
                bits.append(f"a {homer:.0%} chance to go deep")
            if slot:
                bits.append(f"batting {int(slot)}")
            pieces.append(", ".join(bits))

    if not pieces:
        return ""
    return ". ".join(p[0].upper() + p[1:] for p in pieces) + "."


def build(
    box: Any, player: prof.PlayerProfile, *, surname: str,
    opposing_starter: Any = None, opposing_profile: prof.PlayerProfile | None = None,
    projection_line: dict | None = None,
) -> Dossier:
    """All three parts for one player."""
    return Dossier(
        player_id=player.player_id, kind=player.kind,
        profile=write_profile(player, surname=surname),
        form=write_form(box, player, surname=surname),
        matchup=write_matchup(
            box, player, surname=surname, opposing_starter=opposing_starter,
            opposing_profile=opposing_profile, projection_line=projection_line),
        tier=player.tier,
        labels=[m.label for m in player.matches],
        thin=player.thin,
    )
