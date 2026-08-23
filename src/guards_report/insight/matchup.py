"""The game-level read — where the projections belong.

Every other evaluator describes a player. This one describes the game, and it is
the one place where the fitted models have something to say that no amount of
season-line reading reaches: a calibrated win probability, an expected score,
a first-five result, and the decomposition of which input actually moved the
number.

That decomposition is the point. A model paragraph can say Cleveland has the
better starter and the worse offence, and be right, without being able to say
which of those mattered more tonight. The win model is linear on the log-odds
scale, so the contributions are exact rather than an attribution guess, and the
two largest are the comparison a reader should take away.

Unlike the player evaluators this produces a short paragraph rather than a
sentence, because a game has several independent things worth saying and forcing
them into one line reads as a list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from guards_report.insight import profile as prof
from guards_report.insight import voice as voice_module
from guards_report.insight.types import Finding, Reference


def _finding(
    *, code: str, family: str, kind: str, value: float, reference: Reference,
    detail: dict, inferential: bool = False, evidence: int = 1000,
) -> Finding:
    return Finding(
        subject=0, subject_kind="game", code=code, family=family, kind=kind,
        value=float(value), reference=reference, evidence=evidence,
        stabilisation=1, inferential=inferential, detail=detail,
    )


def from_projection(projection: Any, home: str, away: str) -> list[Finding]:
    """Findings the fitted models produce, which no season line contains."""
    if projection is None:
        return []

    findings: list[Finding] = []
    win = float(projection.win_probability)
    favorite = home if win >= 0.5 else away
    confidence = max(win, 1 - win)

    reference = projection.reference or {}
    spread = float(reference.get("win_prob_sd") or 0.087)

    findings.append(_finding(
        code="game.projection", family="projection", kind="matchup",
        value=confidence - 0.5,
        reference=Reference(mean=0.0, sd=spread, population="model_history"),
        detail={
            "favorite": favorite,
            "probability": confidence,
            "percentile": float(getattr(projection, "confidence_percentile", 50.0)),
            "coherent": bool(projection.coherent),
        },
    ))

    score = projection.score or {}
    if score:
        findings.append(_finding(
            code="game.score", family="score", kind="matchup",
            value=float(score.get("expected_total", 0.0)),
            reference=Reference(
                mean=float(reference.get("total_runs_mean") or 8.99),
                sd=1.0, population="league_totals",
            ),
            detail={
                "home_runs": float(score.get("exp_home_runs", 0.0)),
                "away_runs": float(score.get("exp_away_runs", 0.0)),
                "total": float(score.get("expected_total", 0.0)),
                "one_run": float(score.get("p_one_run_game", 0.0)),
                "home": home, "away": away,
            },
        ))

    # What moved the probability. Exact, because the model is linear on the
    # log-odds scale -- these terms sum to the whole departure from an average
    # matchup rather than approximating it.
    contributions = [
        c for c in (projection.contributions or [])
        if abs(c.get("contribution", 0.0)) > 0.01
    ]
    if contributions:
        # Sorted here rather than trusted from the caller. The projection page
        # happens to hand these over largest-first, so reading element zero
        # produced the right answer and would have kept producing it until the
        # day that ordering changed -- at which point the sentence names the
        # wrong driver, with the wrong number, and still reads perfectly.
        contributions.sort(key=lambda c: -abs(float(c.get("contribution", 0.0))))
        top = contributions[0]
        findings.append(_finding(
            code="game.driver", family="driver", kind="matchup",
            value=abs(float(top["contribution"])),
            reference=Reference(mean=0.0, sd=0.15, population="typical_contribution"),
            detail={
                "name": top["name"],
                "toward": home if top["contribution"] > 0 else away,
                "size": abs(float(top["contribution"])),
                "count": len(contributions),
            },
        ))

    first_five = getattr(projection, "first_five", None)
    if first_five is not None:
        findings.append(_finding(
            code="game.first_five", family="first_five", kind="matchup",
            value=abs(first_five.home_leads - first_five.away_leads),
            reference=Reference(mean=0.0, sd=0.08, population="league_f5"),
            detail={
                "home": home, "away": away,
                "home_leads": first_five.home_leads,
                "away_leads": first_five.away_leads,
                "tied": first_five.tied,
                "total": first_five.expected_total,
            },
        ))

    return findings


def key_player(projection: Any, home: str, away: str) -> list[Finding]:
    """The hitter the projection likes most tonight.

    Chosen on the modelled probability rather than on a season line, so it
    accounts for who is pitching, the park and where he bats -- the three things
    a raw home-run total cannot see.
    """
    if projection is None or not getattr(projection, "player_props", None):
        return []

    best = None
    for side, team in (("home", home), ("away", away)):
        for prop in projection.player_props.get(side) or []:
            chance = float((prop.home_runs or {}).get("at_least_one", 0.0))
            if prop.thin:
                continue
            if best is None or chance > best[0]:
                best = (chance, prop, team)

    if best is None or best[0] <= 0:
        return []

    chance, prop, team = best
    return [_finding(
        code="game.key_bat", family="key_player", kind="matchup",
        value=chance,
        reference=Reference(mean=0.12, sd=0.05, population="posted_lineups"),
        detail={
            "name": prop.name or str(prop.player_id),
            "team": team, "slot": prop.slot,
            "homer": chance,
            "hits": float((prop.hits or {}).get("expected", 0.0)),
        },
    )]


def starter_strikeout_edge(projection: Any, home: str, away: str) -> list[Finding]:
    """Which starter the strikeout model expects more from."""
    if projection is None or not getattr(projection, "strikeouts", None):
        return []

    home_k = projection.strikeouts.get("home")
    away_k = projection.strikeouts.get("away")
    if home_k is None or away_k is None:
        return []

    gap = float(home_k.expected - away_k.expected)
    leader, trailer = (home_k, away_k) if gap > 0 else (away_k, home_k)
    return [_finding(
        code="game.strikeouts", family="strikeouts", kind="matchup",
        value=abs(gap),
        reference=Reference(mean=0.0, sd=1.5, population="starter_pairs"),
        detail={
            "leader": leader.name or "", "leader_k": leader.expected,
            "leader_line": leader.line,
            "trailer": trailer.name or "", "trailer_k": trailer.expected,
        },
    )]


def paragraph(findings: list[Finding], *, limit: int = 5) -> str:
    """Assemble the game-level findings into a short paragraph.

    Ordered by what a reader wants first -- the call, then what drove it, then
    the shape of the game, then who to watch -- rather than by significance. A
    matchup note has a natural order and sorting it by z-score would scramble it.
    """
    from guards_report.insight.render import render

    order = ["game.projection", "game.driver", "game.score", "game.first_five",
             "game.strikeouts", "game.key_bat"]
    by_code = {f.code: f for f in findings}

    parts = []
    for code in order:
        if code not in by_code or len(parts) >= limit:
            continue
        text = render(by_code[code])
        if text:
            parts.append(text)

    if not parts:
        return ""
    return " ".join(p[0].upper() + p[1:] + ("." if not p.endswith(".") else "")
                    for p in parts)


# --------------------------------------------------------------------------
# The three-part matchup note
# --------------------------------------------------------------------------
# Same shape as a player's: a fixed reading order, not a significance ranking.
# You establish who these two clubs are before you say who is favoured tonight,
# because "a 57% favourite" means something different about a first-place team
# than about a last-place one.
#
#   1. On paper   -- record, run differential, recent form, the series, rating
#   2. On the mound -- the two starters, against each other and their opponents
#   3. Projections  -- what the fitted models say, and what moved them
#
# Everything here is a difference between the two clubs rather than a fact
# about one. "Cleveland are 71-57" is a standings lookup; "Cleveland are the
# better team on record and the ratings separate them further still" is the
# thing a reader came for.


@dataclass
class MatchupNote:
    """The game-level note, in three parts."""

    on_paper: str = ""
    on_the_mound: str = ""
    projections: str = ""

    @property
    def parts(self) -> list[tuple[str, str]]:
        return [(name, text) for name, text in
                (("On paper", self.on_paper),
                 ("On the mound", self.on_the_mound),
                 ("Projections", self.projections)) if text]

    @property
    def empty(self) -> bool:
        return not self.parts


def _record(section) -> Any:
    profile = getattr(section, "profile", None)
    return getattr(profile, "record", None) if profile else None


def _split(record, key: str) -> tuple[int, int] | None:
    splits = getattr(record, "splits", None) or {}
    pair = splits.get(key)
    if not pair or len(pair) != 2:
        return None
    return int(pair[0]), int(pair[1])


def _say(voice, code: str, *key, **slots) -> str:
    if voice is not None:
        return voice.team(code, *key, **slots)
    return voice_module.team_phrase(code, *key, **slots)


def write_on_paper(bundle: Any, *, voice=None) -> str:
    """Part one: which of these two clubs is actually better.

    Four independent readings of the same question -- the standings, the run
    differential, the last ten, and the model's own rating -- and the useful
    output is where they disagree. A club whose record outruns its run
    differential is a different proposition from one whose does not, and that
    only shows when both are on the page together.
    """
    home, away = bundle.home, bundle.away
    home_rec, away_rec = _record(home), _record(away)
    if home_rec is None or away_rec is None:
        return ""

    pieces: list[str] = []
    key = bundle.game_pk

    def pct(record) -> float:
        try:
            return record.wins / max(record.wins + record.losses, 1)
        except (TypeError, AttributeError):
            return 0.0

    home_pct, away_pct = pct(home_rec), pct(away_rec)
    leader, trailer = ((home, home_rec), (away, away_rec))
    if away_pct > home_pct:
        leader, trailer = ((away, away_rec), (home, home_rec))
    (lead_section, lead_rec), (trail_section, trail_rec) = leader, trailer

    def line(record) -> str:
        return f"{record.wins}-{record.losses}"

    # A tenth of a point of win percentage over a full season is about sixteen
    # games. Below that these are the same team with different luck.
    close = abs(home_pct - away_pct) < 0.060
    pieces.append(_say(
        voice, "rec.close" if close else "rec.gap", key,
        fav=lead_section.abbreviation, dog=trail_section.abbreviation,
        favrec=line(lead_rec), dogrec=line(trail_rec)))

    # Run differential, which is the same question asked of the scoreboard
    # rather than the win column.
    home_diff = getattr(home.profile, "run_differential", None)
    away_diff = getattr(away.profile, "run_differential", None)
    if home_diff is not None and away_diff is not None and not close:
        lead_diff = home_diff if lead_section is home else away_diff
        trail_diff = away_diff if lead_section is home else home_diff
        if abs(lead_diff - trail_diff) >= 40:
            pieces.append(_say(
                voice, "rundiff.gap", key,
                fav=lead_section.abbreviation, dog=trail_section.abbreviation,
                favdiff=f"{lead_diff:+d}", dogdiff=f"{trail_diff:+d}"))

    # Where the two disagree. A club four wins above its Pythagorean record has
    # been getting results its scoring does not support, and saying so is worth
    # more than either number alone.
    for section, record in ((home, home_rec), (away, away_rec)):
        luck = getattr(record, "luck", None)
        if luck is None or abs(int(luck)) < 4:
            continue
        pieces.append(_say(
            voice, "luck.flattered" if int(luck) > 0 else "luck.unlucky", key,
            team=section.abbreviation, luck=abs(int(luck))))
        break

    # Recent form, and whether it agrees with the season.
    home_ten, away_ten = _split(home_rec, "lastTen"), _split(away_rec, "lastTen")
    if home_ten and away_ten:
        def ten(pair) -> str:
            return f"{pair[0]}-{pair[1]}"
        lead_ten = home_ten if lead_section is home else away_ten
        trail_ten = away_ten if lead_section is home else home_ten
        if lead_ten[0] > trail_ten[0]:
            pieces.append(_say(
                voice, "form.gap", key,
                fav=lead_section.abbreviation, dog=trail_section.abbreviation,
                favten=ten(lead_ten), dogten=ten(trail_ten)))
        elif trail_ten[0] > lead_ten[0] + 1:
            # The interesting case: the worse team is playing better right now.
            pieces.append(_say(
                voice, "form.against", key,
                hot=trail_section.abbreviation, cold=lead_section.abbreviation,
                hotten=ten(trail_ten), coldten=ten(lead_ten)))

    # The series, which is the only part of this a reader cannot get from a
    # standings page.
    series = getattr(bundle, "series", None)
    if series is not None:
        played = getattr(series, "games_played", 0) or 0
        total = getattr(series, "games_in_series", 0) or 0
        hw = getattr(series, "home_wins", 0) or 0
        aw = getattr(series, "away_wins", 0) or 0
        if played == 0:
            pieces.append(_say(voice, "series.opener", key, total=total))
        elif hw == aw:
            pieces.append(_say(voice, "series.level", key, lead=f"{hw}-{aw}"))
        else:
            winner = home if hw > aw else away
            pieces.append(_say(
                voice, "series.led", key, leader=winner.abbreviation,
                lead=f"{max(hw, aw)}-{min(hw, aw)}",
                played=f"{played} game{'s' if played != 1 else ''}"))

    # And the model's own rating, which is a fifth reading and the only one
    # that carries forward from previous seasons.
    projection = getattr(bundle, "projection", None)
    elo = None
    for entry in (getattr(projection, "contributions", None) or []):
        if entry.get("name") == "elo_logit":
            elo = entry
            break
    if elo is not None and not close:
        toward_home = float(elo.get("contribution", 0.0)) > 0
        rating_agrees = (toward_home and lead_section is home) or \
                        (not toward_home and lead_section is away)
        strong = abs(float(elo.get("contribution", 0.0))) >= 0.30
        if rating_agrees and strong:
            pieces.append(_say(voice, "elo.gap", key))
        elif rating_agrees:
            pieces.append(_say(voice, "elo.narrow", key))

    pieces = [p for p in pieces if p]
    if not pieces:
        return ""
    return ". ".join(p[0].upper() + p[1:] for p in pieces) + "."


def write_on_the_mound(
    bundle: Any, *, home_starter=None, away_starter=None,
    home_faces=None, away_faces=None, voice=None,
) -> str:
    """Part two: the two starters, against each other and against what they face.

    A starter comparison that only ranks the two men is half the question. The
    other half is who each of them has to get out, and those can point opposite
    ways -- the better pitcher can have the harder assignment, which is exactly
    the case a reader wants flagged and a season line cannot show.
    """
    home, away = bundle.home, bundle.away
    pieces: list[str] = []
    key = bundle.game_pk

    def name(box) -> str:
        raw = (getattr(box, "name", "") or "").split(" (")[0]
        return raw.split()[-1] if raw else ""

    def grade(player, tool) -> float:
        if player is None:
            return float("nan")
        found = player.tools.get(tool)
        return found.grade if found and np.isfinite(found.grade) else float("nan")

    # Who is the better pitcher, on the season. Run value is the single figure
    # that answers it, and it is already computed for both.
    if home_starter is not None and away_starter is not None:
        pairs = [(home_starter, bundle.home), (away_starter, bundle.away)]
        rated = [(p, s) for p, s in pairs if p is not None]
        if len(rated) == 2:
            def overall(player) -> float:
                grades = [t.grade for t in player.tools.values()
                          if np.isfinite(t.grade)]
                return float(np.mean(grades)) if grades else float("nan")

            home_grade = overall(home_starter)
            away_grade = overall(away_starter)
            if np.isfinite(home_grade) and np.isfinite(away_grade):
                gap = abs(home_grade - away_grade)
                better = home if home_grade > away_grade else away
                better_box = (home_starter if home_grade > away_grade
                              else away_starter)
                if gap >= 18:
                    pieces.append(_say(
                        voice, "sp.mismatch", key,
                        better=_starter_name(bundle, better_box)))
                elif gap <= 8:
                    pieces.append(_say(voice, "sp.even", key))

    # What kind of pitchers they are. Two archetypes side by side say more than
    # two run values, and this is where the profile work pays off.
    labels = {}
    for side, player in (("home", home_starter), ("away", away_starter)):
        if player is not None and player.matches:
            labels[side] = player.matches[0].label.lower()
    if len(labels) == 2 and labels["home"] != labels["away"]:
        pieces.append(_say(
            voice, "sp.contrast", key,
            a_desc=labels["away"], h_desc=labels["home"]))

    # And the assignment each of them draws. This is the half a starter
    # comparison usually leaves out.
    for player, lineup, section, opponent in (
        (home_starter, home_faces, home, away),
        (away_starter, away_faces, away, home),
    ):
        if player is None or lineup is None or not getattr(lineup, "tools", None):
            continue
        stuff = grade(player, "stuff")
        contact = lineup.grade("contact")
        power = lineup.grade("power")
        suppress = grade(player, "suppress")
        who = _starter_name(bundle, player)
        if not who:
            continue
        if np.isfinite(stuff) and np.isfinite(contact):
            if stuff >= prof.HIGH and contact <= prof.MID_LO:
                pieces.append(
                    f"{who} draws the easier assignment — {opponent.abbreviation} "
                    "do not make much contact and he misses bats")
                continue
            if stuff <= prof.MID_LO and contact >= prof.HIGH:
                pieces.append(
                    f"{who} has the harder night of it: "
                    f"{opponent.abbreviation} put the bat on the ball and he "
                    "does not miss many")
                continue
        if np.isfinite(power) and np.isfinite(suppress) and power >= prof.HIGH \
                and suppress <= prof.LOW:
            pieces.append(
                f"{opponent.abbreviation}'s power against a pitcher who has "
                f"been squared up all year is the risk in {who}'s start")

    # The bullpens, because a starter is five or six innings of a nine-inning
    # question.
    for section, other in ((home, away), (away, home)):
        profile = getattr(section, "profile", None)
        pitches = getattr(profile, "bullpen_pitches_last_3", None) if profile else None
        if pitches is not None and int(pitches) >= 260:
            pieces.append(_say(
                voice, "bullpen.tired", key,
                team=section.abbreviation, pitches=int(pitches)))
            break

    pieces = [p for p in pieces if p]
    if not pieces:
        return ""
    return ". ".join(p[0].upper() + p[1:] for p in pieces[:4]) + "."


def _starter_name(bundle: Any, player) -> str:
    """The surname of the box a profile belongs to."""
    target = getattr(player, "player_id", None)
    if target is None:
        return ""
    for section in (bundle.home, bundle.away):
        for box in getattr(section, "pitchers", []) or []:
            if int(getattr(box, "player_id", 0) or 0) == int(target):
                raw = (getattr(box, "name", "") or "").split(" (")[0]
                return raw.split()[-1] if raw else ""
    return ""


def write_projections(projection: Any, home: str, away: str, *, voice=None) -> str:
    """Part three: what the fitted models say, and what moved them.

    Unchanged in substance -- this is the section that already worked -- but it
    now sits last rather than alone. A win probability lands differently once
    the reader knows which of these two clubs is actually better and who is
    pitching, which is the whole argument for the three-part shape.
    """
    if projection is None:
        return ""
    findings = from_projection(projection, home, away)
    findings += starter_strikeout_edge(projection, home, away)
    findings += key_player(projection, home, away)
    return paragraph(findings)


def note(
    bundle: Any, *, home_starter=None, away_starter=None,
    home_faces=None, away_faces=None, voice=None,
) -> MatchupNote:
    """The whole game-level note, in reading order.

    Each part is independently guarded: a club with no standings data still gets
    a pitching matchup, and a game with no fitted projection still gets the
    first two. Losing one section is a smaller loss than losing the note.
    """
    projection = getattr(bundle, "projection", None)
    result = MatchupNote()

    try:
        result.on_paper = write_on_paper(bundle, voice=voice)
    except Exception:  # noqa: BLE001 -- a missing section is not a missing note
        pass
    try:
        result.on_the_mound = write_on_the_mound(
            bundle, home_starter=home_starter, away_starter=away_starter,
            home_faces=home_faces, away_faces=away_faces, voice=voice)
    except Exception:  # noqa: BLE001
        pass
    try:
        result.projections = write_projections(
            projection, bundle.home.abbreviation, bundle.away.abbreviation,
            voice=voice)
    except Exception:  # noqa: BLE001
        pass
    return result
