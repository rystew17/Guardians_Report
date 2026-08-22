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

from typing import Any

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
