"""Choosing which findings to print, and how many.

Selection is the part that decides whether this reads as insight or as a
generated list. Fifty-three players each given three correct, interchangeable
sentences is the failure mode, and no amount of template variety fixes it.

Two rules do most of the work. Nothing below the significance floor is printed
at all, because a criterion that fails it is more likely noise than news. And no
two chosen findings may share a family, so a player who is good at everything
yields his two most *distinctive* traits rather than eight restatements of the
same fact.

Returning nothing is a valid outcome and should stay one.
"""

from __future__ import annotations

from guards_report.insight.types import Finding, significance_floor


def select(
    findings: list[Finding],
    *,
    limit: int = 3,
    criteria: int | None = None,
    floor: float | None = None,
    per_kind: int = 2,
    min_reliability: float = 0.30,
) -> list[Finding]:
    """The most significant findings, spread across different families.

    `criteria` is the number of criteria that were actually scanned for this
    subject, which sets the floor. Passing the number of findings that survived
    would understate it -- the multiple-comparisons cost is paid for every
    criterion tested, not every one that happened to clear.
    """
    if not findings:
        return []

    threshold = floor if floor is not None else significance_floor(
        criteria if criteria is not None else len(findings)
    )

    # The floor is a guard against claiming an unusual departure that is really
    # noise. A descriptive finding claims no such thing -- a pitcher's best pitch
    # is his best pitch whatever the league spread -- so it is ranked alongside
    # the rest but not gated.
    # Two different gates. The floor stops an inferential claim that is really
    # noise. The reliability gate stops a descriptive one that is technically
    # true and materially misleading -- a .053 expected wOBA over forty pitches
    # is a real number and says nothing about the hitter.
    eligible = [
        f for f in findings
        if f.reliability >= min_reliability
        and (not f.inferential or f.significance >= threshold)
    ]
    ranked = sorted(eligible, key=lambda f: -f.significance)

    chosen: list[Finding] = []
    families: set[str] = set()
    kinds: dict[str, int] = {}

    def take(finding: Finding) -> bool:
        if len(chosen) >= limit:
            return False
        if finding.family in families:
            return False
        if kinds.get(finding.kind, 0) >= per_kind:
            return False
        chosen.append(finding)
        families.add(finding.family)
        kinds[finding.kind] = kinds.get(finding.kind, 0) + 1
        return True

    # Lead with the strongest finding, then deliberately reach for one pointing
    # the other way. Two strengths read as a list; a strength and a weakness
    # read as a scouting note, because the tension is the information -- what he
    # does well is only useful next to where he can be got at.
    if ranked:
        take(ranked[0])
        opposite = next(
            (f for f in ranked[1:] if f.direction != ranked[0].direction), None
        )
        if opposite is not None:
            take(opposite)

    for finding in ranked:
        if len(chosen) >= limit:
            break
        take(finding)

    return chosen


def contrasts(findings: list[Finding], *, floor: float = 1.5) -> list[tuple[Finding, Finding]]:
    """Pairs that point opposite ways — the findings worth the most.

    A single statistic is inert. What reads as insight is tension: elite exit
    velocity next to a bottom-decile launch angle, a strikeout rate in the 90th
    percentile next to a walk rate in the 15th. None of these are visible to a
    per-metric scan, because neither half is remarkable enough on its own to be
    selected.

    Deliberately a lower floor than `select` uses. A contrast carries more
    information than either of its halves, so it earns its place on weaker
    individual evidence.
    """
    strong = [f for f in findings if f.significance >= floor]
    pairs = []
    for i, first in enumerate(strong):
        for second in strong[i + 1:]:
            if first.family == second.family:
                continue
            if first.direction == second.direction:
                continue
            pairs.append((first, second))

    # Widest tension first: the pair that most contradicts itself.
    pairs.sort(key=lambda pair: -(abs(pair[0].z - pair[1].z)))
    return pairs
