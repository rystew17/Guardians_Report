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

    ranked = sorted(
        (f for f in findings if f.significance >= threshold),
        key=lambda f: -f.significance,
    )

    chosen: list[Finding] = []
    families: set[str] = set()
    kinds: dict[str, int] = {}

    for finding in ranked:
        if len(chosen) >= limit:
            break
        if finding.family in families:
            continue
        if kinds.get(finding.kind, 0) >= per_kind:
            continue
        chosen.append(finding)
        families.add(finding.family)
        kinds[finding.kind] = kinds.get(finding.kind, 0) + 1

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
