"""Findings — computed observations, with the evidence that produced them.

The deliverable of this package is prose that no language model wrote. The hard
part is not the sentences; it is choosing which two of roughly forty computable
criteria matter tonight, which is the whole content of a scouting note and the
thing a model currently does implicitly.

Three ideas carry the design, and each was learned the expensive way elsewhere
in this project.

**Significance is distance from a reference, not magnitude.** ".312" is a number;
".312 against an expected .240 given his contact quality" is a finding. Every
criterion carries the population it was compared against, rebuilt each season,
never a constant -- fixed thresholds have already gone stale twice here.

**Reliability enters the ranking, not the sentence.** A .400 average over twenty
at-bats has to lose to a .340 over four hundred *before* selection, or the
report fills with September call-ups: extremes always live in the smallest
samples. The shrinkage is the same empirical-Bayes form the props use, with
stabilisation points measured by split-half reliability rather than assumed.

**Scanning many criteria and printing the extreme manufactures observations.**
With forty independent noise metrics the chance one clears the 97.5th percentile
is 87%, so every player on the card would receive a compelling and untrue
remark. The floor is derived from how many criteria are actually scanned, and it
moves when that count moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class Reference:
    """The population a value was measured against."""

    mean: float
    sd: float
    population: str
    n: int = 0

    def z(self, value: float) -> float:
        return float((value - self.mean) / self.sd) if self.sd else 0.0


@dataclass
class Finding:
    """One observation, and everything needed to judge whether to print it."""

    subject: int
    subject_kind: str                 # batter | pitcher | team | game
    code: str                         # stable id, e.g. "bat.power.barrel"
    family: str                       # power | contact | discipline | trend | ...
    kind: str                         # skill | weakness | trend | contrast | matchup

    value: float
    reference: Reference
    evidence: int = 0
    stabilisation: int = 100

    detail: dict[str, Any] = field(default_factory=dict)
    window: tuple[date, date] | None = None

    # Whether this finding claims something about the population, or merely
    # describes the subject.
    #
    # Phase B found this distinction the hard way. "His best pitch is the
    # sweeper, .199 expected wOBA against a .300 league" is descriptive: it is
    # true whatever the spread of the league, it makes no claim to be unusual,
    # and gating it behind a multiple-comparisons floor silenced every pitcher
    # on the card. "He is trending up" or "he is elite" is inferential -- it
    # asserts a departure from the population, and scanning forty criteria for
    # the most extreme one is precisely how that becomes false.
    #
    # The floor applies to the second kind. The first is ranked by how much it
    # departs from the reference, but printed on its own merits.
    inferential: bool = True

    @property
    def z_raw(self) -> float:
        return self.reference.z(self.value)

    @property
    def reliability(self) -> float:
        """How much of this player's own record to believe.

        The same n/(n+k) form the rate models use. A finding resting on twenty
        chances is mostly prior, and shrinking before ranking is what stops the
        smallest samples dominating the page.
        """
        return float(self.evidence / (self.evidence + self.stabilisation)) if (
            self.evidence + self.stabilisation
        ) else 0.0

    @property
    def z(self) -> float:
        """Reliability-shrunk distance from the reference. Rank on this."""
        return self.z_raw * self.reliability

    @property
    def significance(self) -> float:
        return abs(self.z)

    @property
    def direction(self) -> int:
        return 1 if self.z > 0 else -1

    @property
    def percentile(self) -> float:
        """Where the shrunk value sits, assuming a normal reference."""
        from math import erf, sqrt

        return float(50.0 * (1.0 + erf(self.z / sqrt(2.0))))


# An evaluator turns a subject's computed figures into zero or more findings.
# Zero is a valid and common answer: a replacement-level player having an
# unremarkable season should produce nothing, where a language model would
# always write a paragraph.
Evaluator = Callable[..., list[Finding]]


def significance_floor(criteria: int, *, expected_false: float = 0.5) -> float:
    """How large |z| must be before a finding is worth printing.

    Derived from how many criteria are scanned, so that fewer than
    `expected_false` findings per subject are expected by chance. At forty
    criteria this is about 2.5.

    Recompute whenever the evaluator count changes. Adding evaluators without
    raising the bar silently increases the false-finding rate, which is exactly
    how this fails while appearing to improve -- more findings, more of them
    wrong, and every one reading plausibly.
    """
    from scipy import stats

    criteria = max(int(criteria), 1)
    alpha = min(expected_false / criteria, 0.5)
    return float(stats.norm.ppf(1.0 - alpha / 2.0))
