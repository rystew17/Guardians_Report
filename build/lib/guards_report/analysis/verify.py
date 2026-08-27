"""Check that written analysis only cites numbers the report actually contains.

The project's rule is that every figure is sourced or computed, never
generated. Adding a written layer puts that at risk in one specific way: a
model can produce fluent prose containing a statistic that was never in the
data. This module closes that gap by treating the written output as untrusted
and checking it.

Every numeric token in the prose is extracted and matched against the values in
the digest the model was given. A figure that does not appear there was not
sourced, and is reported.

The check is deliberately asymmetric. Missing a real problem is far worse than
flagging a harmless one, so matching is exact and the burden is on the prose to
justify its numbers -- with narrow, explicit allowances for counts a writer
legitimately produces from the data (ordinals, small integers) rather than a
loose tolerance that would let a wrong batting average through.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Numbers as the report writes them: .305, 1.034, 22.1%, +.246, 3.25, 12
#
# The lookbehind stops a hyphen joining two things from being read as a minus
# sign. Without it "21-22%" tokenizes as 21 and *negative* 22%, and "sub-.28"
# as negative .28 -- so accurate prose gets flagged, and the flag names a
# number that never appeared. A sign only counts when the token starts cleanly.
NUMBER = re.compile(r"(?<![\w.])[+-]?(?:\d+\.\d+|\.\d+|\d+)%?")

# Small integers a writer legitimately produces by counting things already in
# the digest -- "his last 5 games", "3 of 4 pitches", "the 2 hole". Anything
# larger, or any decimal or percentage, must appear in the digest verbatim.
MAX_FREE_INTEGER = 30

# Years and similar. A season reference is not a statistical claim.
CONTEXT_INTEGERS = frozenset({"2024", "2025", "2026", "2027"})


@dataclass
class Verification:
    """The result of checking one analysis against its digest."""

    checked: int = 0
    unverified: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unverified

    @property
    def summary(self) -> str:
        if self.ok:
            return f"{self.checked} figures checked, all traced to source data"
        return (
            f"{self.checked} figures checked, "
            f"{len(self.unverified)} not found in source data: "
            + ", ".join(self.unverified[:5])
        )


def _variants(token: str) -> set[str]:
    """The forms a single figure might legitimately take in prose.

    The report writes a batting average as `.305`, but a writer may render the
    same number as `0.305`; a percentage as `22.1%` may be written `22.1`.
    These are the same claim, so they must not be flagged. Anything beyond
    re-formatting the identical digits is a different number.
    """
    forms = {token}
    bare = token.lstrip("+-")
    forms.add(bare)

    if bare.endswith("%"):
        forms.add(bare[:-1])
    else:
        forms.add(bare + "%")

    if bare.startswith("."):
        forms.add("0" + bare)
    elif bare.startswith("0."):
        forms.add(bare[1:])

    # Trailing zeros are formatting, not a different claim: the report prints
    # "50.0%" and a writer reasonably types "50%". Only forms that are exactly
    # equal in value are added -- 22 never becomes 22.1.
    forms |= _equal_value_forms(bare)

    return forms


def _equal_value_forms(bare: str) -> set[str]:
    """Renderings of the identical value at other decimal precisions."""
    digits = bare[:-1] if bare.endswith("%") else bare
    suffix = "%" if bare.endswith("%") else ""
    try:
        number = float(digits)
    except ValueError:
        return set()

    forms = set()
    for places in range(4):
        text = f"{number:.{places}f}"
        # Requiring exact round-trip equality is what keeps this from becoming
        # a tolerance: .305 and .3 are different claims and must stay so.
        if float(text) != number:
            continue
        forms.add(text + suffix)
        if text.startswith("0."):
            forms.add(text[1:] + suffix)
    return forms


def _digest_numbers(values: set[str]) -> set[str]:
    """Every numeric token appearing anywhere in the digest's values."""
    found: set[str] = set()
    for value in values:
        for match in NUMBER.findall(str(value)):
            found |= _variants(match)
    return found


def verify(prose: str, digest_values: set[str]) -> Verification:
    """Confirm every number in `prose` traces to a value in the digest."""
    available = _digest_numbers(digest_values)
    result = Verification()

    for token in NUMBER.findall(prose):
        bare = token.lstrip("+-")

        # Context integers and small counts are allowed without a match.
        if bare in CONTEXT_INTEGERS:
            continue
        if bare.isdigit() and int(bare) <= MAX_FREE_INTEGER:
            continue

        result.checked += 1
        if not (_variants(token) & available):
            result.unverified.append(token)

    return result
