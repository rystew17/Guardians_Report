"""Turning findings into sentences.

Templates, not generation. Each `code` carries several phrasings and the variant
is chosen by hashing the subject, so a given player always reads the same way --
deterministic, and it avoids the detectable rhythm that fifty-three
identically-shaped sentences would create.

Every template states the number *and* what it is being compared against. "Elite
barrel rate" is an assertion; "barrels 14.2% of batted balls against a league
8.1%" is a finding the reader can check, which is the whole premise of this
report.
"""

from __future__ import annotations

from guards_report.insight.types import Finding

# Phrasings per finding code. The `{}` slots are filled from `detail` plus a few
# derived values, so a template can never reference a figure the finding does
# not carry.
TEMPLATES: dict[str, tuple[str, ...]] = {
    "bat.rate.hit": (
        "reaches at a {rate:.3f} clip against a league {mean:.3f}",
        "gets a hit in {rate:.1%} of plate appearances, league {mean:.1%}",
    ),
    "bat.rate.home_run": (
        "homers once every {per_hr:.0f} plate appearances, league once every {per_lg:.0f}",
        "goes deep in {rate:.1%} of trips, against a league {mean:.1%}",
    ),
    "bat.rate.strikeout": (
        "strikes out {rate:.1%} of the time, league {mean:.1%}",
        "goes down on strikes in {rate:.1%} of plate appearances, league {mean:.1%}",
    ),
    "bat.trend.window": (
        "has hit {value:.3f} over his last {games} games, against {baseline:.3f} on the season",
        "is at {value:.3f} across {games} games now, {baseline:.3f} otherwise",
    ),
    "pit.trend.window": (
        "has run a {value:.3f} rate over his last {games} outings, {baseline:.3f} on the season",
    ),
    "pit.arsenal.pitch": (
        "his {pitch} has held hitters to a {xwoba:.3f} expected wOBA, league {league:.3f} on the pitch",
        "throws the {pitch} {usage:.0%} of the time and it has been worth it: "
        "{xwoba:.3f} expected against, league {league:.3f}",
    ),
    "pit.weak.order": (
        "falls from {first_pass:.1%} strikeouts the first time through to "
        "{late_pass:.1%} later, a steeper drop than the league's",
        "loses more than most on repeat looks: {first_pass:.1%} down to {late_pass:.1%}",
    ),
    "game.starters": (
        "{better} sends the better starter by the model's reckoning",
    ),
}


def _slots(finding: Finding) -> dict:
    detail = dict(finding.detail)
    detail.setdefault("value", finding.value)
    detail.setdefault("mean", finding.reference.mean)
    detail.setdefault("rate", detail.get("rate", finding.value))

    if finding.code == "bat.rate.home_run":
        rate = max(detail.get("rate", 0.0), 1e-6)
        mean = max(finding.reference.mean, 1e-6)
        detail["per_hr"] = 1.0 / rate
        detail["per_lg"] = 1.0 / mean
    if finding.code == "game.starters":
        detail["better"] = detail["home"] if finding.value > 0 else detail["away"]
    return detail


def render(finding: Finding) -> str:
    """One sentence fragment for this finding, chosen deterministically."""
    options = TEMPLATES.get(finding.code)
    if not options:
        return ""
    variant = options[hash((finding.code, finding.subject)) % len(options)]
    try:
        return variant.format(**_slots(finding))
    except (KeyError, ValueError, TypeError):
        return ""


def sentence(findings: list[Finding], *, subject: str = "") -> str:
    """Join chosen findings into one readable line.

    Two findings become a sentence with a contrast; three get a semicolon. More
    than that reads as a list, which is what selection exists to prevent.
    """
    parts = [p for p in (render(f) for f in findings) if p]
    if not parts:
        return ""

    lead = f"{subject} " if subject else ""
    if len(parts) == 1:
        return f"{lead}{parts[0]}."
    if len(parts) == 2:
        joiner = ", but " if findings[0].direction != findings[1].direction else ", and "
        return f"{lead}{parts[0]}{joiner}{parts[1]}."
    return f"{lead}{parts[0]}; {parts[1]}; {parts[2]}."
