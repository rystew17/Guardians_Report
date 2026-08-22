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
        "reaches at a {rate3} clip against a league {mean3}",
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
    "bat.profile.trajectory": (
        "puts {share:.0%} of his contact into {label}, league {mean_pct:.0%}",
        "{label} account for {share:.0%} of what he hits, against {mean_pct:.0%} league-wide",
    ),
    "bat.profile.barrel": (
        "barrels {rate:.1%} of his batted balls, league {mean_pct:.1%}",
        "squares one up {rate:.1%} of the time he makes contact, {mean_pct:.1%} league-wide",
    ),
    "bat.profile.exit_velocity": (
        "averages {velocity:.1f} mph off the bat, league {mean:.1f}",
    ),
    "bat.profile.spray": (
        "pulls {pull:.0%} of what he puts in play, league {mean_pct:.0%}",
    ),
    "bat.approach.chase": (
        "chases {chase:.1%} of pitches out of the zone, league {mean_abs:.1%}",
        "offers at {chase:.1%} of what he sees outside, against {mean_abs:.1%} league-wide",
    ),
    "bat.approach.whiff": (
        "misses on {whiff:.1%} of his swings, league {mean_abs:.1%}",
    ),
    "bat.approach.pitchtype": (
        "has handled the {pitch} at a {xwoba3} expected wOBA, league {league3}",
        "sees {seen} {pitch}s and has managed {xwoba3} against them, {league3} league",
    ),
    "bat.split.platoon": (
        "is a different hitter by hand: {vs_right3} against right-handers, "
        "{vs_left3} against left-handers",
    ),
    "bat.luck.gap": (
        "is {direction} his contact — {actual3} actual against {expected3} expected",
        "has a {actual3} line on {expected3} worth of contact, so it is {direction} him",
    ),
    "bat.trend.window": (
        "has hit {value3} over his last {games} games, against {baseline3} on the season",
        "is at {value3} across {games} games now, {baseline3} otherwise",
    ),
    "pit.trend.window": (
        "has run a {value3} rate over his last {games} outings, {baseline3} on the season",
    ),
    "pit.absent": (
        "has no recent record to grade — {seen} plate appearances in the window, "
        "so anything said about his form would be invented",
    ),
    "pit.arsenal.best": (
        "his {pitch} is the out pitch: a {whiff:.1%} whiff rate holding hitters to "
        "a {xwoba3} expected wOBA, against {league3} on the pitch league-wide",
        "leans on the {pitch} {usage:.0%} of the time and it earns it — "
        "{xwoba3} expected against, {league3} for the league",
    ),
    "pit.arsenal.worst": (
        "the {pitch} is where he gets hurt: {xwoba3} expected against, "
        "{league3} league, and he still throws it {usage:.0%} of the time",
        "hitters have found the {pitch}, tagging it for {xwoba3} against a "
        "{league3} league mark",
    ),
    "pit.arsenal.pitch": (
        "his {pitch} has held hitters to a {xwoba3} expected wOBA, league {league3} on the pitch",
        "throws the {pitch} {usage:.0%} of the time and it has been worth it: "
        "{xwoba3} expected against, league {league3}",
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
    if finding.code in ("bat.approach.chase", "bat.approach.whiff"):
        detail["mean_abs"] = abs(finding.reference.mean)
    # The reference in the same units the value is printed in.
    detail["mean_pct"] = abs(finding.reference.mean)
    if finding.code == "game.starters":
        detail["better"] = detail["home"] if finding.value > 0 else detail["away"]
    return detail


def _rate(value: float) -> str:
    """Baseball's convention: three decimals, no leading zero."""
    text = format(float(value), ".3f")
    return text[1:] if text.startswith("0.") else text


def render(finding: Finding, *, variant: int | None = None) -> str:
    """One sentence fragment for this finding, chosen deterministically.

    `variant` lets a caller vary the phrasing when two findings on one line
    would otherwise use the same template, which reads as a stutter.
    """
    options = TEMPLATES.get(finding.code)
    if not options:
        return ""
    index = variant if variant is not None else hash((finding.code, finding.subject))
    chosen = options[index % len(options)]
    variant_text = chosen
    slots = _slots(finding)
    for key in ("xwoba", "league", "value", "baseline", "mean", "rate",
                "actual", "expected", "vs_left", "vs_right", "gap", "share",
                "pull", "chase", "whiff"):
        if key in slots and isinstance(slots[key], (int, float)):
            slots[f"{key}3"] = _rate(float(slots[key]))
    try:
        return variant_text.format(**slots)
    except (KeyError, ValueError, TypeError):
        return ""


def sentence(findings: list[Finding], *, subject: str = "") -> str:
    """Join chosen findings into one readable line.

    Two findings become a sentence with a contrast; three get a semicolon. More
    than that reads as a list, which is what selection exists to prevent.
    """
    # Vary the phrasing when two findings share a template, so a line does not
    # repeat its own construction.
    parts, used = [], {}
    for finding in findings:
        seen = used.get(finding.code, 0)
        text = render(finding, variant=seen if seen else None)
        used[finding.code] = seen + 1
        if text:
            parts.append(text)
    if not parts:
        return ""

    def attach(fragment: str) -> str:
        """Join the subject to a fragment without producing "Smith his slider".

        Templates are written to follow a name directly, but some start with a
        possessive because they read better alone. Turning that into the
        subject's own possessive is the difference between a sentence and a
        stutter.
        """
        if not subject:
            return fragment
        if fragment.startswith("his "):
            return f"{subject}'s {fragment[4:]}"
        return f"{subject} {fragment}"

    lead = ""
    if len(parts) == 1:
        return attach(parts[0]) + "."
    if len(parts) == 2:
        joiner = ", but " if findings[0].direction != findings[1].direction else ", and "
        return attach(parts[0]) + joiner + parts[1] + "."
    return attach(parts[0]) + f"; {parts[1]}; {parts[2]}."
