"""Turning findings into sentences.

Templates, not generation. Each `code` carries several phrasings and the variant
is chosen by a stable hash of the subject, so a given player always reads the
same way, in this run and in every later one --
deterministic, and it avoids the detectable rhythm that fifty-three
identically-shaped sentences would create.

Every template states the number *and* what it is being compared against. "Elite
barrel rate" is an assertion; "barrels 14.2% of batted balls against a league
8.1%" is a finding the reader can check, which is the whole premise of this
report.
"""

from __future__ import annotations

import zlib

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
        "leaves the bat at {velocity:.1f} mph on average, against a league "
        "{mean:.1f}",
        "hits it {velocity:.1f} mph on average, where the league manages "
        "{mean:.1f}",
    ),
    "bat.profile.spray": (
        "pulls {pull:.0%} of what he puts in play, league {mean_pct:.0%}",
        "sends {pull:.0%} of his batted balls to the pull side, league "
        "{mean_pct:.0%}",
        "puts {pull:.0%} of his contact into the pull field, against "
        "{mean_pct:.0%} league-wide",
    ),
    "bat.approach.chase": (
        "chases {chase:.1%} of pitches out of the zone, league {mean_abs:.1%}",
        "offers at {chase:.1%} of what he sees outside, against {mean_abs:.1%} league-wide",
    ),
    "bat.approach.whiff": (
        "misses on {whiff:.1%} of his swings, league {mean_abs:.1%}",
        "swings through {whiff:.1%} of what he offers at, against "
        "{mean_abs:.1%} for the league",
        "comes up empty on {whiff:.1%} of his cuts, league {mean_abs:.1%}",
    ),
    "bat.approach.pitchtype": (
        "has handled the {pitch} at a {xwoba3} expected wOBA, league {league3}",
        "sees {seen} {pitch}s and has managed {xwoba3} against them, {league3} league",
    ),
    "bat.split.platoon": (
        "is a different hitter by hand: {vs_right3} against right-handers, "
        "{vs_left3} against left-handers",
        "splits hard by the hand he faces — {vs_right3} against righties, "
        "{vs_left3} against lefties",
        "hits {vs_right3} against right-handers and {vs_left3} against "
        "left-handers, which is two different hitters",
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
        "sits at {value3} across his last {games} times out, against "
        "{baseline3} for the year",
        "has been at {value3} over {games} recent outings, {baseline3} "
        "across the season as a whole",
    ),
    "pit.absent": (
        "has no recent record to grade — {seen} plate appearances in the window, "
        "so anything said about his form would be invented",
        "has faced {seen} hitters in the window, which is too few to read "
        "form from and too few to pretend otherwise",
        "has not pitched enough lately to grade — {seen} plate appearances, "
        "and a trend drawn from that would be invention",
    ),
    "pit.arsenal.best": (
        "leans on the {pitch} {usage:.0%} of the time and it earns it — a "
        "{whiff:.1%} whiff rate and {xwoba3} expected against, {league3} league-wide",
        # Only reach for "leans on" when the usage supports it. A pitch thrown
        # six percent of the time is a show-me offering, not a foundation, and
        # saying otherwise is the kind of small wrongness a reader notices.
        "gets results from the {pitch} — {xwoba3} expected against, {league3} "
        "for the league",
    ),
    # Both of these were complete clauses, and every fragment here has to be a
    # verb phrase that can follow a bare surname. "the Sinker is where he gets
    # hurt" rendered as "Yoho the Sinker is where he gets hurt", and "hitters
    # have found the Cutter" became "Cantillo hitters have found the Cutter" --
    # a sentence whose subject changes halfway through, because the clause
    # brought its own.
    "pit.arsenal.worst": (
        "gets hurt on the {pitch}: {xwoba3} expected against, {league3} league, "
        "and still throws it {usage:.0%} of the time",
        "is getting tagged on the {pitch} — {xwoba3} against a {league3} "
        "league mark",
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
    "game.projection": (
        "the model makes {favorite} a {probability:.1%} favorite, which is the "
        "{percentile_ord} percentile of how confident it ever gets",
        "{favorite} projects at {probability:.1%}, sitting at the {percentile_ord} "
        "percentile of this model's own range",
    ),
    "game.driver": (
        "what separates them is {label}, worth {size:.2f} log-odds toward {toward} "
        "and the largest of {count} inputs",
        "{label} does most of the work here — {size:.2f} log-odds toward {toward}",
    ),
    "game.score": (
        "the run model expects {home} {home_runs3}, {away} {away_runs3} — {total:.1f} "
        "on the night, with a {one_run:.0%} chance it comes down to one",
        "expected score {home} {home_runs3}, {away} {away_runs3}, and {one_run:.0%} of "
        "the time this is a one-run game",
    ),
    "game.first_five": (
        "through five the model has {home} ahead {home_leads:.0%} of the time, "
        "{away} {away_leads:.0%}, level the other {tied:.0%}",
        "after five innings it is {home} in front {home_leads:.0%} of the "
        "time and {away} {away_leads:.0%}, with {tied:.0%} still level",
        "the first five belong to {home} {home_leads:.0%} of the time and to "
        "{away} {away_leads:.0%}; the other {tied:.0%} are level",
    ),
    "game.strikeouts": (
        "{leader} projects for {leader_k:.1f} strikeouts against {trailer_k:.1f} for "
        "{trailer}, with a line at {leader_line:g}",
        "the model has {leader} down for {leader_k:.1f} strikeouts and "
        "{trailer} for {trailer_k:.1f}, against a posted {leader_line:g}",
        "{leader} figures for {leader_k:.1f} punchouts to {trailer_k:.1f} "
        "from {trailer}, with the number set at {leader_line:g}",
    ),
    "game.key_bat": (
        "{name} is the bat the model likes most tonight — a {homer:.0%} chance to go "
        "deep batting {slot} for {team}",
        "the model's favourite bat tonight is {name}, batting {slot} for "
        "{team} with a {homer:.0%} chance to leave the yard",
        "{name} carries the best home run chance on the card at {homer:.0%}, "
        "hitting {slot} for {team}",
    ),
    "game.starters": (
        "{better} sends the better starter by the model's reckoning",
        "the model gives {better} the edge in the pitching matchup",
        "on the mound the model prefers {better}",
    ),
}


def _variant_index(finding: Finding) -> int:
    """A stable choice of phrasing for this finding.

    Deliberately not the builtin `hash`. Python salts string hashing per
    process, so `hash(("game.driver", 0))` differs between runs of the same
    build -- which meant the same game produced a different sentence each time
    the report was generated. That is exactly the property this package was
    written to have and the reason the analysis stopped being generated, so it
    has to come from a hash that does not move.
    """
    return zlib.crc32(f"{finding.code}:{finding.subject}".encode())


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
    if "percentile" in detail:
        # "53th" is the kind of small wrongness that makes a reader stop
        # trusting the rest of the line.
        n = int(round(float(detail["percentile"])))
        suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        detail["percentile_ord"] = f"{n}{suffix}"
    # A heavily used pitch can carry the stronger phrasing; a rare one cannot.
    if finding.code == "pit.arsenal.best":
        detail["heavy"] = float(detail.get("usage", 0.0)) >= 0.20
    if finding.code == "game.driver":
        from guards_report.projections.predict import FEATURE_LABELS

        detail["label"] = FEATURE_LABELS.get(detail.get("name", ""), detail.get("name", "")).lower()
    if finding.code == "game.score":
        detail["home_runs3"] = f"{detail.get('home_runs', 0):.1f}"
        detail["away_runs3"] = f"{detail.get('away_runs', 0):.1f}"
    if finding.code == "game.starters":
        detail["better"] = detail["home"] if finding.value > 0 else detail["away"]
    return detail


def _rate(value: float) -> str:
    """Baseball's convention: three decimals, no leading zero."""
    text = format(float(value), ".3f")
    return text[1:] if text.startswith("0.") else text


# Which template the last render actually used, so the caller can keep two
# findings on one line from spending the same phrasing twice.
_LAST_INDEX = [0]


def render(finding: Finding, *, variant: int | None = None,
           avoid: frozenset[int] = frozenset()) -> str:
    """One sentence fragment for this finding, chosen deterministically.

    `variant` lets a caller vary the phrasing when two findings on one line
    would otherwise use the same template, which reads as a stutter.

    `avoid` carries the template indices already spent on this line. Counting
    occurrences was not enough: for a pitcher's best pitch the usage rule picks
    the phrasing rather than the count, so two rarely thrown offerings both
    landed on the same one and the line read "gets results from the Slider ...
    and gets results from the Curveball".
    """
    options = TEMPLATES.get(finding.code)
    if not options:
        return ""
    if finding.code == "pit.arsenal.best" and variant is None:
        # Usage decides the phrasing rather than the hash, so a rarely thrown
        # pitch never gets described as one he leans on.
        usage = float(finding.detail.get("usage", 0.0))
        index = 0 if usage >= 0.20 else 1
    else:
        index = variant if variant is not None else _variant_index(finding)

    index %= len(options)
    if index in avoid and len(avoid) < len(options):
        for step in range(1, len(options)):
            if (index + step) % len(options) not in avoid:
                index = (index + step) % len(options)
                break
    chosen = options[index]
    _LAST_INDEX[0] = index
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
    parts, spent = [], {}
    for finding in findings:
        used = spent.setdefault(finding.code, set())
        text = render(finding, avoid=frozenset(used))
        used.add(_LAST_INDEX[0])
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
