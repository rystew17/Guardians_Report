"""The deterministic analysis machinery.

This package exists to replace generated prose with computed findings, and its
whole risk is that it produces *plausible* observations that are not true. The
tests below are therefore mostly about restraint: that small samples lose, that
noise stays quiet, and that saying nothing remains a valid outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from guards_report.insight import evaluators, render, select
from guards_report.insight.types import Finding, Reference, significance_floor


def _finding(**kwargs) -> Finding:
    base = dict(
        subject=1, subject_kind="batter", code="bat.rate.hit", family="contact",
        kind="skill", value=0.30,
        reference=Reference(mean=0.25, sd=0.03, population="league"),
        evidence=500, stabilisation=100,
    )
    base.update(kwargs)
    return Finding(**base)


# --------------------------------------------------------------------------
# The multiple-comparisons floor
# --------------------------------------------------------------------------

def test_the_floor_rises_with_the_number_of_criteria():
    """Scanning more criteria must cost more to clear.

    With forty independent noise metrics the chance one clears the 97.5th
    percentile is 87%. Adding evaluators without raising the bar is how this
    fails while appearing to improve.
    """
    assert significance_floor(5) < significance_floor(20) < significance_floor(44)
    assert significance_floor(44) == pytest.approx(2.53, abs=0.02)


def test_the_floor_keeps_expected_false_findings_below_one_in_two():
    from scipy import stats

    for criteria in (5, 20, 44, 60):
        z = significance_floor(criteria)
        expected = criteria * 2 * (1 - stats.norm.cdf(z))
        assert expected <= 0.51


# --------------------------------------------------------------------------
# Reliability shrinkage
# --------------------------------------------------------------------------

def test_a_small_sample_is_shrunk_below_a_large_one():
    """A .400 over twenty at-bats must lose to a .340 over four hundred.

    Extremes live in the smallest samples, so ranking on the raw distance would
    fill the page with September call-ups.
    """
    hot = _finding(value=0.400, evidence=20, stabilisation=475)
    steady = _finding(value=0.340, evidence=400, stabilisation=475)
    assert hot.z_raw > steady.z_raw, "raw distance favours the small sample"
    assert steady.significance > hot.significance, "shrinkage must reverse it"


def test_reliability_is_bounded_and_monotone():
    previous = -1.0
    for evidence in (0, 10, 100, 1000, 10_000):
        r = _finding(evidence=evidence, stabilisation=100).reliability
        assert 0.0 <= r < 1.0
        assert r > previous
        previous = r


def test_a_metric_that_stabilises_faster_keeps_more_of_its_signal():
    """Strikeout rate settles at 55 plate appearances, hit rate at 475."""
    strikeouts = _finding(evidence=300, stabilisation=55)
    hits = _finding(evidence=300, stabilisation=475)
    assert strikeouts.reliability > hits.reliability


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def test_nothing_is_printed_when_nothing_clears_the_floor():
    """An unremarkable player should yield silence, not a paragraph."""
    quiet = [_finding(value=0.252, evidence=400) for _ in range(5)]
    assert select.select(quiet, criteria=5) == []


def test_selection_refuses_two_findings_from_one_family():
    """A player good at everything gets his two most distinctive traits."""
    findings = [
        _finding(value=0.40, family="power", code="a", evidence=900),
        _finding(value=0.39, family="power", code="b", evidence=900),
        _finding(value=0.38, family="contact", code="c", evidence=900),
    ]
    chosen = select.select(findings, criteria=5, limit=3)
    assert len({f.family for f in chosen}) == len(chosen)


def test_selection_is_ordered_by_shrunk_significance():
    findings = [
        _finding(value=0.30, evidence=100, code="small"),
        _finding(value=0.30, evidence=2000, family="power", code="large"),
    ]
    chosen = select.select(findings, criteria=2, limit=2)
    assert chosen[0].code == "large"


def test_the_floor_comes_from_criteria_scanned_not_findings_surviving():
    """The multiple-comparisons cost is paid per criterion tested.

    Passing the surviving count would understate it, and understating it is the
    whole failure mode.
    """
    findings = [_finding(value=0.31, evidence=800)]
    assert select.select(findings, criteria=1, limit=1)
    assert select.select(findings, criteria=200, limit=1) == []


# --------------------------------------------------------------------------
# Contrasts
# --------------------------------------------------------------------------

def test_contrasts_pair_findings_that_point_opposite_ways():
    """Tension is the interesting shape; agreement is not."""
    good = _finding(value=0.34, family="power", evidence=900)
    bad = _finding(value=0.16, family="discipline", evidence=900, code="k")
    pairs = select.contrasts([good, bad], floor=1.0)
    assert pairs and pairs[0][0].direction != pairs[0][1].direction


def test_two_findings_in_the_same_family_are_not_a_contrast():
    a = _finding(value=0.34, family="power", evidence=900)
    b = _finding(value=0.16, family="power", evidence=900, code="b")
    assert select.contrasts([a, b], floor=1.0) == []


# --------------------------------------------------------------------------
# Change-point trends
# --------------------------------------------------------------------------

def test_a_real_change_is_found_with_its_window():
    rng = np.random.default_rng(3)
    series = pd.Series(np.concatenate([
        rng.normal(0.200, 0.15, 60), rng.normal(0.400, 0.15, 30),
    ]))
    found = evaluators.longest_significant_window(series, baseline=0.250)
    assert found is not None
    length, mean, p = found
    assert length >= 15 and mean > 0.30 and p < 0.01


def test_noise_around_the_baseline_produces_no_trend():
    rng = np.random.default_rng(5)
    flat = pd.Series(rng.normal(0.250, 0.15, 120))
    assert evaluators.longest_significant_window(flat, baseline=0.250) is None


def test_false_trends_stay_rare_across_many_players():
    """Trying L5, L10, L15 and L30 and printing the best always finds something.

    Correcting for how many windows were tried is what keeps this honest, and it
    holds the false rate well under the nominal alpha.
    """
    rng = np.random.default_rng(9)
    fired = sum(
        evaluators.longest_significant_window(
            pd.Series(rng.normal(0.250, 0.15, 110)), baseline=0.250
        ) is not None
        for _ in range(200)
    )
    assert fired / 200 <= 0.03


def test_too_short_a_series_yields_nothing():
    assert evaluators.longest_significant_window(
        pd.Series([0.3, 0.4, 0.2]), baseline=0.25
    ) is None


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def test_a_sentence_states_the_number_and_its_reference():
    finding = _finding(
        code="bat.rate.strikeout", family="discipline",
        value=0.10, reference=Reference(mean=0.22, sd=0.04, population="league"),
        evidence=500, stabilisation=55, detail={"rate": 0.10},
    )
    text = render.render(finding)
    assert "10.0%" in text and "22.0%" in text


def test_the_same_player_always_reads_the_same_way():
    finding = _finding(code="bat.rate.strikeout", detail={"rate": 0.10})
    assert render.render(finding) == render.render(finding)


def test_an_unknown_code_renders_nothing_rather_than_raising():
    assert render.render(_finding(code="not.a.real.code")) == ""


def test_two_findings_pointing_opposite_ways_are_joined_with_a_contrast():
    """Built the way `rate_quality` builds them: signed so positive means better.

    A batter's strikeout rate is negated, because a high one is a weakness. That
    sign is what makes a contrast detectable at all -- without it both findings
    read as strengths and the pair is invisible.
    """
    good = _finding(code="bat.rate.hit", detail={"rate": 0.31}, value=0.31)
    bad = _finding(
        code="bat.rate.strikeout", family="discipline", value=-0.30,
        reference=Reference(mean=-0.22, sd=0.04, population="league"),
        detail={"rate": 0.30},
    )
    assert good.direction != bad.direction
    text = render.sentence([good, bad], subject="He")
    assert text.startswith("He ") and ", but " in text


def test_no_findings_produce_no_sentence():
    assert render.sentence([]) == ""

def test_the_same_finding_reads_the_same_way_in_a_fresh_process():
    """The whole reason the analysis stopped being generated.

    Phrasing is chosen from a hash of the finding, and Python salts string
    hashing per process -- so the builtin `hash` gave the same game a different
    sentence on every build. It looked deterministic because a single test run
    shares one seed. Subprocesses are the only way to see it.
    """
    import subprocess
    import sys

    script = (
        "from guards_report.insight import render;"
        "from guards_report.insight.types import Finding, Reference;"
        "print(render.render(Finding("
        "subject=12345, subject_kind='batter', code='bat.rate.hit', family='contact',"
        "kind='skill', value=0.31,"
        "reference=Reference(mean=0.25, sd=0.03, population='league'),"
        "evidence=500, stabilisation=100, detail={'rate': 0.31})))"
    )
    runs = {
        subprocess.run([sys.executable, "-c", script], capture_output=True,
                       text=True, check=True).stdout.strip()
        for _ in range(4)
    }
    assert len(runs) == 1, f"phrasing moved between processes: {runs}"


# ---------------------------------------------------------------------------
# Scope: the profile describes a season, and says so
# ---------------------------------------------------------------------------

class _Box:
    def __init__(self, career=None, season=None):
        self.career = career or {}
        self.season = season or {}


def _profiled(runs=25.0, sample=500, thin=False, kind="batter"):
    from guards_report.insight import profile as prof
    return prof.PlayerProfile(
        player_id=1, kind=kind, runs_per_150=runs, bat_per_150=runs,
        bat_grade="a serious bat", tier="a very good player",
        sample=sample, thin=thin, position="CF")


def test_the_profile_states_a_season_not_an_identity():
    """The error a reader spotted before the code did.

    Every grade here is computed from the current season, and the present tense
    promoted four months of baseball into a claim about the man -- a hitter with
    a first-ballot career reads as "an average bat" when the sentence says "is".
    The measurement was right; the tense was the bug.
    """
    from guards_report.insight import dossier

    said = dossier.write_profile(_profiled(), surname="Trout", box=_Box())
    assert " is a " not in said and " is an " not in said, said

    # The rule, not one phrasing of it: whichever frame was drawn, it has to
    # date the claim. Asserting a literal "this season" would fail the moment
    # the picker chose "On the year", which is equally scoped and equally fine.
    from guards_report.insight import voice
    assert any(marker in said.lower()
               for marker in ("this season", "this year", "on the year")), said


def test_every_season_frame_dates_its_claim():
    """The frames vary so twenty-six profiles do not open identically. Each one
    still has to say *when*, or the variation reintroduces the bug it is
    decorating."""
    from guards_report.insight import voice

    for template in voice.SEASON_FRAMES:
        said = template.format(name="X", tier="a very good player").lower()
        assert any(m in said for m in ("this season", "this year", "on the year")), template
        # Perfect tense, never "X is a very good player".
        assert " is a " not in said and " is an " not in said, template


def test_a_newcomer_is_named_as_one_rather_than_shrugged_at():
    """A rookie in his second week and a veteran off the injured list used to
    get the same line. They are opposite situations."""
    from guards_report.insight import dossier

    rookie = _Box(career={"atBats": 61}, season={"atBats": 61})
    said = dossier.write_profile(
        _profiled(sample=68, thin=True), surname="Delauter", box=rookie)
    assert "new to the majors" in said
    assert "should be read as a verdict" in said


def test_a_veteran_in_limited_action_is_not_called_a_rookie():
    """The failure that matters: 4,200 career at-bats is not a newcomer, and
    saying so about a twelve-year veteran would discredit the whole page."""
    from guards_report.insight import dossier

    veteran = _Box(career={"atBats": 4260}, season={"atBats": 38})
    said = dossier.write_profile(
        _profiled(sample=41, thin=True), surname="Ramirez", box=veteran)
    assert "new to the majors" not in said
    assert "career at-bats" in said


def test_career_totals_have_this_season_subtracted_back_out():
    """Career includes the current year, so a rookie having a big season would
    otherwise look like a player with a track record -- backwards."""
    from guards_report.insight import dossier

    box = _Box(career={"atBats": 400}, season={"atBats": 380})
    assert dossier.service_before_this_season(box, "batter") == 20.0


def test_a_missing_career_block_falls_back_rather_than_guessing():
    from guards_report.insight import dossier

    assert dossier.service_before_this_season(_Box(), "batter") is None
    said = dossier.write_profile(
        _profiled(sample=30, thin=True), surname="Nobody", box=_Box())
    assert "too little to profile" in said


def test_a_short_career_is_not_offered_as_evidence():
    """The gap between the rookie line and a real track record.

    163 career at-bats clears MLB's rookie threshold and still says almost
    nothing. Telling a reader it "says more about him" than 39 this season
    promises evidence that does not exist -- it is a second small sample, not
    a bigger one.
    """
    from guards_report.insight import dossier

    box = _Box(career={"atBats": 202}, season={"atBats": 39})
    said = dossier.write_profile(
        _profiled(sample=44, thin=True), surname="Moore", box=box)
    assert "still establishing himself" in said
    assert "say more about him" not in said
    assert "new to the majors" not in said


def test_a_thin_sample_is_caveated_not_withheld():
    """A hitter who has hammered sixty-three plate appearances is worth reading
    about; the reader simply has to be told it is sixty-three.

    What stays withheld is the verdict -- the tier and the archetype -- because
    a runs-per-150 figure extrapolated from sixteen games multiplies its own
    noise by nine.
    """
    from guards_report.insight import dossier, profile as prof

    tool = prof.Tool(name="hard", label="contact quality", grade=94.0, z=1.9)
    player = prof.PlayerProfile(
        player_id=7, kind="batter", runs_per_150=140.0, bat_per_150=140.0,
        bat_grade="an elite bat", tier="an MVP-caliber player",
        sample=63, thin=True, position="SS", tools={"hard": tool})

    said = dossier.write_profile(
        player, surname="Genao", box=_Box(career={"atBats": 4}, season={"atBats": 4}))

    assert "new to the majors" in said            # the caveat leads
    assert "contact quality" in said              # what he has done still lands
    assert "94th percentile" in said
    assert "MVP-caliber" not in said              # the verdict does not
    assert "runs per 150" not in said             # nor the extrapolation


def test_a_caveat_followed_by_evidence_does_not_double_its_full_stop():
    """Pieces are joined with ". " and the closer adds one, so any piece that
    punctuates itself lands a double. Visible on the card as "player.. Elite"."""
    from guards_report.insight import dossier, profile as prof

    tool = prof.Tool(name="glove", label="defense", grade=95.0, z=2.0)
    player = prof.PlayerProfile(
        player_id=9, kind="batter", sample=68, thin=True, tools={"glove": tool})
    said = dossier.write_profile(
        player, surname="Genao", box=_Box(career={"atBats": 4}, season={"atBats": 4}))
    assert ".." not in said, said


# ---------------------------------------------------------------------------
# Which tools get to speak
# ---------------------------------------------------------------------------

def _tools(**grades):
    from guards_report.insight import profile as prof
    return {name: prof.Tool(name=name, label=name.replace("_", " "),
                            grade=g, z=(g - 50) / 20.0)
            for name, g in grades.items()}


def _batter(**grades):
    from guards_report.insight import profile as prof
    return prof.PlayerProfile(
        player_id=3, kind="batter", runs_per_150=12.0, bat_per_150=12.0,
        bat_grade="a plus bat", tier="a solid regular", position="SS",
        sample=500, tools=_tools(**grades))


def test_the_bat_is_always_described_for_a_hitter():
    """A card that led with a shortstop's glove and never mentioned his bat
    answered a question nobody asked. He is in the lineup to hit."""
    from guards_report.insight import dossier

    # A spectacular glove and an ordinary bat: the glove must not crowd it out.
    player = _batter(defense=97.0, power=58.0, contact=61.0, loft=44.0)
    said = dossier.write_profile(player, surname="Genao", box=_Box())
    assert any(w in said for w in ("bat-to-ball", "power", "contact")), said


def test_an_ordinary_glove_is_not_worth_a_sentence():
    """Only the tails. A 60th-percentile glove says nothing a reader needs."""
    from guards_report.insight import dossier

    player = _batter(defense=60.0, power=92.0, contact=48.0)
    said = dossier.write_profile(player, surname="X", box=_Box())
    assert "defense" not in said


def test_a_glove_speaks_at_either_tail():
    from guards_report.insight import dossier

    for grade in (96.0, 6.0):
        player = _batter(defense=grade, power=70.0, contact=50.0)
        said = dossier.write_profile(player, surname="X", box=_Box())
        assert "defense" in said, (grade, said)


def test_running_speaks_only_when_it_is_a_weapon():
    """One-tailed, and not a new idea -- BASERUNNING_FLOOR already says nobody
    describes a player's weakness as not attempting steals."""
    from guards_report.insight import dossier

    slow = _batter(speed=4.0, baserunning=8.0, power=88.0, contact=50.0)
    said = dossier.write_profile(slow, surname="X", box=_Box())
    assert "speed" not in said and "baserunning" not in said, said

    burner = _batter(speed=97.0, power=88.0, contact=50.0)
    said = dossier.write_profile(burner, surname="X", box=_Box())
    assert "speed" in said


def test_a_pitcher_still_shows_his_best_and_his_worst():
    """The rule is about hitters. Every tool a pitcher has is his craft."""
    from guards_report.insight import dossier, profile as prof

    player = prof.PlayerProfile(
        player_id=4, kind="pitcher", runs_per_150=10.0, tier="a solid regular",
        sample=500, tools=_tools(stuff=93.0, command=9.0))
    said = dossier.write_profile(player, surname="Cantillo", box=_Box())
    assert "stuff" in said and "command" in said
