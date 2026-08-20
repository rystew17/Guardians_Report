"""Every figure the model can read must be registered for verification.

If the two ever fall out of step the failure is quiet and corrosive: the model
cites a real number, the verifier cannot find it, and the report brands an
accurate sentence as unverified. A reader who sees that a few times stops
trusting the badge, which is worse than not having one.

This was a live defect. Groups nest -- `form` is {"L5": {"ERA": ...}} -- and
registration only walked the top level, so a starting pitcher registered 18 of
his 69 figures and 19 of 20 notes were flagged.
"""

from __future__ import annotations

from guards_report.analysis.digest import Digest
from guards_report.analysis.verify import NUMBER, verify


def test_nested_groups_are_registered():
    d = Digest(kind="pitcher", subject_id="p-1", label="Test")
    d.put_group("form", {
        "L5": {"ERA": "6.65", "K%": "18.7%", "HR/9": "1.57"},
        "season": {"ERA": "5.02"},
    })

    for figure in ("6.65", "18.7%", "1.57", "5.02"):
        assert figure in d.values, f"{figure} was shown to the model but not registered"


def test_combined_phrases_register_their_parts():
    # Prose cites "28.6%" alone, never the whole phrase.
    d = Digest(kind="pitcher", subject_id="p-1", label="Test")
    d.put_group("arsenal", {"Changeup": "19.8% usage, 28.6% whiff, .268 xwOBA"})

    for figure in ("19.8%", "28.6%", ".268"):
        assert figure in d.values


def test_every_number_in_the_payload_is_registered():
    """The invariant itself: scan the JSON the model receives, token by token."""
    d = Digest(kind="batter", subject_id="b-1", label="Test")
    d.put("name", "Someone")
    d.put_group("form", {"L15": {"OPS": ".774", "K%": "50.0%"}})
    d.put_group("splits", {"vs LHP": {"OPS": ".636", "PA": 23}})
    d.put_group("deep", {"a": {"b": {"c": "1.234"}}})

    missing = [t for t in NUMBER.findall(d.to_json()) if t not in d.values]
    assert not missing, f"visible to the model but unverifiable: {missing}"


def test_booleans_are_not_treated_as_figures():
    d = Digest(kind="pitcher", subject_id="p-1", label="Test")
    d.put("is_todays_probable_starter", True)
    assert "True" not in d.values


def test_prose_citing_registered_figures_passes_verification():
    d = Digest(kind="pitcher", subject_id="p-1", label="Test")
    d.put_group("form", {"L5": {"ERA": "6.65", "K%": "18.7%"}})

    ok = verify("His last five starts show a 6.65 ERA and an 18.7% strikeout rate.", d.values)
    assert ok.ok, ok.unverified

    bad = verify("His last five starts show a 2.11 ERA.", d.values)
    assert not bad.ok and "2.11" in bad.unverified


# ---------------------------------------------------------------------------
# Tokenizing prose
# ---------------------------------------------------------------------------


def test_hyphen_between_figures_is_not_a_minus_sign():
    """"21-22%" is a range, not twenty-one and negative twenty-two percent."""
    assert NUMBER.findall("from 21-22% over his L15/L30") == ["21", "22%"]
    assert NUMBER.findall("both sub-.28 xwOBA") == [".28"]
    assert NUMBER.findall("17-18% whiff") == ["17", "18%"]


def test_genuine_signs_still_parse():
    # Benchmark deltas are written with an explicit sign and must survive.
    assert NUMBER.findall("a delta of -.28 and +.246") == ["-.28", "+.246"]


def test_trailing_zeros_are_formatting_not_a_new_claim():
    values = {"50.0%", "22.1%"}
    assert verify("a 50% strikeout rate", values).ok


def test_rounding_is_still_flagged():
    """22% is not 22.1%. Approximation is a different claim, and stays caught."""
    result = verify("about 22% of the time", {"22.1%"})
    assert not result.ok and "22%" in result.unverified


def test_fabricated_figures_are_still_caught():
    result = verify("a 4.10 ERA and a .312 average", {"3.25", ".268"})
    assert set(result.unverified) == {"4.10", ".312"}
