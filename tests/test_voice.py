"""The vocabulary, and the two ways variety goes wrong.

Variety has a specific failure mode that repetition does not: it can quietly
change the claim. "Flawed" and "replacement level" are different statements about
a player, and a picker that treats them as interchangeable produces prose that
reads well and misinforms. So most of what follows is about the boundary between
phrasings that mean the same thing and phrasings that do not.

The second failure is subtler. Variety that comes from randomness destroys the
property this whole package exists for -- the same game yielding the same words
forever -- and it does it invisibly, because any single run looks fine.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from guards_report.insight import dossier, voice


# ---------------------------------------------------------------------------
# The claim never moves
# ---------------------------------------------------------------------------

def test_every_phrasing_of_a_band_is_reachable():
    """A phrasing nobody can draw is decoration in a source file."""
    for bands in (voice.VALUE_BANDS, voice.TOOL_BANDS):
        for _, name, phrasings in bands:
            drawn = {voice.choose(phrasings, name, i) for i in range(400)}
            assert drawn == set(phrasings), f"{name} cannot reach all of its wordings"


def test_a_players_worth_decides_the_band_not_the_wording():
    """Two runs values in one band may read differently; across bands they must not."""
    good = {voice.value_phrase(15.0, i)[1] for i in range(50)}
    poor = {voice.value_phrase(-15.0, i)[1] for i in range(50)}
    assert good == {"good"} and poor == {"poor"}
    assert not (set(voice.VALUE_BANDS[2][2]) & set(voice.VALUE_BANDS[6][2]))


def test_the_bands_are_ordered_and_do_not_overlap():
    for bands in (voice.VALUE_BANDS, voice.TOOL_BANDS):
        cuts = [cut for cut, _, _ in bands]
        assert cuts == sorted(cuts, reverse=True)
        seen: set[str] = set()
        for _, name, phrasings in bands:
            assert not (seen & set(phrasings)), f"{name} shares a wording with a neighbour"
            seen |= set(phrasings)


def test_an_mvp_and_a_replacement_player_can_never_read_alike():
    """The whole risk of this module in one assertion."""
    best = {voice.value_phrase(60.0, i)[0] for i in range(100)}
    worst = {voice.value_phrase(-40.0, i)[0] for i in range(100)}
    assert not (best & worst)


def test_a_value_on_a_boundary_takes_the_better_band():
    """Bands are inclusive at the bottom, so the cut is a promise not a range."""
    for cut, name, _ in voice.VALUE_BANDS[:-1]:
        assert voice.value_phrase(cut, 1)[1] == name


# ---------------------------------------------------------------------------
# Grammar — variety that does not parse is worse than repetition
# ---------------------------------------------------------------------------

def test_every_tool_wording_is_an_adjective():
    """These fill "<phrase> command", so a noun phrase breaks the sentence.

    The first version mixed the two and produced "a carrying tool command" and
    "as bad as it gets ground game". Nothing catches that but reading it.
    """
    for _, name, phrasings in voice.TOOL_BANDS:
        for phrase in phrasings:
            assert not phrase.startswith(("a ", "an ", "the ")), (
                f"{name}: {phrase!r} is a noun phrase in an adjective slot")


def test_every_value_wording_completes_the_sentence_it_is_used_in():
    """Each one follows "he is ...", so it carries its own article."""
    for _, name, phrasings in voice.VALUE_BANDS:
        for phrase in phrasings:
            assert phrase == phrase.lstrip(), f"{name}: {phrase!r} has stray space"
            assert not phrase.endswith("."), f"{name}: {phrase!r} carries punctuation"


def test_an_adjectival_profile_label_takes_no_article():
    """"A effectively wild" is not English, and neither is "a struggling"."""
    assert dossier._as_a("Effectively Wild") == "effectively wild"
    assert dossier._as_a("Struggling") == "struggling"


def test_a_noun_profile_label_takes_the_right_article():
    assert dossier._as_a("Command Artist") == "a command artist"
    assert dossier._as_a("On-Base Grinder") == "an on-base grinder"
    assert dossier._as_a("Air-Ball Chaser") == "an air-ball chaser"
    assert dossier._as_a("Innings Eater") == "an innings eater"


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------

def test_a_player_reads_the_same_way_every_time():
    first = voice.value_phrase(14.0, 12345)[0]
    assert all(voice.value_phrase(14.0, 12345)[0] == first for _ in range(50))


def test_the_wording_survives_a_fresh_process():
    """The bug this module was written knowing about.

    Python salts string hashing per process, so a picker built on the builtin
    `hash` gives the same player a different adjective on every build. A single
    test run shares one seed and cannot see it; subprocesses can.
    """
    script = (
        "from guards_report.insight import voice;"
        "print(voice.value_phrase(14.0, 12345)[0], '|',"
        "      voice.tool_phrase(88.0, 999, 'power')[0])"
    )
    runs = {
        subprocess.run([sys.executable, "-c", script], capture_output=True,
                       text=True, check=True).stdout.strip()
        for _ in range(4)
    }
    assert len(runs) == 1, f"wording moved between processes: {runs}"


def test_different_players_in_the_same_band_do_not_all_read_alike():
    """A stable hash that ignored the subject would be a constant."""
    drawn = {voice.value_phrase(14.0, player)[0] for player in range(60)}
    assert len(drawn) > 1


# ---------------------------------------------------------------------------
# Neighbours
# ---------------------------------------------------------------------------

def test_the_page_avoids_saying_the_same_thing_twice_in_a_row():
    """Three men in a row called "solid" reads as a template.

    The vocabulary alone does not prevent it -- a stable hash repeats often
    enough to notice -- so the card keeps a short memory of what it just said.
    """
    speaker = voice.Voice(memory=3)
    said = [speaker.tool(88.0, player, "power") for player in range(4)]
    assert len(set(said[:3])) == 3, f"repeated inside the memory window: {said}"


def test_the_memory_is_short_enough_not_to_exhaust_the_vocabulary():
    """Avoiding everything already used would force the picker back to the start.

    With more players than phrasings the words must recycle; what must not
    happen is an empty string or a crash when every option is blocked.
    """
    speaker = voice.Voice(memory=3)
    said = [speaker.tool(88.0, player, "power") for player in range(40)]
    assert all(said) and len(set(said)) > 1


def test_choose_returns_something_even_when_everything_is_blocked():
    options = ("one", "two")
    assert voice.choose(options, "k", avoid=options) in options


def test_no_options_yields_no_phrase_rather_than_an_error():
    assert voice.choose((), "k") == ""


# ---------------------------------------------------------------------------
# Matchup phrasings
# ---------------------------------------------------------------------------

def test_no_matchup_variant_asks_for_a_slot_the_writers_do_not_supply():
    """A variant needing an unsupplied slot is a silent blank.

    `matchup_phrase` swallows the KeyError and returns "", so the note simply
    loses a sentence -- no error, no warning, and only for the players who
    happen to draw that variant. A variant using *fewer* slots is fine; one
    reaching for a slot nobody passes is not.
    """
    import re

    supplied = {"p", "b", "hand"}
    for code, variants in voice.MATCHUP_PHRASES.items():
        for variant in variants:
            wanted = set(re.findall(r"\{(\w+)\}", variant))
            assert wanted <= supplied, (
                f"{code} wants {wanted - supplied}, which no writer passes")


def test_every_matchup_renders_with_the_slots_its_callers_supply():
    """Both writers pass p, b and hand; nothing may need more than those."""
    supplied = {"p": "Bibee", "b": "Ramirez", "hand": " against right-handers"}
    for code in voice.MATCHUP_PHRASES:
        for i in range(6):
            text = voice.matchup_phrase(code, i, **supplied)
            assert text, f"{code} produced nothing on variant {i}"
            assert "{" not in text, f"{code} left an unfilled slot: {text}"


def test_a_batter_phrasing_never_leaks_into_a_pitcher_matchup():
    """The two writers key off the prefix, so the sets must stay disjoint."""
    batter = {c for c in voice.MATCHUP_PHRASES if c.startswith("bat.")}
    pitcher = {c for c in voice.MATCHUP_PHRASES if c.startswith("pit.")}
    assert batter and pitcher
    assert batter | pitcher == set(voice.MATCHUP_PHRASES)


def test_an_unknown_matchup_code_is_silent_rather_than_fatal():
    assert voice.matchup_phrase("not.a.real.code", 1, p="X", b="Y") == ""


def test_the_same_pairing_reads_the_same_way_every_time():
    first = voice.matchup_phrase("bat.overmatched", 4242, p="A", b="B")
    assert all(
        voice.matchup_phrase("bat.overmatched", 4242, p="A", b="B") == first
        for _ in range(30)
    )
