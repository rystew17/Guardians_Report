"""Nobody on the active roster may be silently dropped.

A missing player does not look like a bug -- the page just has one fewer box --
so this is checked rather than eyeballed. The live case: the source types a
designated hitter as "Hitter", which was not in the accepted set, so Bryce
Eldridge disappeared from a Giants report.
"""

from __future__ import annotations

from guards_report.ingest.preview import _split_roster


def _roster(*types):
    return {"roster": [
        {"person": {"id": i, "fullName": f"P{i}"}, "position": {"type": t}}
        for i, t in enumerate(types, start=1)
    ]}


def test_designated_hitter_is_a_position_player():
    batters, pitchers = _split_roster(_roster("Hitter"))
    assert len(batters) == 1 and not pitchers


def test_every_roster_entry_is_accounted_for():
    types = ["Pitcher", "Catcher", "Infielder", "Outfielder", "Hitter"]
    batters, pitchers = _split_roster(_roster(*types))
    assert len(batters) + len(pitchers) == len(types)


def test_unknown_position_type_is_kept_not_dropped():
    batters, pitchers = _split_roster(_roster("Something New"))
    assert len(batters) + len(pitchers) == 1


def test_two_way_player_appears_on_both_pages():
    batters, pitchers = _split_roster(_roster("Two-Way Player"))
    assert len(batters) == 1 and len(pitchers) == 1
