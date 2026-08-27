"""Career honours, and the line between them and the season analysis.

The season grades answer "what has he done this year", which is what predicts
tonight. These answer "who is this", which the season cannot and must not be
asked to. Keeping them apart is the whole point: a career prior blended into a
season estimate would carry a declining thirty-four-year-old at the level he
held at twenty-seven.
"""

from __future__ import annotations

from guards_report.insight import accolades as acc
from guards_report.insight import dossier


class _Box:
    def __init__(self, awards=None, career=None):
        self.awards = awards or []
        self.career = career or {}


def _award(name, season):
    return {"name": name, "season": season}


def test_minor_league_selections_are_not_career_honours():
    """The failure an allowlist exists to prevent.

    The feed carries 151 distinct award names for twenty players, most of them
    not honours at all. Substring-matching "All-Star" pulls in
    `MiLB.com Organization All-Star` and `PCL Mid-Season All-Star` and turns a
    Triple-A journeyman into a decorated veteran.
    """
    honours = acc.honours_from([
        _award("MiLB.com Organization All-Star", "2010"),
        _award("PCL Mid-Season All-Star", "2011"),
        _award("Baseball America Double-A All-Star", "2011"),
        _award("AL Player of the Week", "2015"),
        _award("AL Rookie of the Month", "2012"),
        _award("Home Run Derby Participant", "2016"),
        _award("MLBPAA Angels Heart and Hustle Award", "2014"),
    ])
    assert honours == []


def test_real_honours_survive_and_are_counted_by_season():
    honours = acc.honours_from([
        _award("AL MVP", "2014"), _award("AL MVP", "2016"),
        _award("AL All-Star", "2012"), _award("AL All-Star", "2013"),
        _award("Rawlings AL Gold Glove", "2018"),
    ])
    by_name = {h.name: h for h in honours}
    assert by_name["MVP"].count == 2
    assert by_name["MVP"].seasons == ["2014", "2016"]
    assert by_name["All-Star"].count == 2
    assert by_name["Gold Glove"].count == 1


def test_honours_are_ordered_by_prestige_not_by_count():
    """Twelve All-Star selections must not bury three MVPs. A reader wants the
    credential that means most, and the list is trimmed to three."""
    honours = acc.honours_from(
        [_award("AL All-Star", str(y)) for y in range(2012, 2024)]
        + [_award("AL MVP", "2014"), _award("AL MVP", "2016"), _award("AL MVP", "2019")]
    )
    assert honours[0].name == "MVP"
    said = acc.summarize(acc.Career(honours=honours))
    assert said.startswith("3x MVP")


def test_a_player_with_no_major_awards_says_nothing():
    """Empty is the normal answer. "No major awards" describes the large
    majority of major leaguers and is not a fact about any of them."""
    assert dossier.write_career(
        _Box(awards=[_award("AL Player of the Week", "2020")]), surname="X") == ""
    assert dossier.write_career(_Box(), surname="X") == ""


def test_career_totals_ride_along_when_there_is_a_career_to_report():
    said = dossier.write_career(
        _Box(awards=[_award("AL MVP", "2014")],
             career={"gamesPlayed": 1763, "homeRuns": 424, "hits": 1854,
                     "avg": ".291"}),
        surname="Trout")
    assert "MVP (2014)" in said
    assert "1,763 games" in said and "424 home runs" in said


def test_a_short_career_gets_the_honour_without_the_totals():
    """Sixty games of counting stats is not context, it is noise with commas."""
    said = dossier.write_career(
        _Box(awards=[_award("AL All-Star", "2026")],
             career={"gamesPlayed": 60, "homeRuns": 4, "hits": 41}),
        surname="Genao")
    assert "All-Star" in said
    assert "games," not in said


def test_the_career_line_never_reaches_the_season_verdict():
    """The separation this whole section exists to preserve. The career is
    stated as fact beside the season, never blended into it."""
    import inspect
    source = inspect.getsource(dossier.write_profile)
    assert "acc." not in source and "write_career" not in source
