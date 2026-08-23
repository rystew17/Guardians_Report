"""The game-level read, and the card that assembles it.

This is the one evaluator whose input is a fitted model rather than a corpus, so
its failure mode differs from every other one here. A player evaluator that
breaks produces a silent player. A matchup that breaks produces a *confident*
paragraph about the wrong team, because the sign of a log-odds contribution is
the only thing separating "what separates them is Cleveland's rating" from the
exact opposite sentence, and both read equally well.

So these tests are mostly about direction and arithmetic, not about prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pytest

from guards_report.insight import card, matchup


# ---------------------------------------------------------------------------
# Stand-ins shaped like the projection the report actually carries.
# ---------------------------------------------------------------------------

@dataclass
class _Strikeouts:
    name: str = ""
    expected: float = 5.0
    line: float = 4.5


@dataclass
class _FirstFive:
    home_leads: float = 0.46
    away_leads: float = 0.41
    tied: float = 0.13
    expected_total: float = 5.1


@dataclass
class _Prop:
    player_id: int = 1
    name: str = ""
    slot: int = 1
    thin: bool = False
    home_runs: dict = field(default_factory=lambda: {"at_least_one": 0.12})
    hits: dict = field(default_factory=lambda: {"expected": 1.1})


@dataclass
class _Projection:
    win_probability: float = 0.567
    coherent: bool = True
    confidence_percentile: float = 53.0
    reference: dict = field(default_factory=dict)
    score: dict = field(default_factory=lambda: {
        "expected_total": 10.4, "exp_home_runs": 5.7,
        "exp_away_runs": 4.7, "p_one_run_game": 0.23,
    })
    contributions: list = field(default_factory=lambda: [
        {"name": "Team rating", "contribution": 0.39},
        {"name": "Starter", "contribution": -0.12},
    ])
    first_five: Any = field(default_factory=_FirstFive)
    strikeouts: dict = field(default_factory=dict)
    player_props: dict = field(default_factory=dict)


def _codes(findings) -> set[str]:
    return {f.code for f in findings}


def _detail(findings, code) -> dict:
    return next(f for f in findings if f.code == code).detail


# ---------------------------------------------------------------------------
# Direction — the failure that reads perfectly
# ---------------------------------------------------------------------------

def test_the_favourite_is_the_side_the_probability_actually_favours():
    home = matchup.from_projection(_Projection(win_probability=0.567), "CLE", "COL")
    away = matchup.from_projection(_Projection(win_probability=0.433), "CLE", "COL")
    assert _detail(home, "game.projection")["favorite"] == "CLE"
    assert _detail(away, "game.projection")["favorite"] == "COL"


def test_confidence_is_the_distance_from_a_coin_flip_not_the_home_probability():
    """0.433 is not 43% confidence — it is 57% confidence in the away side.

    Reporting the raw home number as the strength of the call would describe
    every away favorite as a game the model is unsure about.
    """
    findings = matchup.from_projection(_Projection(win_probability=0.433), "CLE", "COL")
    assert _detail(findings, "game.projection")["probability"] == pytest.approx(0.567)


def test_a_negative_contribution_points_at_the_away_team():
    """The sign convention is the whole content of the driver sentence.

    Contributions are signed toward the home team, so the largest one being
    negative means the away side. Getting this backwards produces a sentence
    that is fluent, specific and exactly wrong.
    """
    projection = _Projection(contributions=[{"name": "Bullpen", "contribution": -0.44}])
    detail = _detail(matchup.from_projection(projection, "CLE", "COL"), "game.driver")
    assert detail["toward"] == "COL"
    assert detail["size"] == pytest.approx(0.44)


def test_the_driver_is_the_largest_contribution_by_magnitude():
    projection = _Projection(contributions=[
        {"name": "Team rating", "contribution": 0.10},
        {"name": "Bullpen", "contribution": -0.55},
    ])
    findings = matchup.from_projection(projection, "CLE", "COL")
    assert _detail(findings, "game.driver")["name"] == "Bullpen"


def test_contributions_too_small_to_matter_are_not_counted_as_drivers():
    """"The largest of four inputs" has to mean four inputs that did something."""
    projection = _Projection(contributions=[
        {"name": "Team rating", "contribution": 0.39},
        {"name": "Rest", "contribution": 0.001},
        {"name": "Travel", "contribution": -0.002},
    ])
    findings = matchup.from_projection(projection, "CLE", "COL")
    assert _detail(findings, "game.driver")["count"] == 1


def test_no_contributions_means_no_driver_sentence_rather_than_a_hedged_one():
    projection = _Projection(contributions=[])
    assert "game.driver" not in _codes(matchup.from_projection(projection, "CLE", "COL"))


# ---------------------------------------------------------------------------
# The pieces that come from separate models
# ---------------------------------------------------------------------------

def test_score_detail_keeps_each_run_total_with_its_own_team():
    detail = _detail(matchup.from_projection(_Projection(), "CLE", "COL"), "game.score")
    assert detail["home"] == "CLE" and detail["home_runs"] == pytest.approx(5.7)
    assert detail["away"] == "COL" and detail["away_runs"] == pytest.approx(4.7)


def test_first_five_probabilities_include_the_tie():
    """Through five, level is a real outcome and not a rounding remainder."""
    detail = _detail(matchup.from_projection(_Projection(), "CLE", "COL"), "game.first_five")
    total = detail["home_leads"] + detail["away_leads"] + detail["tied"]
    assert total == pytest.approx(1.0, abs=0.01)
    assert detail["tied"] > 0


def test_a_projection_missing_a_model_drops_that_sentence_only():
    projection = _Projection(score={}, first_five=None)
    codes = _codes(matchup.from_projection(projection, "CLE", "COL"))
    assert "game.projection" in codes and "game.driver" in codes
    assert "game.score" not in codes and "game.first_five" not in codes


def test_no_projection_at_all_produces_nothing():
    assert matchup.from_projection(None, "CLE", "COL") == []
    assert matchup.key_player(None, "CLE", "COL") == []
    assert matchup.starter_strikeout_edge(None, "CLE", "COL") == []


# ---------------------------------------------------------------------------
# Strikeout edge
# ---------------------------------------------------------------------------

def test_the_strikeout_leader_is_the_higher_projection_whichever_side_he_is_on():
    projection = _Projection(strikeouts={
        "home": _Strikeouts(name="Bibee", expected=4.1),
        "away": _Strikeouts(name="Hughes", expected=6.3, line=5.5),
    })
    findings = matchup.starter_strikeout_edge(projection, "CLE", "COL")
    detail = _detail(findings, "game.strikeouts")
    assert detail["leader"] == "Hughes" and detail["trailer"] == "Bibee"
    assert detail["leader_line"] == 5.5


def test_one_starter_missing_means_no_strikeout_comparison():
    projection = _Projection(strikeouts={"home": _Strikeouts(name="Bibee")})
    assert matchup.starter_strikeout_edge(projection, "CLE", "COL") == []


# ---------------------------------------------------------------------------
# Key player
# ---------------------------------------------------------------------------

def test_the_key_bat_is_the_best_chance_across_both_lineups():
    projection = _Projection(player_props={
        "home": [_Prop(name="Ramirez", home_runs={"at_least_one": 0.14})],
        "away": [_Prop(name="Doe", home_runs={"at_least_one": 0.21})],
    })
    detail = _detail(matchup.key_player(projection, "CLE", "COL"), "game.key_bat")
    assert detail["name"] == "Doe" and detail["team"] == "COL"


def test_a_thin_projection_cannot_be_the_key_bat():
    """A callup with four plate appearances will out-project everyone.

    The props layer marks those thin, and the matchup note has to honor it —
    naming the least-known hitter on the card as the man to watch is precisely
    the small-sample failure this package exists to prevent.
    """
    projection = _Projection(player_props={
        "home": [_Prop(name="Ramirez", home_runs={"at_least_one": 0.14})],
        "away": [_Prop(name="Callup", thin=True, home_runs={"at_least_one": 0.40})],
    })
    detail = _detail(matchup.key_player(projection, "CLE", "COL"), "game.key_bat")
    assert detail["name"] == "Ramirez"


def test_every_hitter_thin_means_no_one_is_named():
    projection = _Projection(player_props={"home": [_Prop(thin=True)], "away": []})
    assert matchup.key_player(projection, "CLE", "COL") == []


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def test_the_paragraph_reads_in_reading_order_not_significance_order():
    """The call comes first, then what drove it — always, whatever the z-scores.

    Every other selector in this package sorts by significance. This one must
    not: a matchup note has a natural sequence, and a paragraph that opens with
    the strikeout line because it happened to score highest reads as a list of
    facts rather than a read on the game.
    """
    findings = matchup.from_projection(_Projection(), "CLE", "COL")
    findings += matchup.starter_strikeout_edge(_Projection(strikeouts={
        "home": _Strikeouts(name="Bibee", expected=4.1),
        "away": _Strikeouts(name="Hughes", expected=9.9),
    }), "CLE", "COL")

    # Lowercased because a finding that opens its sentence gets capitalized,
    # so the driver label's case depends on which template variant it drew.
    text = matchup.paragraph(list(reversed(findings))).lower()
    # Every probe is content common to all of a finding's phrasings. Which
    # variant it draws is a template choice; the order they are spoken in is
    # what is under test.
    call, driver = text.index("56.7%"), text.index("team rating")
    score, five = text.index("cle 5.7"), text.index("through five")
    assert call < driver < score < five < text.index("strikeouts")


def test_the_paragraph_is_capitalised_and_punctuated_throughout():
    text = matchup.paragraph(matchup.from_projection(_Projection(), "CLE", "COL"))
    assert text[0].isupper() and text.endswith(".")
    assert ".." not in text


def test_the_limit_caps_the_paragraph_without_dropping_the_call():
    findings = matchup.from_projection(_Projection(), "CLE", "COL")
    text = matchup.paragraph(findings, limit=2)
    assert "56.7%" in text and "Expected score" not in text


def test_no_findings_produce_no_paragraph():
    assert matchup.paragraph([]) == ""


# ---------------------------------------------------------------------------
# The card runner
# ---------------------------------------------------------------------------

@dataclass
class _Box:
    player_id: int
    name: str


@dataclass
class _Section:
    abbreviation: str
    pitchers: list = field(default_factory=list)
    batters: list = field(default_factory=list)


@dataclass
class _Bundle:
    home: Any
    away: Any
    game_pk: int = 1
    projection: Any = None


def _bundle() -> _Bundle:
    return _Bundle(
        home=_Section("CLE", pitchers=[_Box(1, "Tanner Bibee")],
                      batters=[_Box(2, "Jose Ramirez")]),
        away=_Section("COL", pitchers=[_Box(3, "Gabriel Hughes")],
                      batters=[_Box(4, "Ezequiel Tovar")]),
        projection=_Projection(),
    )


def _stub_corpus(tmp_path, monkeypatch) -> None:
    """A corpus with the right columns and nothing worth saying about it."""
    frame = pd.DataFrame({column: [None] for column in card.COLUMNS})
    frame.to_parquet(tmp_path / "2026_04.parquet")
    monkeypatch.setattr(
        card.batter_evaluators, "league_profile",
        lambda corpus: pd.DataFrame({"chase": [0.3], "whiff": [0.25], "luck_gap": [0.0]}),
    )


def test_an_empty_corpus_warns_and_returns_rather_than_raising(tmp_path):
    """Opening day has no pitches yet, and the report still has to build."""
    result = card.analyse(_bundle(), pitch_dir=tmp_path, on=None)
    assert result.subjects == {} and result.warnings
    assert "no pitch data" in result.warnings[0]


def test_one_broken_subject_costs_that_subject_and_nothing_else(tmp_path, monkeypatch):
    """A report missing one player's line is still a report.

    A report that failed to build is not, which is why every subject is wrapped
    individually rather than the loop as a whole.
    """
    _stub_corpus(tmp_path, monkeypatch)

    def explode(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(card, "analyse_pitcher", explode)
    monkeypatch.setattr(card, "analyse_batter", lambda *a, **k: ("said something", 3))

    result = card.analyse(_bundle(), pitch_dir=tmp_path, on=None)
    assert len(result.warnings) == 2 and all("boom" in w for w in result.warnings)
    assert result.attempted == 4 and result.covered == 2
    assert result.matchup


def test_coverage_counts_silence_as_attempted_but_not_covered(tmp_path, monkeypatch):
    """Saying nothing is a valid outcome, and must not read as full coverage."""
    _stub_corpus(tmp_path, monkeypatch)
    monkeypatch.setattr(card, "analyse_pitcher", lambda *a, **k: ("", 0))
    monkeypatch.setattr(card, "analyse_batter", lambda *a, **k: ("something", 2))

    result = card.analyse(_bundle(), pitch_dir=tmp_path, on=None)
    assert result.attempted == 4 and result.covered == 2
    assert result.coverage == pytest.approx(0.5)


def test_coverage_of_an_empty_card_is_zero_rather_than_a_division_error():
    assert card.CardAnalysis().coverage == 0.0


def test_the_criteria_counts_match_the_evaluators_actually_run():
    """These set the significance floor, so drift here silently admits noise.

    Raising the evaluator count without raising the bar is the failure mode the
    whole package is built to avoid, and it is invisible from the output.
    """
    assert card.PITCHER_CRITERIA >= 8
    assert card.BATTER_CRITERIA >= 12


def test_a_surname_survives_a_qualified_display_name():
    assert card._surname("Jose Ramirez (3B)") == "Ramirez"
    assert card._surname("Tanner Bibee") == "Bibee"
    assert card._surname("") == ""


# ---------------------------------------------------------------------------
# The three-part note
# ---------------------------------------------------------------------------

@dataclass
class _Record:
    wins: int = 70
    losses: int = 58
    luck: int = 0
    streak: str = "W2"
    splits: dict = field(default_factory=lambda: {"lastTen": (6, 4)})


@dataclass
class _Profile:
    record: Any = None
    run_differential: int = 0
    bullpen_pitches_last_3: int = 100


@dataclass
class _Series:
    games_played: int = 1
    games_in_series: int = 3
    home_wins: int = 0
    away_wins: int = 1


def _sides(home_wins=70, away_wins=70, home_diff=0, away_diff=0,
           home_ten=(5, 5), away_ten=(5, 5), home_luck=0, away_luck=0):
    home = _Section("CLE")
    away = _Section("COL")
    home.profile = _Profile(
        record=_Record(wins=home_wins, losses=128 - home_wins, luck=home_luck,
                       splits={"lastTen": home_ten}),
        run_differential=home_diff)
    away.profile = _Profile(
        record=_Record(wins=away_wins, losses=128 - away_wins, luck=away_luck,
                       splits={"lastTen": away_ten}),
        run_differential=away_diff)
    bundle = _Bundle(home=home, away=away, projection=_Projection())
    bundle.series = _Series()
    return bundle


def test_the_note_reads_in_a_fixed_order():
    """Who these clubs are, then who is pitching, then what the model says.

    Not sorted by significance. A 57% call means something different about a
    first-place club than a last-place one, so the standings have to come first
    for the projection to land.
    """
    built = matchup.note(_sides())
    assert [name for name, _ in built.parts] == [
        n for n in ("On paper", "On the mound", "Projections")
        if n in dict(built.parts)]
    if len(built.parts) > 1:
        names = [n for n, _ in built.parts]
        assert names == sorted(names, key=lambda n: (
            "On paper", "On the mound", "Projections").index(n))


def test_the_better_record_is_named_as_the_better_record():
    """The sign of the comparison is the whole sentence."""
    text = matchup.write_on_paper(_sides(home_wins=90, away_wins=50))
    assert "CLE" in text and text.index("CLE") < text.index("COL")


def test_two_clubs_of_a_kind_are_not_declared_separated():
    """A handful of games over a season is not a gap worth asserting."""
    text = matchup.write_on_paper(_sides(home_wins=66, away_wins=64)).lower()
    assert any(w in text for w in ("level", "close", "little between"))


def test_a_record_ahead_of_its_run_differential_is_flagged():
    """The reading a standings page cannot give.

    A club four wins above what its scoring implies has been getting results the
    runs do not support, and that is worth more than either figure alone.
    """
    text = matchup.write_on_paper(_sides(home_wins=80, away_wins=55, home_luck=7)).lower()
    assert any(w in text for w in ("flatter", "not supported", "more than their runs"))


def test_recent_form_running_against_the_season_is_called_out():
    """The interesting case: the worse club is playing better right now."""
    text = matchup.write_on_paper(
        _sides(home_wins=85, away_wins=50, home_ten=(2, 8), away_ten=(8, 2)))
    assert "COL" in text
    lowered = text.lower()
    assert any(w in lowered for w in ("cuts against", "flip", "hotter"))


def test_the_series_state_is_reported():
    built = matchup.write_on_paper(_sides())
    assert "series" in built.lower() or "lead" in built.lower() or "opener" in built.lower()


def test_a_club_with_no_standings_still_gets_the_other_sections():
    """Losing one part is a smaller loss than losing the note."""
    bundle = _sides()
    bundle.home.profile = None
    built = matchup.note(bundle)
    assert built.on_paper == ""
    assert built.projections, "the projection section must survive"


def test_no_projection_leaves_the_first_two_sections_standing():
    bundle = _sides()
    bundle.projection = None
    built = matchup.note(bundle)
    assert built.on_paper and not built.projections


def test_a_starter_is_graded_against_the_lineup_he_actually_faces():
    """The parameters are named for what they are, because they were not.

    `away_lineup` reads like "the away team's lineup"; the caller meant "the
    lineup the away starter faces". Paired the natural way it graded each
    pitcher against his own team while labelling the sentence with the other --
    right words, wrong numbers, and nothing in the output to show it.
    """
    import inspect

    signature = inspect.signature(matchup.write_on_the_mound)
    assert "home_faces" in signature.parameters
    assert "away_faces" in signature.parameters
    assert "home_lineup" not in signature.parameters


def test_the_note_is_empty_rather_than_raising_on_an_empty_bundle():
    bundle = _Bundle(home=_Section("CLE"), away=_Section("COL"), projection=None)
    built = matchup.note(bundle)
    assert built.empty
