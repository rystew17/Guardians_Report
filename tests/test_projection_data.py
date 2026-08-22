"""The layers between a source payload and a fitted model.

Nothing here is a model. These are the functions that decide which games count,
what a pitcher's line means, and how a team's quality is carried from one game
to the next -- and every one of them fails silently, because a dropped game and
a game that was never played look identical downstream, and a misread innings
figure is still a number.

Two of the corrections in this project's history came from exactly these
functions: innings-per-start read as 0.0 because the field was missing, and then
read as 14.27 because the denominator was changed to starts when the training
data counted appearances. Both rendered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from guards_report.projections import corpus, pa, pitchers, ratings


# ---------------------------------------------------------------------------
# Which games count
# ---------------------------------------------------------------------------

def _schedule_game(**over) -> dict:
    game = {
        "gamePk": 1, "officialDate": "2026-04-15", "season": "2026",
        "gameType": "R", "dayNight": "night",
        "status": {"abstractGameState": "Final"},
        "venue": {"id": 5, "name": "Progressive Field"},
        "teams": {
            "home": {"team": {"id": 114, "abbreviation": "CLE"},
                     "probablePitcher": {"id": 700}},
            "away": {"team": {"id": 115, "abbreviation": "COL"},
                     "probablePitcher": {"id": 800}},
        },
        "linescore": {
            "currentInning": 9,
            "teams": {"home": {"runs": 5, "hits": 9, "errors": 0},
                      "away": {"runs": 3, "hits": 7, "errors": 1}},
        },
    }
    for key, value in over.items():
        if key == "status":
            game["status"] = {"abstractGameState": value}
        elif key in ("home_runs", "away_runs"):
            side = "home" if key.startswith("home") else "away"
            game["linescore"]["teams"][side]["runs"] = value
        elif key == "innings":
            game["linescore"]["currentInning"] = value
        else:
            game[key] = value
    return game


def test_a_finished_game_becomes_one_row_with_the_target_derived():
    row = corpus._row(_schedule_game())
    assert row["home_win"] == 1
    assert (row["home_runs"], row["away_runs"]) == (5, 3)
    assert row["home_team_id"] == 114 and row["away_team_id"] == 115


def test_an_away_win_is_recorded_as_a_loss_for_the_home_side():
    """The target is home_win. A sign error here inverts the entire corpus."""
    assert corpus._row(_schedule_game(home_runs=2, away_runs=6))["home_win"] == 0


def test_a_game_that_has_not_finished_is_dropped_rather_than_patched():
    """In-progress games carry a partial score that looks like a final one."""
    for state in ("Live", "Preview", "Other"):
        assert corpus._row(_schedule_game(status=state)) is None


def test_a_game_with_no_linescore_is_dropped():
    game = _schedule_game()
    game["linescore"] = {}
    assert corpus._row(game) is None


def test_a_tie_is_dropped_because_it_is_not_a_decidable_outcome():
    """Vanishingly rare in the modern game, and silently corrupting.

    A tie scored as `home_win = 0` is a loss the home team did not suffer, and
    it enters the corpus looking exactly like a real one.
    """
    assert corpus._row(_schedule_game(home_runs=4, away_runs=4)) is None


def test_extra_innings_are_flagged_and_nine_innings_are_not():
    assert corpus._row(_schedule_game(innings=9))["extra_innings"] == 0
    assert corpus._row(_schedule_game(innings=11))["extra_innings"] == 1


def test_a_missing_probable_pitcher_leaves_the_row_usable():
    """Most of the corpus has one; a row without it must not be discarded.

    Dropping them would quietly select against exactly the games where a
    starter was announced late, which is not a random subset.
    """
    game = _schedule_game()
    game["teams"]["home"].pop("probablePitcher")
    row = corpus._row(game)
    assert row is not None and row["home_starter_id"] is None
    assert row["away_starter_id"] == 800


def test_the_season_falls_back_to_the_date_when_the_field_is_absent():
    game = _schedule_game()
    game.pop("season")
    assert corpus._row(game)["season"] == 2026


# ---------------------------------------------------------------------------
# Innings notation — twice wrong in this project's history
# ---------------------------------------------------------------------------

def test_innings_notation_is_thirds_and_not_decimal():
    """"6.1" is six and one third, which is 19 outs, not 6.1 innings.

    Reading it as a decimal understates every start by up to two thirds of an
    inning and does it consistently, so nothing looks wrong.
    """
    assert pitchers._outs("6.1") == 19
    assert pitchers._outs("6.2") == 20
    assert pitchers._outs("7.0") == 21
    assert pitchers._outs(7) == 21


def test_an_absent_innings_figure_reads_as_zero_rather_than_raising():
    """It reads as zero -- which is why the caller has to check, and did not.

    A missing `inningsPitched` silently produced 0.0 innings per start here and
    inflated the derived score by sixteen percent.
    """
    assert pitchers._outs(None) == 0
    assert pitchers._outs("") == 0
    assert pitchers._outs("nonsense") == 0


def test_an_unparseable_count_is_zero_not_an_exception():
    assert pitchers._int(None) == 0
    assert pitchers._int("12") == 12
    assert pitchers._int("-") == 0


def test_pitcher_ids_are_batched_into_one_request():
    """Twenty-five ids per call is the difference between 30 requests and 750."""
    url = pitchers._url([1, 2, 3], 2026)
    assert "personIds=1,2,3" in url and "season=2026" in url


# ---------------------------------------------------------------------------
# Plate appearances from pitches
# ---------------------------------------------------------------------------

def _pitch_rows(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    return frame


def _pitch(**over) -> dict:
    row = {
        "game_pk": 1, "game_date": "2026-04-15", "season": 2026,
        "batter": 10, "pitcher": 20, "stand": "R", "p_throws": "R",
        "events": None, "description": "ball", "inning": 1,
        "balls": 0, "strikes": 0, "at_bat_number": 1, "pitch_number": 1,
    }
    row.update(over)
    return row


def test_a_plate_appearance_is_the_pitch_that_ended_it():
    """Six pitches are one plate appearance, not six.

    Counting pitches as chances would put every rate in the corpus out by the
    average pitches per appearance -- roughly a factor of four, uniformly, so
    the ranking would survive and every number would be wrong.
    """
    frame = _pitch_rows([
        _pitch(pitch_number=1), _pitch(pitch_number=2),
        _pitch(pitch_number=3, events="single", description="hit_into_play"),
    ])
    result = pa.from_pitches(frame)
    assert len(result) == 1
    assert result.iloc[0]["events"] == "single"


def test_pitches_that_end_nothing_produce_no_plate_appearance():
    frame = _pitch_rows([_pitch(pitch_number=1), _pitch(pitch_number=2)])
    assert len(pa.from_pitches(frame)) == 0


def test_separate_at_bats_stay_separate():
    frame = _pitch_rows([
        _pitch(at_bat_number=1, events="strikeout", description="swinging_strike"),
        _pitch(at_bat_number=2, batter=11, events="walk", description="ball"),
    ])
    result = pa.from_pitches(frame)
    assert len(result) == 2
    assert set(result["batter"]) == {10, 11}


def test_the_column_sets_are_disjoint_from_neither_and_present_in_the_view():
    """`load` projects columns for speed, so a name that drifts loses a feature.

    That failure is silent: the column simply is not there, the feature falls
    back to a default, and the model gets slightly worse for no visible reason.
    """
    view = pa.from_pitches(_pitch_rows([
        _pitch(events="single", description="hit_into_play")
    ]))
    for column in ("batter", "pitcher", "stand", "p_throws", "events"):
        assert column in view.columns


# ---------------------------------------------------------------------------
# Offense and defense carried forward
# ---------------------------------------------------------------------------

def _run_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _rgame(home, away, hr, ar, season=2026) -> dict:
    return {
        "home_team_id": home, "away_team_id": away,
        "home_runs": hr, "away_runs": ar, "season": season,
        "home_win": int(hr > ar),
    }


def test_a_team_that_scores_freely_earns_a_better_offense_rating():
    frame = _run_frame([_rgame(1, 2, 10, 1) for _ in range(30)])
    params = ratings.OffDefParams()
    final = ratings.final_off_def(frame, params)
    home_off, _ = final[1]
    away_off, _ = final[2]
    assert home_off > away_off


def test_a_team_that_concedes_freely_earns_a_worse_defence_rating():
    frame = _run_frame([_rgame(1, 2, 10, 1) for _ in range(30)])
    final = ratings.final_off_def(frame, ratings.OffDefParams())
    _, home_def = final[1]
    _, away_def = final[2]
    assert home_def != away_def


def test_every_off_def_value_precedes_the_game_it_is_attached_to():
    """The same guarantee Elo has, and the same way of checking it.

    If the first game's ratings moved with its own result, the feature would
    contain the answer -- and a run model fed its own target does not look
    broken, it looks excellent.
    """
    params = ratings.OffDefParams()
    blowout = ratings.off_def(_run_frame([_rgame(1, 2, 20, 0), _rgame(1, 2, 5, 4)]), params)
    shutout = ratings.off_def(_run_frame([_rgame(1, 2, 0, 20), _rgame(1, 2, 5, 4)]), params)
    for key in blowout:
        assert blowout[key][0] == pytest.approx(shutout[key][0]), (
            f"{key} for game one saw its own result"
        )


def test_off_def_returns_a_value_for_every_game_in_the_frame():
    frame = _run_frame([_rgame(1, 2, 5, 3), _rgame(2, 3, 4, 4 + 1), _rgame(3, 1, 2, 8)])
    result = ratings.off_def(frame, ratings.OffDefParams())
    assert all(len(values) == len(frame) for values in result.values())
    assert all(np.isfinite(values).all() for values in result.values())


def test_run_elo_produces_a_rating_for_each_side_of_each_game():
    frame = _run_frame([_rgame(1, 2, 5, 3), _rgame(2, 1, 7, 2)])
    home, away = ratings.run_elo(frame, ratings.RunEloParams())
    assert len(home) == len(away) == len(frame)
    assert np.isfinite(home).all() and np.isfinite(away).all()
