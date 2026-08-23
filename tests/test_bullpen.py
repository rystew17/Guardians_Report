"""Bullpen availability — who can actually pitch tonight, and how good they are.

The first attempt at this block used season-to-date bullpen runs allowed and
contributed nothing: -0.00006 log loss at p = 0.61. The diagnosis was that an
aggregate destroys the signal. A pen averaging 4.00 with its two leverage arms
rested and the same pen with both burned are the same number and not remotely
the same proposition.

So the module models the pen as a roster of individuals, and everything below
tests the two things that makes possible: that an unavailable arm is actually
excluded, and that quality is computed over what remains rather than over the
whole roster. Both fail silently — a pen that counts a man who pitched three
days running still produces a plausible ERA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from guards_report.projections import bullpen


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _corpus(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    return frame


def _game(pk, date, home=10, away=20, season=2026) -> dict:
    return {
        "game_pk": pk, "game_date": date, "season": season,
        "home_team_id": home, "away_team_id": away,
        "home_runs": 4, "away_runs": 3, "venue_id": 1,
    }


def _log(pitcher, pk, date, *, outs=3, is_home=True, started=0, season=2026,
         hr=0, bb=0, hbp=0, k=3) -> dict:
    """One appearance, in the column names the MLB logs actually use.

    camelCase, because that is what the source sends and the loader keeps. A
    fixture in snake_case builds a frame that looks right and shares no columns
    with the real one.
    """
    return {
        "pitcher_id": pitcher, "game_pk": pk, "game_date": date,
        "season": season, "gamesStarted": started, "is_home": is_home,
        "outs": outs, "homeRuns": hr, "baseOnBalls": bb, "hitByPitch": hbp,
        "strikeOuts": k, "battersFaced": outs + 3, "earnedRuns": 1,
        "hits": 1, "runs": 1,
    }


def _logs(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    return frame


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------

def test_a_starter_is_not_counted_as_a_reliever():
    """`gamesStarted` is the only thing separating the two, and a starter
    folded into the pen would make every club's bullpen look deeper than it is."""
    corpus = _corpus([_game(1, "2026-05-01")])
    logs = _logs([
        _log(7, 1, "2026-05-01", started=1),
        _log(8, 1, "2026-05-01", started=0),
    ])
    pool = bullpen.reliever_pool(logs, corpus)
    assert set(pool["pitcher_id"]) == {8}


def test_a_relievers_team_is_recovered_from_the_game_he_appeared_in():
    """Team identity is not in the pitcher logs.

    It is read off which side of the game he was on, and getting it backwards
    would credit every reliever to his opponent -- which produces two complete,
    plausible bullpens belonging to the wrong clubs.
    """
    corpus = _corpus([_game(1, "2026-05-01", home=10, away=20)])
    logs = _logs([
        _log(7, 1, "2026-05-01", is_home=True),
        _log(8, 1, "2026-05-01", is_home=False),
    ])
    pool = bullpen.reliever_pool(logs, corpus).set_index("pitcher_id")
    assert pool.loc[7, "team_id"] == 10
    assert pool.loc[8, "team_id"] == 20


def test_an_appearance_in_no_known_game_is_dropped():
    """An inner join, deliberately: a log row with no matching game cannot have
    its team resolved, and guessing would put an arm on the wrong roster."""
    corpus = _corpus([_game(1, "2026-05-01")])
    logs = _logs([_log(7, 1, "2026-05-01"), _log(8, 999, "2026-05-01")])
    pool = bullpen.reliever_pool(logs, corpus)
    assert set(pool["pitcher_id"]) == {7}


# ---------------------------------------------------------------------------
# Availability — the whole point of the block
# ---------------------------------------------------------------------------

def _season_of_relief(*, pitchers=(1, 2, 3, 4, 5), days=40, team=10):
    """A club whose whole pen works every few days through a season."""
    corpus, logs = [], []
    for day in range(days):
        pk = 1000 + day
        date = f"2026-05-{(day % 28) + 1:02d}"
        corpus.append(_game(pk, date, home=team, away=99))
        for offset, pitcher in enumerate(pitchers):
            if (day + offset) % 3 == 0:
                logs.append(_log(pitcher, pk, date, is_home=True))
    return _corpus(corpus), _logs(logs)


def test_the_table_reports_one_row_per_team_game():
    corpus, logs = _season_of_relief()
    table = bullpen.availability_table(logs, corpus)
    assert len(table) > 0
    assert not table.duplicated(["game_pk", "team_id"]).any()


def test_the_metrics_the_feature_block_needs_are_all_present():
    corpus, logs = _season_of_relief()
    table = bullpen.availability_table(logs, corpus)
    for metric in bullpen.PEN_METRICS:
        assert metric in table.columns, metric


def test_an_arm_that_pitched_yesterday_counts_against_availability():
    """Back-to-back is the commonest reason a manager will not use a man, and a
    pen that ignored it would report full strength every night."""
    assert bullpen.BACK_TO_BACK_OUTS > 0
    assert bullpen.RECENT_WINDOW_DAYS == 3

    corpus, logs = _season_of_relief()
    table = bullpen.availability_table(logs, corpus)
    counts = table["pen_n_available"].dropna()
    assert len(counts) and counts.min() < counts.max(), (
        "availability that never varies is not being computed")


def test_availability_never_exceeds_the_roster():
    corpus, logs = _season_of_relief(pitchers=(1, 2, 3, 4, 5))
    table = bullpen.availability_table(logs, corpus)
    assert table["pen_n_available"].max() <= 5


def test_the_depth_gap_is_the_distance_between_best_and_worst():
    """The feature that encodes the failure mode the aggregate missed.

    A club with a rested closer and a rested long man has a small gap; one down
    to its last arm has a large one, and the season aggregate cannot tell them
    apart.
    """
    corpus, logs = _season_of_relief()
    table = bullpen.availability_table(logs, corpus).dropna(
        subset=["pen_best_fip", "pen_worst_fip", "pen_depth_gap"])
    if len(table):
        expected = table["pen_worst_fip"] - table["pen_best_fip"]
        assert np.allclose(table["pen_depth_gap"], expected, atol=1e-6)
        assert (table["pen_depth_gap"] >= -1e-9).all(), "a gap cannot be negative"


def test_the_best_arm_is_never_worse_than_the_worst_arm():
    """FIP is lower-is-better, so best must be the minimum. An inverted
    comparison would report every pen backwards while staying in range."""
    corpus, logs = _season_of_relief()
    table = bullpen.availability_table(logs, corpus).dropna(
        subset=["pen_best_fip", "pen_worst_fip"])
    if len(table):
        assert (table["pen_best_fip"] <= table["pen_worst_fip"] + 1e-9).all()


# ---------------------------------------------------------------------------
# As-of quality
# ---------------------------------------------------------------------------

def test_a_relievers_quality_uses_only_his_earlier_appearances():
    """The same leak guard as everywhere else, one layer down.

    A reliever rated on an outing that includes tonight would let the model see
    the game it is predicting.
    """
    corpus, logs = _season_of_relief(pitchers=(1,), days=30)
    pool = bullpen._as_of_quality(bullpen.reliever_pool(logs, corpus))
    ordered = pool.sort_values("game_date")
    assert pd.isna(ordered.iloc[0]["rp_fip"]), (
        "the first appearance has no prior record to rate him on")


def test_a_reliever_below_the_workload_floor_is_unknown_not_bad():
    """Assigning a misleading number is worse than admitting you have none.

    Below the floor a reliever's rate is noise, and a noisy number entering the
    pen average would move the feature without carrying information.
    """
    assert bullpen.MIN_PRIOR_OUTS > 0
    corpus, logs = _season_of_relief(pitchers=(1,), days=5)
    pool = bullpen._as_of_quality(bullpen.reliever_pool(logs, corpus))
    assert pool["rp_fip"].isna().all()


# ---------------------------------------------------------------------------
# The feature block
# ---------------------------------------------------------------------------

def test_the_feature_columns_are_all_differences():
    """The block joins onto a game frame where every column favours the home
    side, and a raw one-club figure would break that convention silently."""
    for name in bullpen.PEN_COLUMNS:
        assert name.endswith("_diff"), name


def test_adding_the_block_keeps_one_row_per_game():
    """A join that fans out would duplicate games and quietly reweight the fit."""
    corpus, logs = _season_of_relief()
    frame = corpus[["game_pk", "game_date", "season",
                    "home_team_id", "away_team_id"]].copy()
    before = len(frame)
    joined = bullpen.add_features(frame, logs, corpus)
    assert len(joined) == before
    assert not joined.duplicated("game_pk").any()


def test_the_block_adds_its_columns_and_leaves_the_rest_alone():
    corpus, logs = _season_of_relief()
    frame = corpus[["game_pk", "game_date", "season",
                    "home_team_id", "away_team_id"]].copy()
    joined = bullpen.add_features(frame, logs, corpus)
    for name in bullpen.PEN_COLUMNS:
        assert name in joined.columns, name
    for name in frame.columns:
        assert name in joined.columns, name


def test_a_club_with_no_relief_history_yields_nulls_rather_than_raising():
    """Opening day, and every expansion or relocation case."""
    corpus = _corpus([_game(1, "2026-04-01")])
    logs = _logs([_log(7, 1, "2026-04-01", started=1)])
    frame = corpus[["game_pk", "game_date", "season",
                    "home_team_id", "away_team_id"]].copy()
    joined = bullpen.add_features(frame, logs, corpus)
    assert len(joined) == 1
