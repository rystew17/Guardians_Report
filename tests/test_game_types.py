"""Exhibition games must never reach a record or a chart.

The bug these cover was live: a Guardians-Giants head-to-head showed two prior
meetings on a day that was actually their first, because two March spring
training games came back from the schedule endpoint and were counted.
"""

from __future__ import annotations

from guards_report.metrics.statcast import parse_pitches
from guards_report.metrics.team_context import parse_series, parse_series_records


def _game(pk, gtype, date, home_id, away_id, home_win, home_score=4, away_score=2):
    return {
        "gamePk": pk,
        "gameType": gtype,
        "officialDate": date,
        "seriesGameNumber": 1,
        "gamesInSeries": 3,
        "status": {"abstractGameState": "Final"},
        "teams": {
            "home": {"team": {"id": home_id}, "isWinner": home_win, "score": home_score},
            "away": {"team": {"id": away_id}, "isWinner": not home_win, "score": away_score},
        },
    }


def _payload(games):
    return {"dates": [{"games": games}]}


def test_spring_games_excluded_from_head_to_head():
    payload = _payload([
        _game(1, "S", "2026-03-10", 114, 137, True),
        _game(2, "S", "2026-03-21", 114, 137, False),
        _game(9, "R", "2026-08-18", 114, 137, True),   # today
    ])
    series = parse_series(payload, home_team_id=114, today_game_pk=9)

    assert series.games_played == 0, "first meeting of the season"
    assert series.home_wins == 0 and series.away_wins == 0
    assert series.recent == []


def test_postseason_counts_toward_head_to_head():
    payload = _payload([
        _game(1, "S", "2026-03-10", 114, 137, True),
        _game(2, "R", "2026-08-18", 114, 137, True),
        _game(3, "D", "2026-10-05", 114, 137, False),
    ])
    series = parse_series(payload, home_team_id=114, today_game_pk=99)

    assert series.games_played == 2
    assert series.home_wins == 1 and series.away_wins == 1


def test_series_records_ignore_spring():
    payload = _payload([
        _game(1, "S", "2026-03-10", 114, 137, True),
        _game(2, "S", "2026-03-11", 114, 137, True),
        _game(3, "S", "2026-03-12", 114, 137, True),
    ])
    record = parse_series_records(payload, team_id=114)

    assert record.completed == 0
    assert record.won == 0 and record.sweeps_for == 0


def test_spring_pitches_excluded_from_statcast():
    csv = (
        "game_date,game_type,zone\n"
        "2026-02-21,S,5\n"
        "2026-03-08,S,7\n"
        "2026-03-30,R,5\n"
        "2026-10-05,D,9\n"
    )
    rows = parse_pitches(csv)

    assert [r["game_date"] for r in rows] == ["2026-03-30", "2026-10-05"]


def test_pitches_kept_when_source_omits_game_type():
    # Losing the column should degrade to an unfiltered chart, not an empty one.
    rows = parse_pitches("game_date,zone\n2026-05-01,5\n")
    assert len(rows) == 1
