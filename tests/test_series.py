"""Box score summaries for the current series.

The subtle bug these cover: `seriesGameNumber` restarts at 1 every series, so
selecting the earlier games of "this set" by that number alone pulled in games
1 and 2 of the *previous* series against a different club. A CLE-SD preview
showed CLE-CWS box scores, which looks entirely plausible until you read the
team names.
"""

from __future__ import annotations

from guards_report.metrics import series as sr


def _batting(**kw):
    base = {name: 0 for name in sr.BATTING_COUNTS}
    base.update(kw)
    return base


def _pitching(**kw):
    base = {name: 0 for name in sr.PITCHING_COUNTS}
    base.update(kw)
    return base


def _boxscore(home_players=None, away_players=None):
    def wrap(players):
        return {
            f"ID{pid}": {"person": {"id": pid, "fullName": name}, "stats": stats}
            for pid, name, stats in (players or [])
        }
    return {
        "teams": {
            "home": {"team": {"abbreviation": "CLE"}, "players": wrap(home_players)},
            "away": {"team": {"abbreviation": "SD"}, "players": wrap(away_players)},
        }
    }


# ---------------------------------------------------------------------------
# Player lines
# ---------------------------------------------------------------------------


def test_batting_line_reads_like_a_box_score():
    line = sr.SeriesLine(games=2, stat=_batting(
        hits=2, atBats=6, doubles=1, homeRuns=1, rbi=3, runs=1,
        plateAppearances=7,
    ))
    assert line.summary == "2-6, 2B, HR, 3 RBI, 1 R"


def test_repeated_extra_base_hits_are_counted():
    line = sr.SeriesLine(games=2, stat=_batting(
        hits=3, atBats=8, doubles=2, plateAppearances=9,
    ))
    assert line.summary.startswith("3-8, 2 2B")


def test_pitching_line_uses_innings_notation():
    line = sr.SeriesLine(games=1, pitching=True, stat=_pitching(
        outs=19, hits=4, runs=2, earnedRuns=2, baseOnBalls=1, strikeOuts=7,
        battersFaced=26,
    ))
    # 19 outs is six and one third innings, not 6.33.
    assert line.summary == "6.1 IP, 4 H, 2 R, 2 ER, 1 BB, 7 K"


def test_lines_accumulate_across_games():
    box1 = _boxscore(home_players=[(1, "A", {"batting": _batting(
        hits=1, atBats=4, plateAppearances=4)})])
    box2 = _boxscore(home_players=[(1, "A", {"batting": _batting(
        hits=2, atBats=3, homeRuns=1, rbi=2, plateAppearances=3)})])

    batting, _ = sr.player_series_lines([box1, box2])
    assert batting[1].games == 2
    assert batting[1].summary == "3-7, HR, 2 RBI"


def test_player_who_did_not_appear_has_no_line():
    box = _boxscore(home_players=[(1, "A", {"batting": _batting()})])
    batting, _ = sr.player_series_lines([box])
    # An all-zero stat block means he dressed but did not play.
    assert 1 not in batting


# ---------------------------------------------------------------------------
# Top performers
# ---------------------------------------------------------------------------


def test_top_performers_ranked_by_the_published_formula():
    box = _boxscore(
        home_players=[
            (1, "Big Game", {"batting": _batting(
                hits=3, atBats=4, homeRuns=1, rbi=6, runs=1, totalBases=6,
                plateAppearances=4)}),
            (2, "Quiet Day", {"batting": _batting(
                hits=0, atBats=4, plateAppearances=4)}),
        ],
        away_players=[
            (3, "Reliever", {"pitching": _pitching(
                outs=6, strikeOuts=5, battersFaced=7)}),
        ],
    )
    top = sr.top_performers(box, limit=3)

    assert top[0].name == "Big Game"          # 6 TB + 6 RBI + 1 R = 13
    assert top[1].name == "Reliever"          # 6 outs + 5 K = 11
    assert top[0].team == "CLE" and top[1].team == "SD"


def test_ranking_is_stable_across_runs():
    """Ties break on name, so a rebuild produces the same list."""
    box = _boxscore(home_players=[
        (1, "Zeta", {"batting": _batting(hits=1, atBats=4, totalBases=1,
                                         plateAppearances=4)}),
        (2, "Alpha", {"batting": _batting(hits=1, atBats=4, totalBases=1,
                                          plateAppearances=4)}),
    ])
    names = [p.name for p in sr.top_performers(box, limit=2)]
    assert names == ["Alpha", "Zeta"]


# ---------------------------------------------------------------------------
# Game box
# ---------------------------------------------------------------------------


def _schedule_game(pk=1, number=2, date_="2026-08-15"):
    return {
        "gamePk": pk,
        "officialDate": date_,
        "seriesGameNumber": number,
        "teams": {
            "home": {"team": {"abbreviation": "CLE", "id": 114}},
            "away": {"team": {"abbreviation": "SD", "id": 135}},
        },
        "decisions": {
            "winner": {"fullName": "Winner Pitcher"},
            "loser": {"fullName": "Loser Pitcher"},
        },
    }


def _linescore():
    return {
        "innings": [
            {"num": 1, "away": {"runs": 1}, "home": {"runs": 0}},
            {"num": 2, "away": {"runs": 0}, "home": {"runs": 3}},
            # The home side did not bat in the ninth -- it was already ahead.
            {"num": 9, "away": {"runs": 0}, "home": {}},
        ],
        "teams": {
            "away": {"runs": 1, "hits": 7, "errors": 2},
            "home": {"runs": 6, "hits": 12, "errors": 0},
        },
    }


def test_game_box_carries_the_final_and_decisions():
    box = sr.parse_game_box(
        schedule_game=_schedule_game(),
        boxscore=_boxscore(home_players=[
            (9, "Winner Pitcher", {"pitching": _pitching(
                outs=21, hits=4, runs=1, earnedRuns=1, strikeOuts=8,
                battersFaced=27)}),
        ]),
        linescore=_linescore(),
    )

    assert box.final == "SD 1, CLE 6"
    assert box.winner_abbr == "CLE"
    assert box.winning_pitcher == "Winner Pitcher"
    assert box.winning_line == "7.0 IP, 4 H, 1 R, 1 ER, 0 BB, 8 K"


def test_unplayed_half_inning_stays_empty():
    """A home ninth that never happened must not render as a zero."""
    box = sr.parse_game_box(
        schedule_game=_schedule_game(),
        boxscore=_boxscore(),
        linescore=_linescore(),
    )
    ninth = [i for i in box.innings if i["num"] == 9][0]
    assert ninth["home"] is None
    assert ninth["away"] == 0
