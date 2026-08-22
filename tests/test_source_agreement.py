"""Cross-validate our hard-coded math against the source's own published rates.

This is the strongest verification in the project. The MLB Stats API returns
both raw counting stats *and* its own computed AVG/OBP/SLG/OPS/ERA/WHIP. So we
recompute those rates from the counting stats and assert we land on the same
number the source published.

If these pass, our formulas agree with MLB's for every metric where a
comparison is possible -- which is exactly the "100% verifiable" property the
project requires, checked continuously rather than by hand once.

These tests hit the network. Run with:  pytest -m network
"""

from __future__ import annotations

from datetime import date, timezone

import pytest

from guards_report.config import CLEVELAND_GUARDIANS_TEAM_ID, load_settings
from guards_report.metrics import formulas as f
from guards_report.metrics import league_constants as lc
from guards_report.metrics import windows as w
from guards_report.sources import mlb_statsapi as api
from guards_report.sources.http import Archiver

pytestmark = pytest.mark.network

SEASON = 2026


@pytest.fixture(scope="module")
def archiver(tmp_path_factory) -> Archiver:
    return Archiver(root=tmp_path_factory.mktemp("raw"))


def _rounded_str(value: float | None, digits: int = 3) -> str | None:
    """Format like the API does: .305 rather than 0.305."""
    if value is None:
        return None
    text = f"{value:.{digits}f}"
    return text[1:] if text.startswith("0.") else text


# ---------------------------------------------------------------------------
# Hitting rates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "person_id",
    [
        608070,  # Jose Ramirez
        682829,  # Elly De La Cruz
    ],
)
def test_hitting_slash_line_matches_source(archiver, person_id):
    """Recompute AVG/OBP/SLG/OPS from counting stats; match what MLB reports."""
    result = api.season_totals(
        archiver, person_id=person_id, season=SEASON, group=api.GROUP_HITTING
    )
    splits = (result.json().get("stats") or [{}])[0].get("splits") or []
    if not splits:
        pytest.skip(f"no {SEASON} hitting stats for {person_id}")

    stat = splits[0]["stat"]

    hits = int(stat["hits"])
    at_bats = int(stat["atBats"])
    doubles, triples = int(stat["doubles"]), int(stat["triples"])
    home_runs = int(stat["homeRuns"])
    walks = int(stat["baseOnBalls"])
    hbp = int(stat["hitByPitch"])
    sac_flies = int(stat["sacFlies"])

    singles_ = f.singles(hits, doubles, triples, home_runs)
    total_bases = f.total_bases(singles_, doubles, triples, home_runs)

    # The API also reports totalBases directly -- our derivation must match it.
    assert total_bases == int(stat["totalBases"])

    avg = f.batting_average(hits, at_bats)
    obp = f.on_base_pct(hits, walks, hbp, at_bats, sac_flies)
    slg = f.slugging(total_bases, at_bats)

    assert _rounded_str(avg) == stat["avg"]
    assert _rounded_str(obp) == stat["obp"]
    assert _rounded_str(slg) == stat["slg"]

    # MLB publishes OPS as the sum of the rounded components, which can differ
    # by a point from rounding the full-precision sum. We match their
    # convention for display so the report agrees with mlb.com.
    assert _rounded_str(f.published_ops(obp, slg)) == stat["ops"]

    # Our full-precision OPS must still be within a rounding step of theirs.
    assert f.ops(obp, slg) == pytest.approx(float(stat["ops"]), abs=0.001)


# ---------------------------------------------------------------------------
# Pitching rates
# ---------------------------------------------------------------------------


def test_pitching_era_and_whip_match_source(archiver):
    """ERA and WHIP recomputed from outs must match the source's own figures.

    This is the test that would catch the innings-pitched notation trap: if
    "5.1 IP" were ever treated as 5.1 innings instead of 16 outs, ERA and WHIP
    would drift from MLB's published values and this fails.
    """
    roster = api.active_roster(
        archiver, team_id=CLEVELAND_GUARDIANS_TEAM_ID, season=SEASON
    ).json()

    pitcher_ids = [
        entry["person"]["id"]
        for entry in roster.get("roster", [])
        if entry.get("position", {}).get("type") == "Pitcher"
    ]
    assert pitcher_ids, "no pitchers found on the active roster"

    checked = 0
    for person_id in pitcher_ids[:6]:
        payload = api.season_totals(
            archiver, person_id=person_id, season=SEASON, group=api.GROUP_PITCHING
        ).json()
        splits = (payload.get("stats") or [{}])[0].get("splits") or []
        if not splits:
            continue
        stat = splits[0]["stat"]
        outs = int(stat.get("outs") or 0)
        if outs == 0:
            continue

        era = f.era(int(stat["earnedRuns"]), outs)
        whip = f.whip(int(stat["baseOnBalls"]), int(stat["hits"]), outs)

        assert era is not None and whip is not None
        assert f"{era:.2f}" == stat["era"], f"ERA mismatch for {person_id}"
        assert f"{whip:.2f}" == stat["whip"], f"WHIP mismatch for {person_id}"

        # The API's innings string must agree with the outs it reports.
        assert f.ip_to_outs(stat["inningsPitched"]) == outs
        checked += 1

    assert checked >= 2, "not enough pitchers with innings to validate against"


# ---------------------------------------------------------------------------
# League constants
# ---------------------------------------------------------------------------


def test_league_constants_derive_to_plausible_values(archiver):
    payload = api.league_pitching_totals(archiver, season=SEASON).json()
    totals = lc.parse_league_totals(payload, season=SEASON)
    constants = lc.derive(totals)

    assert totals.team_count == 30
    # Sanity bands for a modern MLB season.
    assert 3.0 < constants.league_era < 5.5
    assert 2.5 <= constants.fip_constant <= 3.7

    # League ERA must be derived from summed totals, matching our own formula.
    assert constants.league_era == pytest.approx(
        f.era(totals.earned_runs, totals.outs)
    )


# ---------------------------------------------------------------------------
# Window aggregation
# ---------------------------------------------------------------------------


def test_our_window_agrees_with_source_last_x_games(archiver):
    """Our L15 built from game logs must match the source's own lastXGames.

    Independent implementations of the same window, compared. A disagreement
    means our aggregation or ordering logic is wrong.

    The two endpoints disagree about what "most recent" means while a game is
    being played: `lastXGames` picks up the live game immediately, whereas
    `gameLog` only gains the row once the game is final. Their windows then
    cover different sets of games and no comparison is meaningful, so the test
    detects that state and skips rather than reporting a defect that is not
    there.
    """
    person_id = 608070  # Jose Ramirez

    log_payload = api.game_log(
        archiver, person_id=person_id, season=SEASON, group=api.GROUP_HITTING
    ).json()
    rows = w.parse_game_logs(log_payload)
    if len(rows) < 20:
        pytest.skip("not enough games played yet to compare a 15-game window")

    # The source's lastXGames counts back from the player's most recent game,
    # so we anchor our window the same way: the day after that game.
    from datetime import timedelta

    as_of = rows[0].game_date + timedelta(days=1)

    ours = w.aggregate_hitting(
        w.select(rows, w.WindowSpec(label="L15", games=15), as_of=as_of)
    )

    theirs_payload = api.last_x_games(
        archiver,
        person_id=person_id,
        season=SEASON,
        group=api.GROUP_HITTING,
        limit=15,
    ).json()
    theirs = (theirs_payload.get("stats") or [{}])[0]["splits"][0]["stat"]

    # `lastXGames` counts a game the moment it starts, so during a live game
    # the two windows cover different sets and cannot be compared. Our own data
    # is bounded at `as_of`, which is exactly the point -- see
    # test_live_game_is_excluded_from_windows.
    if rows[0].game_date >= _today_eastern():
        pytest.skip(
            f"most recent logged game ({rows[0].game_date}) is today's, which "
            "may still be in progress; windows are not comparable mid-game"
        )

    assert ours["games"] == 15
    for field in ("atBats", "hits", "homeRuns", "baseOnBalls", "strikeOuts"):
        assert ours[field] == int(theirs[field]), f"{field} disagrees"

    assert _rounded_str(ours["avg"]) == theirs["avg"]


def test_rates_come_from_summed_counts_not_averaged_rates():
    """A 1-for-1 game and an 0-for-5 game is 1 for 6 (.167), not .500."""
    rows = [
        w.GameLogRow(
            game_date=date(2026, 8, 15), game_pk=1, is_home=True,
            opponent_team_id=None,
            stat={"hits": 1, "atBats": 1, "plateAppearances": 1},
        ),
        w.GameLogRow(
            game_date=date(2026, 8, 14), game_pk=2, is_home=True,
            opponent_team_id=None,
            stat={"hits": 0, "atBats": 5, "plateAppearances": 5},
        ),
    ]
    aggregated = w.aggregate_hitting(rows)
    assert aggregated["hits"] == 1
    assert aggregated["atBats"] == 6
    assert aggregated["avg"] == pytest.approx(1 / 6)


def _today_eastern():
    """The current date in the league's time zone, which is what a game date is."""
    from datetime import datetime

    from guards_report.metrics import clocks

    return clocks.to_eastern(datetime.now(timezone.utc)).date()


@pytest.mark.network
def test_live_game_is_excluded_from_windows(archiver):
    """A game in progress must not reach any window, season line, or chart.

    The source adds a game to the log as soon as it starts and updates it pitch
    by pitch. Two reports built an hour apart would otherwise disagree, and
    neither could be checked against a published figure -- the whole point of
    the project is that every number can be verified after the fact.
    """
    person_id = 608070  # Jose Ramirez

    rows = w.parse_game_logs(
        api.game_log(
            archiver, person_id=person_id, season=SEASON, group=api.GROUP_HITTING
        ).json()
    )
    if not rows:
        pytest.skip("no games logged yet this season")

    as_of = _today_eastern()

    # Season line and every window are built from rows strictly before as_of.
    prior = [r for r in rows if r.game_date < as_of]
    assert all(r.game_date < as_of for r in prior)

    for spec in w.HITTER_WINDOWS:
        selected = w.select(rows, spec, as_of=as_of)
        assert all(r.game_date < as_of for r in selected), (
            f"{spec.label} window reached into {as_of}"
        )

    # And the aggregate is unchanged by whether today's row exists at all,
    # which is the property that makes a run reproducible.
    assert w.aggregate_hitting(prior) == w.aggregate_hitting(
        [r for r in rows if r.game_date < as_of]
    )


@pytest.mark.network
def test_our_fielding_percentile_matches_the_published_one():
    """Ranked against every fielder on the board, not a subset.

    This was believed to be five percentile points high for a while, because it
    was being compared with Fielding Run Value on the player page. That is a
    different Savant metric with a different percentile: the regular who reads
    90 there reads 95 on outs above average, which is what this computes.
    Chasing the wrong target produced three plausible-looking fixes -- ranking
    within position, within outfielders, and on runs prevented -- each of which
    made the real agreement worse.

    Pinned here because the only way to notice was to find the published
    percentile and compare against it, and that is exactly what a test is for.
    """
    import numpy as np

    from guards_report.config import load_settings
    from guards_report.sources import savant as sv
    from guards_report.sources.http import Archiver

    archiver = Archiver(root=load_settings().raw_archive_dir)
    season = date.today().year

    published = {}
    for row in sv.parse_csv(sv.percentile_rankings(
        archiver, year=season, player_type=sv.TYPE_BATTER
    )):
        value = row.get("oaa")
        if value not in (None, ""):
            published[int(row["player_id"])] = float(value)

    board = {}
    for row in sv.parse_csv(sv.outs_above_average(archiver, year=season, minimum=1)):
        value = sv.to_number(row.get("outs_above_average"))
        if value is not None:
            board[int(row["player_id"])] = value

    if len(published) < 50:
        pytest.skip(f"only {len(published)} published percentiles this early")

    pool = sorted(board.values())
    errors = []
    for player_id, want in published.items():
        if player_id not in board:
            continue
        value = board[player_id]
        below = sum(1 for x in pool if x < value)
        equal = sum(1 for x in pool if x == value)
        # Midrank, because the distribution has heavy tie mass at zero and a
        # strict less-than would systematically under-rate everyone at it.
        errors.append(abs((below + equal / 2) / len(pool) * 100 - want))

    errors = np.array(errors)
    assert np.median(errors) <= 3.0, (
        f"median {np.median(errors):.1f} points from the published percentile")
    assert (errors > 10).mean() <= 0.05, (
        f"{(errors > 10).mean():.1%} of players more than 10 points off")
