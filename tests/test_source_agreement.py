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

from datetime import date

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
