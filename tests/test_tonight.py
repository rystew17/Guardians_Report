"""The live projection path — the code that runs on every report.

`tonight.py` assembles what the reader actually sees and had no coverage, which
made it the largest untested surface in the project. It is also where a leak
would be least visible: the props read a season's plate appearances at
projection time, and reading one row too many means the model has seen the game
it is predicting.

Synthetic data throughout, so every expectation is arithmetic.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from guards_report.projections import props, tonight
from guards_report.projections.train_props import PropsArtifact


def _plate(seasons=(2025, 2026), days: int = 20) -> pd.DataFrame:
    """Nine batters against three pitchers, with known per-batter rates."""
    rows = []
    rng = np.random.default_rng(11)
    for season in seasons:
        for day in range(1, days + 1):
            for slot, batter in enumerate(range(100, 109), start=1):
                rate = 0.12 + 0.03 * slot
                for turn in range(4):
                    hit = rng.random() < rate
                    homer = rng.random() < rate / 6
                    struck = rng.random() < 0.22
                    events = (
                        "home_run" if homer else
                        "single" if hit else
                        "strikeout" if struck else "field_out"
                    )
                    rows.append({
                        "game_date": date(season, 6, day),
                        "game_pk": season * 1000 + day,
                        "at_bat_number": slot + turn * 9,
                        "batter": batter,
                        "pitcher": 900 + (day % 3),
                        "stand": "L" if slot % 2 else "R",
                        "p_throws": "R",
                        "events": events,
                        "batting_team": "AAA",
                        "home_team": "AAA",
                        "away_team": "BBB",
                        "season": season,
                    })
    return pd.DataFrame(rows)


@pytest.fixture
def plate() -> pd.DataFrame:
    return _plate()


@pytest.fixture
def artifact(plate) -> PropsArtifact:
    from guards_report.projections.train_props import fit_props

    return fit_props(plate[plate["season"] == 2025], verbose=False)


CARD = list(range(100, 109))
STANDS = {b: ("L" if (i + 1) % 2 else "R") for i, b in enumerate(CARD)}


# --------------------------------------------------------------------------
# Leakage — the failure that would look like a brilliant model
# --------------------------------------------------------------------------

def test_props_never_read_the_day_being_projected(artifact, plate):
    """A plate appearance from the game itself must not reach the estimate."""
    cutoff = date(2026, 6, 10)
    baseline = tonight.batter_props(
        artifact, plate, on=cutoff, lineup=CARD, names={},
        opposing_starter=900, opposing_throws="R", stands=STANDS, home_team="AAA",
    )

    poisoned = plate.copy()
    later = poisoned["game_date"] >= cutoff
    poisoned.loc[later, "events"] = "home_run"

    after = tonight.batter_props(
        artifact, poisoned, on=cutoff, lineup=CARD, names={},
        opposing_starter=900, opposing_throws="R", stands=STANDS, home_team="AAA",
    )

    for before, now in zip(baseline, after):
        assert now.hits["expected"] == pytest.approx(before.hits["expected"])
        assert now.home_runs["expected"] == pytest.approx(before.home_runs["expected"])


def test_strikeouts_never_read_the_day_being_projected(artifact, plate):
    cutoff = date(2026, 6, 10)
    baseline = tonight.starter_strikeouts(
        artifact, plate, on=cutoff, pitcher_id=900, name="",
        opposing_lineup=CARD, stands=STANDS, throws="R", expected_bf=22,
    )
    poisoned = plate.copy()
    poisoned.loc[poisoned["game_date"] >= cutoff, "events"] = "strikeout"
    after = tonight.starter_strikeouts(
        artifact, poisoned, on=cutoff, pitcher_id=900, name="",
        opposing_lineup=CARD, stands=STANDS, throws="R", expected_bf=22,
    )
    assert after.expected == pytest.approx(baseline.expected)


def test_the_current_season_does_reach_the_estimate(artifact, plate):
    """The other half of the guard: as-of must not mean prior-seasons-only.

    Discarding the current season was a real error earlier in this project, and
    it is invisible unless something asserts the estimate actually moves.
    """
    early = tonight.batter_props(
        artifact, plate, on=date(2026, 6, 2), lineup=CARD, names={},
        opposing_starter=900, opposing_throws="R", stands=STANDS, home_team="AAA",
    )
    changed = plate.copy()
    window = (changed["season"] == 2026) & (changed["game_date"] < date(2026, 6, 15))
    changed.loc[window & (changed["batter"] == 100), "events"] = "single"
    late = tonight.batter_props(
        artifact, changed, on=date(2026, 6, 15), lineup=CARD, names={},
        opposing_starter=900, opposing_throws="R", stands=STANDS, home_team="AAA",
    )
    assert late[0].hits["expected"] != pytest.approx(early[0].hits["expected"])


# --------------------------------------------------------------------------
# Batter props
# --------------------------------------------------------------------------

def test_no_posted_card_yields_no_projections(artifact, plate):
    """Nine unnamed players is a guess about the lineup, not a statement."""
    assert tonight.batter_props(
        artifact, plate, on=date(2026, 6, 10), lineup=[], names={},
        opposing_starter=900, opposing_throws="R", stands=STANDS, home_team="AAA",
    ) == []


def test_no_announced_starter_yields_no_projections(artifact, plate):
    assert tonight.batter_props(
        artifact, plate, on=date(2026, 6, 10), lineup=CARD, names={},
        opposing_starter=None, opposing_throws="R", stands=STANDS, home_team="AAA",
    ) == []


def test_every_slot_is_projected_in_batting_order(artifact, plate):
    card = tonight.batter_props(
        artifact, plate, on=date(2026, 6, 10), lineup=CARD, names={},
        opposing_starter=900, opposing_throws="R", stands=STANDS, home_team="AAA",
    )
    assert [p.slot for p in card] == list(range(1, 10))
    assert [p.player_id for p in card] == CARD


def test_probabilities_are_internally_consistent(artifact, plate):
    card = tonight.batter_props(
        artifact, plate, on=date(2026, 6, 10), lineup=CARD, names={},
        opposing_starter=900, opposing_throws="R", stands=STANDS, home_team="AAA",
    )
    for p in card:
        for block in (p.hits, p.home_runs):
            assert 0.0 <= block["at_least_two"] <= block["at_least_one"] <= 1.0
            assert sum(block["distribution"].values()) == pytest.approx(1.0, abs=1e-6)
            assert block["expected"] == pytest.approx(
                sum(k * v for k, v in block["distribution"].items()), abs=1e-9
            )


def test_home_runs_are_rarer_than_hits(artifact, plate):
    card = tonight.batter_props(
        artifact, plate, on=date(2026, 6, 10), lineup=CARD, names={},
        opposing_starter=900, opposing_throws="R", stands=STANDS, home_team="AAA",
    )
    for p in card:
        assert p.home_runs["expected"] < p.hits["expected"]


# --------------------------------------------------------------------------
# Strikeouts
# --------------------------------------------------------------------------

def test_strikeout_total_is_a_distribution_not_a_point(artifact, plate):
    k = tonight.starter_strikeouts(
        artifact, plate, on=date(2026, 6, 10), pitcher_id=900, name="",
        opposing_lineup=CARD, stands=STANDS, throws="R", expected_bf=22,
    )
    assert sum(k.distribution.values()) == pytest.approx(1.0, abs=1e-4)
    assert k.at_least(1) > k.at_least(5) > k.at_least(12)


def test_the_line_splits_the_distribution(artifact, plate):
    """The half-integer where the chance of going over first drops below half."""
    k = tonight.starter_strikeouts(
        artifact, plate, on=date(2026, 6, 10), pitcher_id=900, name="",
        opposing_lineup=CARD, stands=STANDS, throws="R", expected_bf=22,
    )
    assert k.at_least(int(k.line) + 1) < 0.5
    assert k.at_least(int(k.line)) >= 0.5


def test_a_missing_pitcher_returns_nothing_rather_than_a_guess(artifact, plate):
    assert tonight.starter_strikeouts(
        artifact, plate, on=date(2026, 6, 10), pitcher_id=None, name="",
        opposing_lineup=CARD, stands=STANDS, throws="R",
    ) is None


# --------------------------------------------------------------------------
# First five
# --------------------------------------------------------------------------

def _first5_artifact():
    from guards_report.projections.train_props import First5Artifact

    return First5Artifact(
        columns=["off_rpg", "is_home"],
        coef=[0.02, 0.05],
        mean=[4.5, 0.5],
        alpha=0.45,
        share=5.100 / 8.99,
    )


def test_first_five_outcomes_form_a_distribution():
    result = tonight.first_five(
        _first5_artifact(),
        {"league_rpg": 4.5, "home_off_rpg": 4.8, "home_is_home": 1.0,
         "away_off_rpg": 4.2, "away_is_home": 0.0},
    )
    total = result.home_leads + result.tied + result.away_leads
    assert total == pytest.approx(1.0, abs=1e-6)
    assert result.coherent


def test_first_five_keeps_ties_rather_than_resolving_them():
    """A full game has no ties; five innings genuinely can be level."""
    result = tonight.first_five(
        _first5_artifact(),
        {"league_rpg": 4.5, "home_off_rpg": 4.5, "home_is_home": 1.0,
         "away_off_rpg": 4.5, "away_is_home": 0.0},
    )
    assert result.tied > 0.05, "level after five must be a real outcome"


def test_a_missing_artifact_returns_nothing(plate):
    assert tonight.first_five(None, {}) is None
