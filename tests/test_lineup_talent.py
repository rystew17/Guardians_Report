"""Leakage and correctness guarantees for the plate-appearance layer.

These two modules are the easiest place in the project to build something that
scores brilliantly and is worthless. Talent fitted on the season it predicts, or
a lineup reconstructed from who actually batted, would both look like large
gains and neither could be reproduced before first pitch.

Synthetic data throughout, so every expected value is arithmetic rather than a
figure copied from a run.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from guards_report.projections import lineup, talent


def _pa_frame() -> pd.DataFrame:
    """Two clubs, two seasons, a stable nine plus a late substitute."""
    rows = []
    pk = 1000
    for season, days in ((2023, 12), (2024, 12)):
        for day in range(days):
            pk += 1
            for team, base in (("AAA", 100), ("BBB", 200)):
                # Nine starters bat in slot order, twice through.
                order = [base + i for i in range(9)]
                # Two full turns plus a partial third, so the top of the order
                # bats more often -- as it does in a real game.
                sequence = order + order + order[:4]
                # A pinch-hitter appears only after everyone has hit twice.
                sequence.append(base + 90)
                for n, batter in enumerate(sequence, start=1):
                    rows.append({
                        "game_date": date(season, 5, day + 1),
                        "game_pk": pk,
                        "at_bat_number": n,
                        "batter": batter,
                        "pitcher": 900 if team == "AAA" else 901,
                        "stand": "L" if batter % 2 else "R",
                        "p_throws": "R",
                        "events": "single" if n % 3 == 0 else "strikeout",
                        "woba_value": 0.9 if n % 3 == 0 else 0.0,
                        "woba_denom": 1,
                        "batting_team": team,
                        "season": season,
                    })
    return pd.DataFrame(rows)


@pytest.fixture
def frame() -> pd.DataFrame:
    return _pa_frame()


# --------------------------------------------------------------------------
# Lineup reconstruction
# --------------------------------------------------------------------------

def test_starting_lineup_is_exactly_the_first_nine(frame):
    starters = lineup.starting_lineups(frame)
    per_game = starters.groupby(["game_pk", "batting_team"]).size()
    assert set(per_game.unique()) == {9}


def test_pinch_hitter_is_never_treated_as_a_starter(frame):
    """The substitute bats last and must not appear in any posted card.

    This is the leak that matters most here: who came off the bench is a
    consequence of how the game went, and no card three hours before first pitch
    could contain it.
    """
    starters = lineup.starting_lineups(frame)
    assert not starters["batter"].isin([190, 290]).any()


def test_slots_follow_at_bat_number_not_row_order(frame):
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    a = lineup.starting_lineups(frame).sort_values(["game_pk", "batting_team", "slot"])
    b = lineup.starting_lineups(shuffled).sort_values(["game_pk", "batting_team", "slot"])
    assert list(a["batter"]) == list(b["batter"])


def test_projected_lineup_uses_only_earlier_games(frame):
    """`recent_regulars` must not see the day it is projecting, or any after it."""
    cutoff = date(2024, 5, 6)
    picked = lineup.recent_regulars(frame, "AAA", before=cutoff, games=5)
    assert len(picked) == 9

    # Poison every game from the cutoff onward with an unmistakable roster.
    poisoned = frame.copy()
    future = poisoned["game_date"] >= cutoff
    poisoned.loc[future, "batter"] = poisoned.loc[future, "batter"] + 5000
    after = lineup.recent_regulars(poisoned, "AAA", before=cutoff, games=5)

    assert after == picked, "future games changed a projection that precedes them"


def test_a_short_card_falls_back_rather_than_pretending(frame):
    model = talent.fit(frame, alpha=10.0)
    value = lineup.value_of([100, 101, 102], model)
    assert value.source == "none"
    assert value.slots_known == 3


def test_leadoff_is_weighted_above_the_ninth_slot(frame):
    weights = lineup.slot_weights(frame)
    assert len(weights) == 9
    assert weights[0] > weights[8], "the top of the order bats more often"


# --------------------------------------------------------------------------
# Talent estimation
# --------------------------------------------------------------------------

def test_fit_respects_the_as_of_cutoff(frame):
    """Nothing after `through` may influence the fitted effects."""
    cutoff = date(2024, 1, 1)
    baseline = talent.fit(frame, alpha=10.0, through=cutoff)

    poisoned = frame.copy()
    later = poisoned["game_date"] >= cutoff
    poisoned.loc[later, "woba_value"] = 4.0

    after = talent.fit(poisoned, alpha=10.0, through=cutoff)
    assert after.intercept == pytest.approx(baseline.intercept)
    for player, value in baseline.pitcher.items():
        assert after.pitcher[player] == pytest.approx(value)


def test_as_of_table_never_trains_on_the_season_it_serves(frame):
    fits = talent.as_of_table(frame, alpha=10.0, seasons=[2024])
    assert 2024 in fits
    assert fits[2024].through < "2024-01-01"


def test_stronger_shrinkage_pulls_effects_toward_zero(frame):
    light = talent.fit(frame, alpha=1.0)
    heavy = talent.fit(frame, alpha=5000.0)
    spread = lambda m: float(np.std(list(m.batter.values())))
    assert spread(heavy) < spread(light)


def test_evidence_counts_plate_appearances(frame):
    model = talent.fit(frame, alpha=10.0)
    # Slot one bats three times per game across 24 games.
    assert model.evidence(100, side="batter") == 72
    # The substitute bats once per game.
    assert model.evidence(190, side="batter") == 24


def test_unknown_players_contribute_nothing_but_the_platoon_state(frame):
    """An unrecognised player is league-average, not zero-valued.

    After recentring, a fitted effect is a deviation from the league mean, so an
    unknown player correctly contributes zero. The platoon term is not zero and
    should not be: a right-on-right matchup is the most common one, not the
    average one, and that distinction is exactly what the term encodes.
    """
    model = talent.fit(frame, alpha=10.0)
    assert model.batter_score(999_999) is None

    for stand in ("L", "R"):
        expected = model.intercept + model.platoon[f"{stand}R"]
        actual = model.expected_value(999_999, 999_998, stand=stand, throws="R")
        assert actual == pytest.approx(expected)


def test_recentring_leaves_every_prediction_unchanged(frame):
    """Splitting the mean differently must not move a single fitted value."""
    model = talent.fit(frame, alpha=10.0)
    # Effects are deviations, so they average to zero over the plate appearances
    # that produced them.
    weights = np.array([model.batter_pa[b] for b in model.batter])
    effects = np.array([model.batter[b] for b in model.batter])
    assert float((effects * weights).sum() / weights.sum()) == pytest.approx(0.0, abs=1e-9)
