"""The rating refresh must stay leak-free and reproducible.

Bringing ratings up to date is the one place in the projection path that reads
games the model was never fitted on, which makes it the one place a future
result could slip in. A rating that has already seen tonight's game would make
the projection look far better than it is, and nothing downstream would notice.

These tests use a small synthetic corpus so the arithmetic is checkable by hand.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from guards_report.projections import refresh
from guards_report.projections.model import OutcomeModel


def _model() -> OutcomeModel:
    return OutcomeModel(
        elo_ratings={"1": 1500.0, "2": 1500.0},
        elo_params={"k": 4.0, "hfa": 24.0, "carry": 0.70, "mov": True},
        off_def={"1": [0.0, 0.0], "2": [0.0, 0.0]},
        league_rpg=4.5,
        corpus_through="2025-09-28",
    )


def _games(dates: list[str]) -> pd.DataFrame:
    """Team 1 beats team 2 at home, once per date given."""
    columns = [
        "game_pk", "game_date", "season", "game_type", "venue_id",
        "home_team_id", "away_team_id", "home_runs", "away_runs", "home_win",
        "home_starter_id", "away_starter_id",
    ]
    if not dates:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([
        {
            "game_pk": 1000 + i, "game_date": d, "season": int(d[:4]),
            "game_type": "R", "venue_id": 5,
            "home_team_id": 1, "away_team_id": 2,
            "home_runs": 6, "away_runs": 2, "home_win": 1,
            "home_starter_id": 10, "away_starter_id": 20,
        }
        for i, d in enumerate(dates)
    ])


@pytest.fixture
def patched(monkeypatch):
    """Swap the network fetch for a fixed set of games."""
    def install(frame: pd.DataFrame):
        def fake(season, *, cache_dir):
            return frame[frame["season"] == season].reset_index(drop=True)
        monkeypatch.setattr(refresh, "current_season_games", fake)
    return install


def test_games_on_or_after_the_projection_date_are_never_applied(patched):
    """A game played on the day being projected must not move the ratings.

    This is the leak that would matter most: the corpus lists tonight's game
    with its final score, and applying it would let the model rate a club on a
    result it is about to predict.
    """
    patched(_games(["2026-04-01", "2026-04-02", "2026-04-03"]))

    model, note = refresh.refresh(_model(), on=date(2026, 4, 3), cache_dir=None)

    assert note["games_applied"] == 2, "only games strictly before `on` may count"
    assert note["ratings_through"] == "2026-04-02"


def test_refresh_is_a_no_op_when_the_fit_already_covers_the_date(patched):
    patched(_games(["2026-04-01"]))
    original = _model()
    model, note = refresh.refresh(original, on=date(2025, 6, 1), cache_dir=None)

    assert note["games_applied"] == 0
    assert model.elo_ratings == {"1": 1500.0, "2": 1500.0}


def test_repeated_refresh_gives_identical_ratings(patched):
    """Two reports built for the same game must agree exactly."""
    patched(_games(["2026-04-01", "2026-04-02"]))

    first, _ = refresh.refresh(_model(), on=date(2026, 5, 1), cache_dir=None)
    second, _ = refresh.refresh(_model(), on=date(2026, 5, 1), cache_dir=None)

    assert first.elo_ratings == second.elo_ratings
    assert first.off_def == second.off_def
    assert first.league_rpg == second.league_rpg


def test_a_new_season_regresses_ratings_toward_the_mean(patched):
    """Rosters turn over, so a season boundary must pull ratings back.

    Without this the refresh would carry a club's September form into April as
    though nothing had changed, which is exactly what the fitted `carry`
    parameter (0.70) exists to prevent.
    """
    patched(_games([]))
    model = _model()
    model.elo_ratings = {"1": 1600.0, "2": 1400.0}

    refreshed, note = refresh.refresh(model, on=date(2026, 4, 1), cache_dir=None)

    # No games yet, so the carry is the entire update -- the opening-day case.
    assert note["games_applied"] == 0
    assert note["seasons_carried"] == 1
    # 1500 + 0.70 * (1600 - 1500) = 1570, and symmetrically 1430.
    assert refreshed.elo_ratings["1"] == pytest.approx(1570.0)
    assert refreshed.elo_ratings["2"] == pytest.approx(1430.0)


def test_winning_raises_the_winner_and_lowers_the_loser(patched):
    patched(_games(["2026-04-01"]))

    model, note = refresh.refresh(_model(), on=date(2026, 5, 1), cache_dir=None)

    assert note["games_applied"] == 1
    # Elo is zero-sum: what one side gains the other loses.
    gain = model.elo_ratings["1"] - 1500.0 * 0.70 - 1500.0 * 0.30
    loss = 1500.0 - model.elo_ratings["2"]
    assert model.elo_ratings["1"] > model.elo_ratings["2"]
    assert gain == pytest.approx(loss, abs=1e-9)
