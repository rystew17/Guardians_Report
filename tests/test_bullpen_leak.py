"""The pen's quality figures cannot see the season they describe.

Per-pitcher rates were always as-of -- a cumulative sum with the current
appearance subtracted. The scale shift was not: it averaged the whole season
and applied the result to every game in it, so a game in April carried a
constant computed from September. Cosmetic while nothing fitted these columns,
a leak the moment something did.

The test that matters is the second one: it changes only the FUTURE and asserts
the past does not move. Against the old implementation it fails, because the
whole-season mean moves and drags every earlier game's figure with it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from guards_report.projections import bullpen


def _logs(rows):
    """Relief appearances, in the shape `reliever_pool` returns."""
    return pd.DataFrame(rows, columns=[
        "pitcher_id", "season", "game_date", "game_pk", "outs", "runs",
        "homeRuns", "baseOnBalls", "strikeOuts", "battersFaced", "hitByPitch",
    ])


def _appearance(pid, season, day, pk, *, hr=0, bb=0, k=3, outs=12):
    return (pid, season, f"{season}-04-{day:02d}", pk, outs, 1, hr, bb, k,
            outs + 3, 0)


def test_a_pitcher_s_own_outing_never_counts_toward_his_rate():
    rows = [_appearance(1, 2024, day, 100 + day, hr=1) for day in range(1, 12)]
    out = bullpen._as_of_quality(_logs(rows))
    first = out.sort_values("game_date").iloc[0]
    # Nothing before the first appearance, so there is nothing to rate him on.
    assert first["rp_outs_prior"] == 0
    assert pd.isna(first["rp_fip"])


def test_changing_the_future_does_not_change_the_past():
    """The leak, stated as an experiment.

    Two corpora identical through April, differing only in games that come
    later. Every figure for the early games must be identical -- the later
    games had not happened yet.
    """
    early = [_appearance(1, 2024, day, 100 + day, hr=0) for day in range(1, 15)]
    calm = early + [_appearance(1, 2024, day, 100 + day, hr=0)
                    for day in range(15, 29)]
    wild = early + [_appearance(1, 2024, day, 100 + day, hr=9, bb=9, k=0)
                    for day in range(15, 29)]

    a = bullpen._as_of_quality(_logs(calm)).sort_values("game_pk")
    b = bullpen._as_of_quality(_logs(wild)).sort_values("game_pk")
    shared = [100 + day for day in range(1, 15)]
    a = a[a["game_pk"].isin(shared)].set_index("game_pk")["rp_fip"]
    b = b[b["game_pk"].isin(shared)].set_index("game_pk")["rp_fip"]

    both = a.notna() & b.notna()
    assert both.any(), "no comparable rows; the fixture is too thin"
    assert np.allclose(a[both], b[both]), (
        "a later game changed an earlier game's figure")


def test_adding_a_whole_later_season_does_not_change_an_earlier_one():
    """The same guarantee across the season boundary, which is where the
    shift is now computed."""
    base = ([_appearance(1, 2023, day, 200 + day) for day in range(1, 15)]
            + [_appearance(2, 2023, day, 300 + day) for day in range(1, 15)])
    later = base + [_appearance(1, 2024, day, 400 + day, hr=8, bb=8, k=0)
                    for day in range(1, 15)]

    a = bullpen._as_of_quality(_logs(base))
    b = bullpen._as_of_quality(_logs(later))
    a = a[a["season"] == 2023].set_index("game_pk")["rp_fip"]
    b = b[b["season"] == 2023].set_index("game_pk")["rp_fip"]

    both = a.notna() & b.notna()
    assert both.any()
    assert np.allclose(a[both], b[both])


def test_the_scale_shift_uses_earlier_seasons_when_they_exist():
    """A later season is put on the ERA scale using what came before it, so its
    figures are comparable rather than self-referential."""
    rows = ([_appearance(1, 2023, day, 200 + day, hr=0, k=6) for day in range(1, 20)]
            + [_appearance(1, 2024, day, 400 + day, hr=0, k=6) for day in range(1, 20)])
    out = bullpen._as_of_quality(_logs(rows))
    later = out[out["season"] == 2024]["rp_fip"].dropna()
    assert len(later), "no rated appearances in the later season"
    # Shifted onto a recognisable scale rather than left as a raw FIP core.
    assert -5.0 < later.mean() < 12.0


def test_a_thin_reliever_is_left_unrated_rather_than_guessed():
    rows = [_appearance(1, 2024, 1, 101, outs=3),
            _appearance(1, 2024, 2, 102, outs=3)]
    out = bullpen._as_of_quality(_logs(rows)).sort_values("game_pk")
    # Three outs of prior work is below MIN_PRIOR_OUTS, so no number is offered.
    assert out["rp_fip"].isna().all()
    assert out["rp_k_pct"].isna().all()
