"""The run value percentiles, checked offline against a frozen calibration.

Savant publishes these on player pages rather than through any leaderboard, so
establishing which population they rank against meant reading the pages. That
was research; this is the regression test, and it needs no network.

The fixture freezes the inputs as well as the answers -- the pool of run values,
one player's value, and the percentile Savant printed for him. That makes this a
pure-function check: given this pool and this value, the ranking must produce
this percentile. A fixture of percentiles alone would go stale within a week as
the season moved; one that carries its own inputs never does.

The live check lives beside it in `test_source_agreement.py` under the network
marker. This one guards against our code drifting, which is the likely failure.
That one guards against Savant changing, which is the rare one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "savant_run_value_calibration.json"


def _load():
    if not FIXTURE.exists():
        pytest.skip("calibration fixture not present")
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _percentile(value: float, pool: list[float]) -> float:
    """Midrank, because the run value distribution has real ties at zero.

    A strict less-than systematically under-rates everyone sitting on a tied
    value, which for fielding is a third of the league.
    """
    below = sum(1 for x in pool if x < value)
    equal = sum(1 for x in pool if x == value)
    return (below + equal / 2) / len(pool) * 100


@pytest.mark.parametrize("kind", ["batter", "pitcher"])
def test_the_ranking_reproduces_the_published_percentile(kind):
    """The calibration itself: r >= 0.99 and a small median error.

    The population was reasoned about against a single observation before the
    published figures were found, and reasoning gave the wrong answer -- a
    shrunk rate against qualified players, four points off.
    """
    data = _load()[kind]
    pool = data["pool"]
    errors = [abs(_percentile(c["value"], pool) - c["published"]) for c in data["cases"]]
    assert len(errors) >= 5, "too few frozen cases to mean anything"
    assert np.median(errors) <= 3.0, (
        f"{kind}: median {np.median(errors):.1f} points from the published figure")
    assert max(errors) <= 12.0, f"{kind}: worst case {max(errors):.1f} points"


@pytest.mark.parametrize("kind", ["batter", "pitcher"])
def test_the_ranking_tracks_the_published_order(kind):
    """Agreement on the ordering, not only on the level.

    A pool cut in the wrong place shifts every player the same way and can still
    keep the order; a wrong *unit* -- ranking a rate where Savant ranks a total,
    which is the mistake that was actually made -- reorders them.
    """
    data = _load()[kind]
    mine = [_percentile(c["value"], data["pool"]) for c in data["cases"]]
    theirs = [c["published"] for c in data["cases"]]
    assert np.corrcoef(mine, theirs)[0, 1] >= 0.99


@pytest.mark.parametrize("kind", ["batter", "pitcher"])
def test_the_frozen_pool_matches_the_floor_it_claims(kind):
    """The floor is the finding; a fixture built at a different cut proves nothing."""
    data = _load()[kind]
    assert data["floor"] in (200, 250)
    assert len(data["pool"]) >= 150


def test_the_floors_match_what_the_report_actually_uses():
    """A constant that drifts from its calibration is worse than an uncalibrated one.

    These live in `preview.py` as literals, so nothing but this connects them
    back to the evidence that chose them.
    """
    source = (Path(__file__).parents[1] / "src" / "guards_report" / "ingest"
              / "preview.py").read_text(encoding="utf-8")
    data = _load()
    assert f"QUALIFIED_PA = {data['batter']['floor']}" in source
    assert f"QUALIFIED_BF = {data['pitcher']['floor']}" in source
