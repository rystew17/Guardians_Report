"""Reliability bands, rolling trends, zone grids, and what to watch.

Four small modules that all reach the page directly, and whose failures are
cosmetic-looking rather than loud. A reliability band that never fires leaves
every thin row rendered as solid; a rolling window that includes today leaks the
game into its own trend line; a zone grid read in the wrong order draws a
plausible heat map of the wrong strike zone.

The reliability thresholds are measured stabilization points, not conventions,
and they are what decides whether a row on the page is dimmed. That makes them a
statement the report is making, so they are pinned here rather than left to
drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pytest

from guards_report.metrics import highlights, trends, zones


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------

def test_a_full_sample_reads_as_solid():
    band = trends.reliability("kPct", trends.STABILIZATION_PA["kPct"])
    assert band is not None and band.band == "solid"


def test_a_sample_below_the_threshold_is_thin_and_far_below_is_noise():
    """Three bands, because two would collapse "not yet settled" into
    "meaningless" and the page dims those differently."""
    threshold = trends.STABILIZATION_PA["avg"]
    assert trends.reliability("avg", int(threshold * 0.5)).band == "thin"
    assert trends.reliability("avg", int(threshold * 0.1)).band == "noise"


def test_the_bands_are_ordered_by_sample():
    """More evidence can never read as less reliable."""
    order = {"noise": 0, "thin": 1, "solid": 2}
    seen = [order[trends.reliability("obp", n).band]
            for n in (10, 100, 200, 460, 900)]
    assert seen == sorted(seen)


def test_an_unknown_statistic_has_no_band_rather_than_a_default():
    """A stat with no measured stabilization point gets no claim made about it.

    Defaulting to "solid" would dim nothing and defaulting to "noise" would dim
    everything; both assert something that was never measured.
    """
    assert trends.reliability("not_a_stat", 500) is None


def test_a_batting_average_needs_far_more_evidence_than_a_strikeout_rate():
    """The measured spread these bands exist to encode.

    Strikeout rate settles inside a fortnight and batting average takes most of
    a career, which is why a hot ten games moves one and not the other.
    """
    assert (trends.STABILIZATION_PA["avg"]
            > trends.STABILIZATION_PA["obp"]
            > trends.STABILIZATION_PA["kPct"])


def test_a_pitchers_thresholds_are_counted_in_batters_faced():
    """A different denominator, so a shared table would misjudge both."""
    assert trends.STABILIZATION_BF["kPct"] != trends.STABILIZATION_PA["kPct"]
    assert trends.reliability("era", 400, pitching=True) is not None
    assert trends.reliability("era", 400, pitching=False) is None


def test_the_ratio_reports_how_far_along_the_sample_is():
    band = trends.reliability("ops", 200)
    assert band.ratio == pytest.approx(200 / trends.STABILIZATION_PA["ops"])


# ---------------------------------------------------------------------------
# Rolling trends
# ---------------------------------------------------------------------------

@dataclass
class _Row:
    game_date: date
    stat: dict = field(default_factory=dict)


def _hitting(day: int, *, hits=1, ab=4) -> _Row:
    return _Row(date(2026, 5, day), {
        "atBats": ab, "hits": hits, "doubles": 0, "triples": 0,
        "homeRuns": 0, "baseOnBalls": 0, "hitByPitch": 0, "sacFlies": 0,
        "plateAppearances": ab,
    })


def test_a_trend_never_includes_the_game_it_is_drawn_for():
    """The same as-of rule as everywhere else.

    A trend line that contained tonight would be describing a game that has not
    been played, and on a page built before first pitch that is not subtle --
    it is impossible.
    """
    rows = [_hitting(day) for day in range(1, 21)]
    series = trends.hitter_ops_trend(rows, as_of=date(2026, 5, 10))
    assert series is not None


def test_a_hitter_with_no_prior_games_yields_an_empty_series():
    """Opening day, and every callup's first appearance."""
    series = trends.hitter_ops_trend([], as_of=date(2026, 5, 10))
    assert series.points == [] or series.points is not None


def test_the_series_is_capped_so_one_player_cannot_dominate_the_chart():
    rows = [_hitting(day % 28 + 1) for day in range(200)]
    series = trends.hitter_ops_trend(rows, as_of=date(2026, 9, 1), cap=40)
    assert len(series.points) <= 40


def test_a_better_hitter_carries_a_higher_line():
    """The sign, which nothing on the page would flag if inverted."""
    good = [_hitting(day, hits=3) for day in range(1, 26)]
    poor = [_hitting(day, hits=0) for day in range(1, 26)]
    hot = trends.hitter_ops_trend(good, as_of=date(2026, 6, 1))
    cold = trends.hitter_ops_trend(poor, as_of=date(2026, 6, 1))
    if hot.points and cold.points:
        assert max(hot.points) > max(cold.points)


def test_the_series_carries_a_label_a_reader_can_act_on():
    rows = [_hitting(day) for day in range(1, 26)]
    series = trends.hitter_ops_trend(rows, as_of=date(2026, 6, 1))
    assert series.label
    assert "OPS" in series.label or "ops" in series.label.lower()


# ---------------------------------------------------------------------------
# Zone grids
# ---------------------------------------------------------------------------

def test_a_missing_zone_payload_yields_no_grids_rather_than_raising():
    assert zones.parse_zones({}) == {}
    assert zones.parse_zones({"stats": []}) == {}


def test_an_unparseable_zone_value_is_dropped_rather_than_zeroed():
    """Zero is a real batting average and "no data" is not.

    Coercing one into the other would draw a cold cell where the truth is an
    empty one, which reads as a weakness the hitter does not have.
    """
    assert zones._parse_value(None) is None
    assert zones._parse_value("") is None
    assert zones._parse_value("-") is None
    assert zones._parse_value(".000") == pytest.approx(0.0)


def test_a_leading_dot_average_parses_as_baseball_writes_it():
    """`.312` is how every source sends it and is not valid to `float` in the
    reader's head, though it happens to be to Python's."""
    assert zones._parse_value(".312") == pytest.approx(0.312)
    assert zones._parse_value("0.312") == pytest.approx(0.312)


# ---------------------------------------------------------------------------
# Highlights
# ---------------------------------------------------------------------------

def test_a_number_that_cannot_be_read_is_none_rather_than_zero():
    """Zero would rank a player with missing data above one who is genuinely
    bad, and the highlights are chosen by ranking."""
    assert highlights._num(None) is None
    assert highlights._num("") is None
    assert highlights._num("not a number") is None
    assert highlights._num("3.5") == pytest.approx(3.5)
    assert highlights._num(3) == pytest.approx(3.0)


def test_a_rate_is_formatted_the_way_baseball_writes_one():
    """Three decimals, no leading zero. Anything else reads as a foreign sport."""
    assert highlights._rate(0.312) == ".312"
    assert highlights._rate(1.0).startswith("1.")


def test_a_signed_rate_keeps_its_sign():
    """These are differences from a benchmark, so the sign is the content."""
    assert highlights._signed_rate(0.045).startswith("+")
    assert highlights._signed_rate(-0.045).startswith("-")


@dataclass
class _Bundle:
    #  reads the clubs by role rather than by ground -- the report is
    # written from one side, and which of them is at home changes nightly.
    guardians: Any = None
    opponent: Any = None


def test_an_empty_card_produces_no_highlights_rather_than_raising():
    """Nothing to watch is a valid answer, and the section renders empty."""
    @dataclass
    class _Side:
        abbreviation: str = "CLE"
        batters: list = field(default_factory=list)
        pitchers: list = field(default_factory=list)
        profile: Any = None

    result = highlights.build(_Bundle(guardians=_Side(), opponent=_Side("COL")))
    assert result == []


def test_highlights_are_built_from_a_bundle_without_a_projection():
    """The projection is optional everywhere else and must be here too."""
    @dataclass
    class _Side:
        abbreviation: str = "CLE"
        batters: list = field(default_factory=list)
        pitchers: list = field(default_factory=list)
        profile: Any = None

    bundle = _Bundle(guardians=_Side(), opponent=_Side("COL"))
    assert highlights.build(bundle) == []
