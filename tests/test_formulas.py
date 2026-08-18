"""Tests for the hard-coded metric formulas.

Three kinds of assertion here, in increasing order of what they prove:

1. Hand-computable arithmetic -- catches transcription errors in a formula.
2. Structural invariants -- e.g. a pitcher whose line *is* the league line must
   have FIP exactly equal to league ERA. These catch scaling and sign errors
   that hand-picked examples can miss.
3. Golden values from documented real games.

The strongest check of all lives in test_source_agreement.py, which asserts our
formulas reproduce the source API's own published rates from its own counting
stats. That one needs network, so it is kept separate.
"""

from __future__ import annotations

import pytest

from guards_report.metrics import formulas as f


# ---------------------------------------------------------------------------
# Innings pitched notation -- the single most common source of silent error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("notation", "expected_outs"),
    [
        ("0.0", 0),
        ("0.1", 1),
        ("0.2", 2),
        ("1.0", 3),
        ("5.1", 16),
        ("6.2", 20),
        ("9.0", 27),
        ("200.1", 601),
        (7, 21),
        (7.0, 21),
    ],
)
def test_ip_to_outs(notation, expected_outs):
    assert f.ip_to_outs(notation) == expected_outs


def test_ip_notation_is_not_decimal():
    """5.1 IP is five and one third innings, not 5.1 innings.

    If this ever regresses to float arithmetic, every rate stat with an
    innings denominator silently shifts. Pinning it explicitly.
    """
    assert f.outs_to_innings(f.ip_to_outs("5.1")) == pytest.approx(5.3333333, abs=1e-6)
    assert f.outs_to_innings(f.ip_to_outs("5.1")) != 5.1


@pytest.mark.parametrize("bad", ["5.3", "5.9", "5.45"])
def test_ip_to_outs_rejects_impossible_notation(bad):
    """There is no such thing as 5.3 innings pitched -- fail loudly."""
    with pytest.raises(ValueError):
        f.ip_to_outs(bad)


@pytest.mark.parametrize("outs", range(0, 40))
def test_outs_ip_roundtrip(outs):
    assert f.ip_to_outs(f.outs_to_ip(outs)) == outs


# ---------------------------------------------------------------------------
# Undefined rates must be None, never zero
# ---------------------------------------------------------------------------


def test_zero_denominators_return_none():
    """A player with no PA has an undefined rate, not a rate of zero.

    Rendering 0.000 for "hasn't batted yet" would be a false claim, so the
    contract is None and the renderer shows a dash.
    """
    assert f.k_pct(0, 0) is None
    assert f.bb_pct(0, 0) is None
    assert f.batting_average(0, 0) is None
    assert f.on_base_pct(0, 0, 0, 0, 0) is None
    assert f.slugging(0, 0) is None
    assert f.era(0, 0) is None
    assert f.whip(0, 0, 0) is None
    assert f.babip(0, 0, 0, 0, 0) is None
    assert f.siera(0, 0, 0, 0, 0, 0) is None


def test_zero_earned_runs_is_zero_not_none():
    """Distinguish "0.00 ERA over 6 innings" from "no innings pitched"."""
    assert f.era(0, 18) == 0.0
    assert f.era(0, 0) is None


# ---------------------------------------------------------------------------
# Hand-computable arithmetic
# ---------------------------------------------------------------------------


def test_slash_line_hand_computed():
    # 100 AB, 30 H (5 2B, 1 3B, 4 HR), 10 BB, 2 HBP, 1 SF
    hits, doubles, triples, home_runs = 30, 5, 1, 4
    singles_ = f.singles(hits, doubles, triples, home_runs)
    assert singles_ == 20

    tb = f.total_bases(singles_, doubles, triples, home_runs)
    assert tb == 20 + 10 + 3 + 16  # 49

    assert f.batting_average(hits, 100) == pytest.approx(0.300)
    assert f.slugging(tb, 100) == pytest.approx(0.490)
    # (30 + 10 + 2) / (100 + 10 + 2 + 1) = 42 / 113
    assert f.on_base_pct(hits, 10, 2, 100, 1) == pytest.approx(42 / 113)
    assert f.iso(0.490, 0.300) == pytest.approx(0.190)


def test_babip_removes_home_runs_from_both_halves():
    # 30 H incl 4 HR, 100 AB, 25 K, 1 SF -> (30-4) / (100-25-4+1) = 26/72
    assert f.babip(30, 4, 100, 25, 1) == pytest.approx(26 / 72)


def test_whip_and_era_use_true_innings():
    # 5.1 IP = 16 outs = 5.333 innings; 2 ER; 3 BB + 2 H = 5
    outs = f.ip_to_outs("5.1")
    assert f.era(2, outs) == pytest.approx(9.0 * 2 / (16 / 3))
    assert f.whip(3, 2, outs) == pytest.approx(5 / (16 / 3))


def test_per_nine():
    assert f.per_nine(9, 27) == pytest.approx(9.0)  # 9 K in 9 IP -> 9.0 K/9
    assert f.per_nine(6, f.ip_to_outs("5.1")) == pytest.approx(9.0 * 6 / (16 / 3))


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


def test_league_fip_equals_league_era_by_construction():
    """FIP is defined so that the league's FIP equals the league's ERA.

    Feeding the league totals back in as if they were one pitcher must return
    lgERA exactly. This catches sign errors and coefficient typos that a
    single hand-picked example would sail past.
    """
    lg = dict(
        lg_era=4.15,
        lg_home_runs=5_215,
        lg_walks=15_400,
        lg_hit_by_pitch=1_950,
        lg_strikeouts=41_000,
        lg_outs=130_000,
    )
    c = f.fip_constant(**lg)
    assert c is not None

    league_as_pitcher = f.fip(
        home_runs=lg["lg_home_runs"],
        walks=lg["lg_walks"],
        hit_by_pitch=lg["lg_hit_by_pitch"],
        strikeouts=lg["lg_strikeouts"],
        outs=lg["lg_outs"],
        fip_constant_=c,
    )
    assert league_as_pitcher == pytest.approx(lg["lg_era"], abs=1e-9)


def test_fip_rewards_strikeouts_and_punishes_home_runs():
    base = dict(walks=20, hit_by_pitch=2, outs=180, fip_constant_=3.10)
    more_k = f.fip(home_runs=8, strikeouts=90, **base)
    fewer_k = f.fip(home_runs=8, strikeouts=60, **base)
    more_hr = f.fip(home_runs=15, strikeouts=90, **base)
    assert more_k < fewer_k
    assert more_hr > more_k


def test_xfip_matches_fip_when_hr_equals_expected_hr():
    """If a pitcher allowed exactly the league rate of homers on their fly
    balls, xFIP and FIP must agree."""
    fly_balls, lg_hr_per_fb = 100, 0.13
    expected_hr = fly_balls * lg_hr_per_fb  # 13.0
    shared = dict(walks=20, hit_by_pitch=2, strikeouts=90, outs=180)
    c = 3.10

    from_fip = f.fip(home_runs=expected_hr, fip_constant_=c, **shared)
    from_xfip = f.xfip(
        fly_balls=fly_balls, lg_hr_per_fb=lg_hr_per_fb, fip_constant_=c, **shared
    )
    assert from_xfip == pytest.approx(from_fip)


def test_siera_sign_flip_is_continuous_at_zero():
    """The piecewise squared term must not create a discontinuity: at exactly
    net-neutral batted-ball profile both branches give the same value."""
    balanced = f.siera(
        strikeouts=100, walks=30, ground_balls=50, fly_balls=40, pop_ups=10,
        plate_appearances=400,
    )
    assert balanced is not None
    # ground_balls - fly_balls - pop_ups == 0, so the squared term vanishes
    # regardless of which coefficient was selected.
    assert balanced == pytest.approx(
        6.145 - 16.986 * 0.25 + 11.434 * 0.075 + 7.653 * 0.25**2
    )


def test_siera_improves_with_strikeouts():
    common = dict(walks=30, ground_balls=120, fly_balls=80, pop_ups=20,
                  plate_appearances=500)
    assert f.siera(strikeouts=150, **common) < f.siera(strikeouts=80, **common)


# ---------------------------------------------------------------------------
# Golden values from documented real games
# ---------------------------------------------------------------------------


def test_game_score_kerry_wood_20_strikeouts():
    """Kerry Wood, 1998-05-06 vs Houston: 9 IP, 1 H, 0 R, 0 BB, 20 K.

    Game Score 105 -- the highest ever recorded for a nine-inning start.
    """
    assert (
        f.game_score_v1(
            outs=27, strikeouts=20, hits=1, earned_runs=0, unearned_runs=0, walks=0
        )
        == 105
    )


def test_game_score_v1_average_start_lands_near_50():
    """Roughly league-average line: 6 IP, 6 H, 3 ER, 2 BB, 5 K."""
    score = f.game_score_v1(
        outs=18, strikeouts=5, hits=6, earned_runs=3, unearned_runs=0, walks=2
    )
    # 50 + 18 + 4 + 5 - 12 - 12 - 0 - 2
    assert score == 51


def test_game_score_v2_perfect_game_ceiling():
    """27 up, 27 down, 27 K: 40 + 54 + 27 = 121, the theoretical maximum."""
    assert (
        f.game_score_v2(
            outs=27, strikeouts=27, unintentional_walks=0, hits=0, runs=0,
            home_runs=0,
        )
        == 121
    )


def test_game_score_v2_average_start_lands_near_50():
    score = f.game_score_v2(
        outs=18, strikeouts=5, unintentional_walks=2, hits=6, runs=3, home_runs=1
    )
    # 40 + 36 + 5 - 4 - 12 - 9 - 6
    assert score == 50


def test_published_ops_uses_sum_of_rounded_components():
    """Regression: MLB adds rounded OBP and SLG rather than rounding the sum.

    Elly De La Cruz's 2026 line as of 2026-08-17: OBP 162/473, SLG 196/416.
    Full precision sums to .813649 (rounds to .814); MLB publishes .813.
    """
    obp, slg = 162 / 473, 196 / 416
    assert round(f.ops(obp, slg), 3) == 0.814
    assert round(f.published_ops(obp, slg), 3) == 0.813
