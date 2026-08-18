"""Deterministic baseball metric calculations.

Every number the report displays that is not verbatim from a source API is
computed here. Rules for this module:

* Pure functions only. No I/O, no network, no global state, no LLM.
* Every function takes raw counting stats and returns a float or None.
* A None return means "not computable from these inputs" (zero denominator).
  We never substitute 0.0 for an undefined rate -- a 0.000 ERA and an
  undefined ERA are different claims, and the report must not conflate them.
* Rates are always computed from summed counting stats, never by averaging
  rates. See metrics/windows.py for why that matters over a date window.

Formula definitions are documented per function so any number in the report
can be re-derived by hand from the raw stat line.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Innings pitched
# ---------------------------------------------------------------------------
# MLB reports innings pitched in the "5.1 = five and one third" convention,
# where the digit after the decimal counts outs, not tenths. Treating "5.1" as
# the float 5.1 understates ERA/FIP/WHIP denominators by up to ~4%. Everything
# in this module works in whole outs internally and converts at the edges.


def ip_to_outs(ip: float | str) -> int:
    """Convert MLB innings-pitched notation to a whole number of outs.

    ip_to_outs("5.1") == 16
    ip_to_outs("6.2") == 20
    ip_to_outs(7) == 21
    """
    text = str(ip).strip()
    if "." in text:
        whole_text, frac_text = text.split(".", 1)
    else:
        whole_text, frac_text = text, "0"

    whole = int(whole_text) if whole_text else 0
    if frac_text not in ("0", "1", "2", ""):
        raise ValueError(
            f"invalid innings-pitched notation {ip!r}: the digit after the "
            "decimal counts outs and must be 0, 1, or 2"
        )
    partial = int(frac_text) if frac_text else 0
    return whole * 3 + partial


def outs_to_ip(outs: int) -> float:
    """Convert whole outs back to MLB innings-pitched notation, for display."""
    return outs // 3 + (outs % 3) / 10.0


def outs_to_innings(outs: int) -> float:
    """Convert whole outs to true decimal innings, for use in rate math."""
    return outs / 3.0


def _rate(numerator: float, denominator: float) -> float | None:
    """Divide, or return None when the rate is undefined."""
    if denominator == 0:
        return None
    return numerator / denominator


# ---------------------------------------------------------------------------
# Plate-appearance rate stats
# ---------------------------------------------------------------------------


def k_pct(strikeouts: int, plate_appearances: int) -> float | None:
    """Strikeout rate, K/PA. The standard denominator is PA, not AB."""
    return _rate(strikeouts, plate_appearances)


def bb_pct(walks: int, plate_appearances: int) -> float | None:
    """Walk rate, BB/PA. Includes intentional walks, matching convention."""
    return _rate(walks, plate_appearances)


def k_minus_bb_pct(
    strikeouts: int, walks: int, plate_appearances: int
) -> float | None:
    """K% minus BB%. Undefined when PA is zero."""
    k = k_pct(strikeouts, plate_appearances)
    bb = bb_pct(walks, plate_appearances)
    if k is None or bb is None:
        return None
    return k - bb


# ---------------------------------------------------------------------------
# Slash line
# ---------------------------------------------------------------------------


def batting_average(hits: int, at_bats: int) -> float | None:
    return _rate(hits, at_bats)


def singles(hits: int, doubles: int, triples: int, home_runs: int) -> int:
    """Singles are not reported directly; they are backed out of the others."""
    return hits - doubles - triples - home_runs


def total_bases(singles_: int, doubles: int, triples: int, home_runs: int) -> int:
    return singles_ + 2 * doubles + 3 * triples + 4 * home_runs


def slugging(total_bases_: int, at_bats: int) -> float | None:
    return _rate(total_bases_, at_bats)


def on_base_pct(
    hits: int, walks: int, hit_by_pitch: int, at_bats: int, sac_flies: int
) -> float | None:
    """OBP = (H + BB + HBP) / (AB + BB + HBP + SF).

    The denominator includes sacrifice flies but excludes sacrifice bunts.
    """
    numerator = hits + walks + hit_by_pitch
    denominator = at_bats + walks + hit_by_pitch + sac_flies
    return _rate(numerator, denominator)


def ops(obp: float | None, slg: float | None) -> float | None:
    """OPS at full precision. Use this for any further arithmetic."""
    if obp is None or slg is None:
        return None
    return obp + slg


def published_ops(obp: float | None, slg: float | None) -> float | None:
    """OPS as MLB publishes it: the sum of the *rounded* components.

    MLB rounds OBP and SLG to three places and then adds, which can differ by
    a point from rounding the full-precision sum. Elly De La Cruz on
    2026-08-17 is a worked example: OBP .342495 and SLG .471154 sum to .813649,
    which rounds to .814, but MLB publishes .813 because it adds .342 + .471.

    Neither is wrong; they are different conventions. The report displays this
    one so that every figure matches what the reader sees on mlb.com when they
    go to check it, which is the whole point of the verifiability rule.
    """
    if obp is None or slg is None:
        return None
    return round(obp, 3) + round(slg, 3)


def iso(slg: float | None, avg: float | None) -> float | None:
    """Isolated power: extra bases per at-bat, SLG - AVG."""
    if slg is None or avg is None:
        return None
    return slg - avg


def babip(
    hits: int, home_runs: int, at_bats: int, strikeouts: int, sac_flies: int
) -> float | None:
    """Batting average on balls in play.

    BABIP = (H - HR) / (AB - K - HR + SF)

    Home runs come out of both halves because they are never "in play".
    """
    return _rate(hits - home_runs, at_bats - strikeouts - home_runs + sac_flies)


# ---------------------------------------------------------------------------
# Pitching
# ---------------------------------------------------------------------------


def era(earned_runs: int, outs: int) -> float | None:
    """Earned run average, per nine innings."""
    return _rate(9.0 * earned_runs, outs_to_innings(outs))


def whip(walks: int, hits: int, outs: int) -> float | None:
    """Walks plus hits per inning pitched. Excludes HBP by definition."""
    return _rate(walks + hits, outs_to_innings(outs))


def per_nine(count: int, outs: int) -> float | None:
    """Generic per-nine-innings rate, for K/9, BB/9, HR/9."""
    return _rate(9.0 * count, outs_to_innings(outs))


def fip_constant(
    lg_era: float,
    lg_home_runs: int,
    lg_walks: int,
    lg_hit_by_pitch: int,
    lg_strikeouts: int,
    lg_outs: int,
) -> float | None:
    """The league-specific additive constant that puts FIP on the ERA scale.

    cFIP = lgERA - ((13*HR + 3*(BB+HBP) - 2*K) / IP)

    Derived from the season's actual league totals rather than hardcoded, so
    FIP stays on the correct scale for that year's run environment.
    """
    raw = _rate(
        13 * lg_home_runs + 3 * (lg_walks + lg_hit_by_pitch) - 2 * lg_strikeouts,
        outs_to_innings(lg_outs),
    )
    if raw is None:
        return None
    return lg_era - raw


def fip(
    home_runs: int,
    walks: int,
    hit_by_pitch: int,
    strikeouts: int,
    outs: int,
    fip_constant_: float,
) -> float | None:
    """Fielding Independent Pitching.

    FIP = (13*HR + 3*(BB+HBP) - 2*K) / IP + cFIP

    Uses only the outcomes that do not involve fielders.
    """
    raw = _rate(
        13 * home_runs + 3 * (walks + hit_by_pitch) - 2 * strikeouts,
        outs_to_innings(outs),
    )
    if raw is None:
        return None
    return raw + fip_constant_


def xfip(
    fly_balls: int,
    walks: int,
    hit_by_pitch: int,
    strikeouts: int,
    outs: int,
    lg_hr_per_fb: float,
    fip_constant_: float,
) -> float | None:
    """Expected FIP: FIP with home runs allowed replaced by the pitcher's own
    fly balls times the league home-run-per-fly-ball rate.

    xFIP = (13*(FB * lgHR/FB) + 3*(BB+HBP) - 2*K) / IP + cFIP

    `fly_balls` must be actual fly balls allowed, a batted-ball count. The MLB
    Stats API reports fly *outs*, which is a different and smaller number;
    passing those in silently understates the home-run term.
    """
    raw = _rate(
        13 * (fly_balls * lg_hr_per_fb)
        + 3 * (walks + hit_by_pitch)
        - 2 * strikeouts,
        outs_to_innings(outs),
    )
    if raw is None:
        return None
    return raw + fip_constant_


def siera(
    strikeouts: int,
    walks: int,
    ground_balls: int,
    fly_balls: int,
    pop_ups: int,
    plate_appearances: int,
) -> float | None:
    """Skill-Interactive ERA (Swartz).

    Requires true batted-ball classifications. The MLB Stats API game log
    exposes groundOuts and airOuts, which are *outs*, not batted-ball counts --
    feeding those in here produces a wrong number. Callers must source these
    from Statcast batted-ball data.

    The squared net-ground-ball term flips sign depending on whether the
    pitcher leans ground-ball or fly-ball. That piecewise behaviour is part of
    the published specification, not a simplification on our side.
    """
    if plate_appearances == 0:
        return None

    pa = float(plate_appearances)
    so_pa = strikeouts / pa
    bb_pa = walks / pa
    net_gb_pa = (ground_balls - fly_balls - pop_ups) / pa

    squared_gb_coefficient = 6.664 if net_gb_pa < 0 else -6.664

    return (
        6.145
        - 16.986 * so_pa
        + 11.434 * bb_pa
        - 1.858 * net_gb_pa
        + 7.653 * so_pa**2
        + squared_gb_coefficient * net_gb_pa**2
        + 10.130 * so_pa * net_gb_pa
        - 5.195 * bb_pa * net_gb_pa
    )


# ---------------------------------------------------------------------------
# Game Score
# ---------------------------------------------------------------------------


def game_score_v1(
    outs: int,
    strikeouts: int,
    hits: int,
    earned_runs: int,
    unearned_runs: int,
    walks: int,
) -> int:
    """Bill James Game Score.

    Start at 50; +1 per out recorded; +2 per inning completed after the 4th;
    +1 per strikeout; -2 per hit; -4 per earned run; -2 per unearned run;
    -1 per walk.
    """
    completed_innings = outs // 3
    innings_after_fourth = max(0, completed_innings - 4)
    return (
        50
        + outs
        + 2 * innings_after_fourth
        + strikeouts
        - 2 * hits
        - 4 * earned_runs
        - 2 * unearned_runs
        - walks
    )


def game_score_v2(
    outs: int,
    strikeouts: int,
    unintentional_walks: int,
    hits: int,
    runs: int,
    home_runs: int,
) -> int:
    """Tom Tango's Game Score v2, recentred so that about 50 is average.

    Start at 40; +2 per out; +1 per strikeout; -2 per unintentional walk;
    -2 per hit; -3 per run (earned or not); -6 per home run.
    """
    return (
        40
        + 2 * outs
        + strikeouts
        - 2 * unintentional_walks
        - 2 * hits
        - 3 * runs
        - 6 * home_runs
    )
