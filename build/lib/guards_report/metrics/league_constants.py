"""Derive season-specific league constants from actual league totals.

FIP and xFIP are only meaningful against the run environment they were
computed in. The FIP constant drifts year to year -- it moves with league ERA,
strikeout rate, and home-run rate -- so hardcoding a remembered value would put
every FIP in the report on the wrong scale.

Everything here is computed from the 30 team season lines the source API
returns, so the constants are reproducible from archived payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from guards_report.metrics import formulas as f


@dataclass(frozen=True)
class LeagueTotals:
    """Summed pitching totals across all 30 clubs for one season."""

    season: int
    outs: int
    earned_runs: int
    home_runs: int
    walks: int
    hit_by_pitch: int
    strikeouts: int
    batters_faced: int
    team_count: int

    @property
    def innings(self) -> float:
        return f.outs_to_innings(self.outs)

    @property
    def era(self) -> float | None:
        """League ERA computed from summed totals.

        Deliberately not the mean of the 30 team ERAs. Clubs do not pitch equal
        innings, so an unweighted mean of team ERAs is not the league ERA; the
        difference is small but it would propagate into every FIP we report.
        """
        return f.era(self.earned_runs, self.outs)


@dataclass(frozen=True)
class LeagueConstants:
    season: int
    fip_constant: float
    league_era: float
    # Retained so the report's audit appendix can show what the constant was
    # derived from, rather than asking the reader to trust a bare number.
    totals: LeagueTotals


def _stat_int(stat: dict[str, Any], key: str) -> int:
    value = stat.get(key, 0)
    if value in (None, ""):
        return 0
    return int(value)


def parse_league_totals(payload: dict[str, Any], *, season: int) -> LeagueTotals:
    """Sum the per-team season pitching lines from /v1/teams/stats.

    The response holds one split per club. We require all 30 to be present:
    a partial response would silently shift every constant derived from it,
    and a wrong-but-plausible FIP is worse than a missing one.
    """
    stats_blocks = payload.get("stats") or []
    if not stats_blocks:
        raise ValueError("league pitching totals payload contained no stats block")

    splits = stats_blocks[0].get("splits") or []
    if len(splits) != 30:
        raise ValueError(
            f"expected 30 team pitching lines for {season}, got {len(splits)}; "
            "refusing to derive league constants from a partial response"
        )

    totals = dict(
        outs=0,
        earned_runs=0,
        home_runs=0,
        walks=0,
        hit_by_pitch=0,
        strikeouts=0,
        batters_faced=0,
    )
    field_map = {
        "outs": "outs",
        "earned_runs": "earnedRuns",
        "home_runs": "homeRuns",
        "walks": "baseOnBalls",
        "hit_by_pitch": "hitByPitch",
        "strikeouts": "strikeOuts",
        "batters_faced": "battersFaced",
    }

    for split in splits:
        stat = split.get("stat") or {}
        for our_name, api_name in field_map.items():
            totals[our_name] += _stat_int(stat, api_name)

    return LeagueTotals(season=season, team_count=len(splits), **totals)


def derive(totals: LeagueTotals) -> LeagueConstants:
    """Compute the FIP constant for a season from its league totals."""
    league_era = totals.era
    if league_era is None:
        raise ValueError(
            f"cannot derive league constants for {totals.season}: zero innings"
        )

    constant = f.fip_constant(
        lg_era=league_era,
        lg_home_runs=totals.home_runs,
        lg_walks=totals.walks,
        lg_hit_by_pitch=totals.hit_by_pitch,
        lg_strikeouts=totals.strikeouts,
        lg_outs=totals.outs,
    )
    if constant is None:
        raise ValueError(
            f"cannot derive FIP constant for {totals.season}: zero innings"
        )

    # The constant has sat between roughly 2.9 and 3.3 for the whole modern
    # era. A value outside this band means the totals were parsed wrong -- most
    # likely innings or outs -- and we would rather fail than publish a FIP
    # that is quietly on the wrong scale.
    if not 2.5 <= constant <= 3.7:
        raise ValueError(
            f"derived FIP constant {constant:.4f} for {totals.season} is outside "
            "the plausible range; league totals were probably parsed incorrectly"
        )

    return LeagueConstants(
        season=totals.season,
        fip_constant=constant,
        league_era=league_era,
        totals=totals,
    )
