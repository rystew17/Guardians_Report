"""League-average benchmarks, derived from actual league totals.

Every stat the report shows for a player is shown against the league average
for that stat, so a number like a .750 OPS carries its own context. Those
averages are computed here from the summed season lines of all 30 clubs -- not
looked up, not remembered, not averaged across players.

The distinction matters. The league batting average is total hits over total
at-bats, *not* the mean of individual players' averages: those differ, because
players have wildly unequal playing time, and the mean-of-players figure is
not a thing that describes the league.
"""

from __future__ import annotations

from guards_report.sources import savant as sv

from dataclasses import dataclass
from typing import Any

from guards_report.metrics import formulas as f


def _sum_field(splits: list[dict[str, Any]], key: str) -> int:
    total = 0
    for split in splits:
        value = (split.get("stat") or {}).get(key)
        if value in (None, ""):
            continue
        total += int(value)
    return total


def _splits(payload: dict[str, Any], *, expect: int = 30) -> list[dict[str, Any]]:
    blocks = payload.get("stats") or []
    if not blocks:
        raise ValueError("league totals payload contained no stats block")
    splits = blocks[0].get("splits") or []
    if len(splits) != expect:
        raise ValueError(
            f"expected {expect} team lines, got {len(splits)}; refusing to "
            "derive league averages from a partial response"
        )
    return splits


@dataclass(frozen=True)
class LeagueHitting:
    """League-wide offensive rates. Keys match aggregate_hitting output."""

    season: int
    values: dict[str, float | None]

    def get(self, key: str) -> float | None:
        return self.values.get(key)


@dataclass(frozen=True)
class LeaguePitching:
    season: int
    values: dict[str, float | None]

    def get(self, key: str) -> float | None:
        return self.values.get(key)


def hitting_from_payload(payload: dict[str, Any], *, season: int) -> LeagueHitting:
    splits = _splits(payload)

    pa = _sum_field(splits, "plateAppearances")
    ab = _sum_field(splits, "atBats")
    hits = _sum_field(splits, "hits")
    doubles = _sum_field(splits, "doubles")
    triples = _sum_field(splits, "triples")
    home_runs = _sum_field(splits, "homeRuns")
    walks = _sum_field(splits, "baseOnBalls")
    strikeouts = _sum_field(splits, "strikeOuts")
    hbp = _sum_field(splits, "hitByPitch")
    sac_flies = _sum_field(splits, "sacFlies")

    singles = f.singles(hits, doubles, triples, home_runs)
    total_bases = f.total_bases(singles, doubles, triples, home_runs)

    avg = f.batting_average(hits, ab)
    obp = f.on_base_pct(hits, walks, hbp, ab, sac_flies)
    slg = f.slugging(total_bases, ab)

    return LeagueHitting(
        season=season,
        values={
            "avg": avg,
            "obp": obp,
            "slg": slg,
            "ops": f.ops(obp, slg),
            "iso": f.iso(slg, avg),
            "babip": f.babip(hits, home_runs, ab, strikeouts, sac_flies),
            "kPct": f.k_pct(strikeouts, pa),
            "bbPct": f.bb_pct(walks, pa),
        },
    )


def pitching_from_payload(
    payload: dict[str, Any], *, season: int, fip_constant: float
) -> LeaguePitching:
    splits = _splits(payload)

    outs = _sum_field(splits, "outs")
    earned_runs = _sum_field(splits, "earnedRuns")
    hits = _sum_field(splits, "hits")
    walks = _sum_field(splits, "baseOnBalls")
    strikeouts = _sum_field(splits, "strikeOuts")
    home_runs = _sum_field(splits, "homeRuns")
    hbp = _sum_field(splits, "hitByPitch")
    batters_faced = _sum_field(splits, "battersFaced")

    return LeaguePitching(
        season=season,
        values={
            "era": f.era(earned_runs, outs),
            "whip": f.whip(walks, hits, outs),
            "kPer9": f.per_nine(strikeouts, outs),
            "bbPer9": f.per_nine(walks, outs),
            "hrPer9": f.per_nine(home_runs, outs),
            "kPct": f.k_pct(strikeouts, batters_faced),
            "bbPct": f.bb_pct(walks, batters_faced),
            "kMinusBbPct": f.k_minus_bb_pct(strikeouts, walks, batters_faced),
            # League FIP equals league ERA by construction -- that is what the
            # FIP constant is defined to make true. Included so the benchmark
            # row is complete rather than because it carries new information.
            "fip": f.fip(
                home_runs=home_runs,
                walks=walks,
                hit_by_pitch=hbp,
                strikeouts=strikeouts,
                outs=outs,
                fip_constant_=fip_constant,
            ),
        },
    )


# ---------------------------------------------------------------------------
# Benchmarks for Savant leaderboards
# ---------------------------------------------------------------------------


def leaderboard_means(
    rows: list[dict[str, str]],
    fields: tuple[str, ...],
    *,
    weight_field: str | None = None,
) -> dict[str, float | None]:
    """League benchmark for each column of a Savant leaderboard.

    Weighted by playing time where a sensible weight column exists, because an
    unweighted mean lets a player with twelve batted balls count as much as a
    regular with four hundred -- which describes neither the league nor any
    player in it.

    Rows missing a value are skipped for that column only, so one blank cell
    does not drop a player out of every benchmark.
    """
    totals: dict[str, float] = {}
    weights: dict[str, float] = {}

    for row in rows:
        weight = 1.0
        if weight_field:
            raw = (row.get(weight_field) or "").strip()
            try:
                weight = float(raw)
            except ValueError:
                continue
            if weight <= 0:
                continue

        for name in fields:
            raw = (row.get(name) or "").strip()
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            # Same percentage convention the player rows are read with, so a
            # value and its benchmark are always in the same units.
            value = sv.scale_rate(name, value)
            totals[name] = totals.get(name, 0.0) + value * weight
            weights[name] = weights.get(name, 0.0) + weight

    return {
        name: (totals[name] / weights[name] if weights.get(name) else None)
        for name in fields
    }


# ---------------------------------------------------------------------------
# Delta presentation
# ---------------------------------------------------------------------------

# For most stats a higher number is better. These are the ones where it is not:
# a pitcher wants a low ERA, a hitter wants a low strikeout rate.
LOWER_IS_BETTER = frozenset(
    {"era", "whip", "fip", "xfip", "siera", "bbPer9", "hrPer9", "kPct_batter"}
)


def delta(
    value: float | None, league: float | None, *, lower_is_better: bool = False
) -> dict[str, Any] | None:
    """Compare a value to the league average.

    Returns the signed difference plus a direction ('good' / 'bad' / 'even')
    that accounts for stats where lower is better, so the renderer never has to
    know which way a given metric points.
    """
    if value is None or league is None:
        return None

    difference = value - league
    if abs(difference) < 1e-9:
        direction = "even"
    else:
        better = difference < 0 if lower_is_better else difference > 0
        direction = "good" if better else "bad"

    return {"diff": difference, "direction": direction, "league": league}
