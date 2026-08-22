"""Player propositions — will he get a hit, homer, or strike out this many.

Three of the four remaining projections are the same question asked at different
rarities: what is the chance a specific batter does something in one plate
appearance, and how many plate appearances will he get. Building them separately
would mean three copies of the same shrinkage and the same aggregation, so this
is one substrate with the outcome as a parameter.

**This is where the pitch corpus was always going to pay.** The game-outcome work
failed ten times because Elo already summarises everything measured at team level
from past results. There is no Elo for "does Jose Ramirez get a hit tonight" --
no rating already encodes it, so a per-plate-appearance estimate is not competing
with an incumbent that has seen the same evidence.

Two things carry the model:

**Rates are shrunk toward the league, by evidence.** Extremes live in the
smallest samples: a hitter at .400 over twenty at-bats is not a .400 hitter, and
ranking on the raw rate would fill the report with September call-ups. The
stabilisation point is measured per outcome rather than assumed, since strikeout
rate settles in a few dozen plate appearances and home-run rate takes most of a
season.

**Batter and pitcher combine by log5.** The odds form is the standard result and
the right one: a .300 hitter against a pitcher who allows a .200 average is not
.250, because the league baseline has to be divided back out. Averaging the two
rates would systematically flatten every extreme matchup, in a report whose whole
purpose is to find them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

# Plate-appearance outcomes, defined by the events Statcast records. Kept here so
# a projection and the report's own counting can never disagree about what a hit
# is.
HIT_EVENTS = frozenset({"single", "double", "triple", "home_run"})
HOMER_EVENTS = frozenset({"home_run"})
STRIKEOUT_EVENTS = frozenset({"strikeout", "strikeout_double_play"})

OUTCOMES = {
    "hit": HIT_EVENTS,
    "home_run": HOMER_EVENTS,
    "strikeout": STRIKEOUT_EVENTS,
}

# Plate appearances at which a rate is half league prior and half the player's
# own record. Measured by split-half reliability rather than chosen -- see
# `stabilisation_points`. These are the fallbacks.
# Measured on 2,051,513 plate appearances by interleaved split-half reliability.
# The ordering is the finding and it is the textbook one recovered from data:
# strikeout rate settles in a few dozen chances, home-run rate takes a couple of
# hundred, and hit rate is slowest of all because batting average carries
# balls-in-play luck that never becomes skill. A single constant across the three
# would badly overrate whichever outcome is rarest.
DEFAULT_STABILISATION = {
    "hit": 475,
    "home_run": 193,
    "strikeout": 55,
}

# Pitchers stabilise on a different schedule, most sharply for home runs -- a
# pitcher's home-run rate is mostly the hitters he faced and the park.
PITCHER_STABILISATION = {
    "hit": 455,
    "home_run": 727,
    "strikeout": 83,
}


@dataclass
class RateModel:
    """As-of rates for one outcome, for every batter and pitcher."""

    outcome: str
    league: float = 0.0
    stabilisation: int = 100
    batter: dict[int, float] = field(default_factory=dict)
    pitcher: dict[int, float] = field(default_factory=dict)
    batter_pa: dict[int, int] = field(default_factory=dict)
    pitcher_pa: dict[int, int] = field(default_factory=dict)
    through: str = ""

    def batter_rate(self, batter_id: int) -> float:
        return self.batter.get(int(batter_id), self.league)

    def pitcher_rate(self, pitcher_id: int) -> float:
        return self.pitcher.get(int(pitcher_id), self.league)

    def evidence(self, player_id: int, *, side: str) -> int:
        table = self.batter_pa if side == "batter" else self.pitcher_pa
        return int(table.get(int(player_id), 0))

    def matchup(self, batter_id: int, pitcher_id: int) -> float:
        return log5(
            self.batter_rate(batter_id), self.pitcher_rate(pitcher_id), self.league
        )


def log5(batter: float, pitcher: float, league: float) -> float:
    """Chance of the outcome when this batter faces this pitcher.

    The odds-ratio combination. Dividing the league rate back out is what stops
    a good hitter against a good pitcher landing at the average of the two --
    both are already measured against the same league, and averaging would count
    that baseline twice.
    """
    league = min(max(league, 1e-6), 1 - 1e-6)
    batter = min(max(batter, 1e-6), 1 - 1e-6)
    pitcher = min(max(pitcher, 1e-6), 1 - 1e-6)

    numerator = batter * pitcher / league
    denominator = numerator + (1 - batter) * (1 - pitcher) / (1 - league)
    return float(numerator / denominator) if denominator else league


def at_least_one(rate: float, chances: float) -> float:
    """Chance of at least one success in a fractional number of attempts.

    Plate appearances are not an integer before the game: a leadoff hitter gets
    four or five depending on how the innings fall. Using the expected count in
    the exponent is the continuous form of the same binomial, which is closer
    than rounding to four and pretending certainty.
    """
    rate = min(max(rate, 0.0), 1.0)
    return float(1.0 - (1.0 - rate) ** max(chances, 0.0))


def _flag(frame: pd.DataFrame, outcome: str) -> np.ndarray:
    return frame["events"].isin(OUTCOMES[outcome]).to_numpy(dtype=float)


def league_rate(frame: pd.DataFrame, outcome: str) -> float:
    return float(_flag(frame, outcome).mean())


def fit_rates(
    frame: pd.DataFrame, outcome: str, *, stabilisation: int | None = None
) -> RateModel:
    """Shrunk career rates for every batter and pitcher in the frame.

    The posterior mean under a beta prior centred on the league rate, which is
    the same empirical-Bayes shrinkage the talent model uses and reduces to
    (successes + k*league) / (chances + k).
    """
    k = stabilisation or DEFAULT_STABILISATION.get(outcome, 100)
    k_pitcher = PITCHER_STABILISATION.get(outcome, k)
    league = league_rate(frame, outcome)
    hit = _flag(frame, outcome)

    tables: dict[str, tuple[dict, dict]] = {}
    for side in ("batter", "pitcher"):
        side_k = k if side == "batter" else k_pitcher
        grouped = pd.DataFrame({"id": frame[side].to_numpy(), "hit": hit}).groupby("id")
        totals = grouped["hit"].agg(["sum", "size"])
        rates = (totals["sum"] + side_k * league) / (totals["size"] + side_k)
        tables[side] = (
            {int(i): float(v) for i, v in rates.items()},
            {int(i): int(v) for i, v in totals["size"].items()},
        )

    return RateModel(
        outcome=outcome,
        league=league,
        stabilisation=k,
        batter=tables["batter"][0],
        pitcher=tables["pitcher"][0],
        batter_pa=tables["batter"][1],
        pitcher_pa=tables["pitcher"][1],
        through=str(frame["game_date"].max()) if len(frame) else "",
    )


def running_rates(
    prior: RateModel, frame: pd.DataFrame, *, side: str = "batter"
) -> pd.DataFrame:
    """Each player's as-of rate after every plate appearance of a season.

    One pass. The shrunk rate is (successes + k*prior) / (chances + k), so a
    running sum and a running count carry everything, and the prior enters as
    pseudo-observations rather than being blended afterwards.
    """
    ordered = frame.sort_values(["game_date", "game_pk", "at_bat_number"])
    if ordered.empty:
        return pd.DataFrame(columns=[side, "game_date", "rate", "pa"])

    hit = _flag(ordered, prior.outcome)
    out = pd.DataFrame({
        side: ordered[side].to_numpy(),
        "game_date": ordered["game_date"].to_numpy(),
        "hit": hit,
    })
    grouped = out.groupby(side, sort=False)
    out["successes"] = grouped["hit"].cumsum()
    out["pa"] = grouped.cumcount() + 1

    base = out[side].map(
        prior.batter if side == "batter" else prior.pitcher
    ).fillna(prior.league)
    k = (
        prior.stabilisation if side == "batter"
        else PITCHER_STABILISATION.get(prior.outcome, prior.stabilisation)
    )
    out["rate"] = (out["successes"] + k * base) / (out["pa"] + k)

    return (
        out.groupby([side, "game_date"], as_index=False)
        .last()[[side, "game_date", "rate", "pa"]]
    )


def stabilisation_points(
    frame: pd.DataFrame, outcome: str, *, side: str = "batter", minimum: int = 300
) -> int:
    """Where split-half reliability reaches one half, measured not assumed.

    Splitting each player-season in two and correlating the halves gives the
    reliability of a half-season sample; Spearman-Brown converts that to the
    sample size at which the player's own record and the league prior deserve
    equal weight. Strikeout rate settles quickly, home-run rate barely settles
    at all, and using one constant for both would overrate whichever is rarer.
    """
    hit = _flag(frame, outcome)
    work = pd.DataFrame({
        "id": frame[side].to_numpy(),
        "season": frame["season"].to_numpy(),
        "hit": hit,
    })
    work["n"] = work.groupby(["id", "season"], sort=False).cumcount()
    totals = work.groupby(["id", "season"], sort=False)["n"].transform("max") + 1
    work = work[totals >= minimum]
    if work.empty:
        return DEFAULT_STABILISATION.get(outcome, 100)

    work["half"] = np.where(work["n"] % 2 == 0, 1, 2)   # interleaved, not first/second
    split = work.groupby(["id", "season", "half"])["hit"].agg(["sum", "size"])
    split["rate"] = split["sum"] / split["size"]
    wide = split["rate"].unstack("half").dropna()
    if len(wide) < 30:
        return DEFAULT_STABILISATION.get(outcome, 100)

    r = float(np.corrcoef(wide[1], wide[2])[0, 1])
    if not np.isfinite(r) or r <= 0:
        return DEFAULT_STABILISATION.get(outcome, 100)

    # Reliability of one half is r; the sample giving reliability 0.5 is the
    # half-sample size scaled by (1 - r) / r.
    half_n = float(split["size"].median())
    return int(max(10, round(half_n * (1 - r) / r)))


# How much weight one plate appearance retains after the next. Below 1.0 the
# running rate becomes an exponentially weighted average, so a hot month counts
# for more than the same month a year ago.
#
# This is the opposite of the team-level result and worth stating plainly. Recent
# form added nothing to the win model -- partial correlation +0.003 against a
# standard error of 0.006 -- because Elo is itself a recency-weighted rating of
# the same evidence. There is no such incumbent for a hitter: measured against
# the shrunk career rate, the last fifty plate appearances carry +0.012 for hits,
# +0.024 for home runs and +0.062 for strikeouts, the last at 45 standard errors.
DEFAULT_DECAY = {"hit": 0.999, "home_run": 0.999, "strikeout": 0.997}


def platoon_factors(frame: pd.DataFrame, outcome: str) -> dict[str, float]:
    """League outcome rate in each handedness matchup, relative to overall.

    Applied as a multiplier rather than fitted per player: a per-player platoon
    split needs several seasons to say anything, and the league effect is large
    and consistent -- home-run rate swings 28% across the four states, from
    0.0238 for a left-handed batter facing a left-hander to 0.0326 the other way.
    """
    league = league_rate(frame, outcome)
    if not league:
        return {}
    state = frame["stand"].fillna("R") + frame["p_throws"].fillna("R")
    rates = pd.Series(_flag(frame, outcome)).groupby(state.to_numpy()).mean()
    return {str(k): float(v / league) for k, v in rates.items()}


def park_factors(frame: pd.DataFrame, outcome: str, *, minimum: int = 5000) -> dict[str, float]:
    """Outcome rate by ballpark, relative to league.

    Keyed by the home club, which is the park regardless of who is batting.
    Matters most for home runs, where the spread is 1.71x from the least to the
    most generous yard -- larger than almost any individual batter effect.
    """
    league = league_rate(frame, outcome)
    if not league:
        return {}
    grouped = pd.Series(_flag(frame, outcome)).groupby(frame["home_team"].to_numpy())
    stats = grouped.agg(["mean", "size"])
    return {
        str(k): float(row["mean"] / league)
        for k, row in stats.iterrows()
        if row["size"] >= minimum
    }


def adjust(probability: float, *factors: float) -> float:
    """Apply multiplicative rate factors in odds space.

    Multiplying a probability directly can carry it past one, and it distorts
    rare events least where they matter most. The odds form is closed on (0, 1)
    and is the same operation log5 already performs.
    """
    probability = min(max(probability, 1e-9), 1 - 1e-9)
    odds = probability / (1 - probability)
    for factor in factors:
        if factor and np.isfinite(factor):
            odds *= factor
    return float(odds / (1 + odds))


def running_rates_decayed(
    prior: RateModel, frame: pd.DataFrame, *, side: str = "batter",
    decay: float | None = None,
) -> pd.DataFrame:
    """As-of rates with recent plate appearances weighted above older ones.

    Exponentially weighted: each prior chance is discounted by `decay` per
    subsequent plate appearance, so the estimate tracks form without the
    arbitrary cliff of a fixed window. At decay 1.0 this is exactly
    `running_rates`.

    The prior enters as pseudo-observations that do not decay, which is what
    keeps a player with twelve plate appearances near the league rate instead of
    at whatever he did last week.
    """
    lam = decay if decay is not None else DEFAULT_DECAY.get(prior.outcome, 1.0)
    ordered = frame.sort_values(["game_date", "game_pk", "at_bat_number"])
    if ordered.empty:
        return pd.DataFrame(columns=[side, "game_date", "rate", "pa"])
    if lam >= 1.0:
        # No decay is the flat cumulative rate, and asking pandas for an
        # exponential mean with alpha zero is an error rather than that limit.
        return running_rates(prior, frame, side=side)

    hit = _flag(ordered, prior.outcome)
    work = pd.DataFrame({
        side: ordered[side].to_numpy(),
        "game_date": ordered["game_date"].to_numpy(),
        "hit": hit,
    })

    # `adjust=True` normalises by the running sum of weights. With adjust=False
    # and an alpha this small the series is seeded on the player's first plate
    # appearance and needs hundreds of chances to escape it, which made every
    # early-season rate approximately "whatever happened his first time up".
    def weighted(series: pd.Series) -> pd.Series:
        return series.ewm(alpha=1 - lam, adjust=True).mean()

    grouped = work.groupby(side, sort=False)["hit"]
    work["ewm_rate"] = grouped.transform(weighted)
    work["pa"] = work.groupby(side, sort=False).cumcount() + 1

    # Effective sample size is the sum of the geometric weights actually
    # accumulated, which approaches 1/(1-lambda) but is far smaller early on --
    # exactly when the prior should still dominate.
    counts = work["pa"].to_numpy()
    effective = (1.0 - lam ** counts) / max(1.0 - lam, 1e-9)
    base = work[side].map(
        prior.batter if side == "batter" else prior.pitcher
    ).fillna(prior.league).to_numpy()
    k = (
        prior.stabilisation if side == "batter"
        else PITCHER_STABILISATION.get(prior.outcome, prior.stabilisation)
    )
    work["rate"] = (
        work["ewm_rate"].to_numpy() * effective + k * base
    ) / (effective + k)

    return (
        work.groupby([side, "game_date"], as_index=False)
        .last()[[side, "game_date", "rate", "pa"]]
    )


# Plate appearances a lineup slot actually gets, measured over 2022-2026. The
# distribution matters more than the mean: the leadoff spot is close to a coin
# flip between four and five turns, the ninth between three and four, and a
# projection that used 4.50 and 3.47 as certainties would understate the spread
# of every count it produces.
SLOT_PA_DISTRIBUTION = {
    1: {3: 0.04, 4: 0.44, 5: 0.46, 6: 0.06},
    2: {3: 0.05, 4: 0.52, 5: 0.40, 6: 0.03},
    3: {3: 0.06, 4: 0.58, 5: 0.33, 6: 0.03},
    4: {3: 0.08, 4: 0.63, 5: 0.27, 6: 0.02},
    5: {2: 0.02, 3: 0.12, 4: 0.65, 5: 0.20, 6: 0.01},
    6: {2: 0.03, 3: 0.16, 4: 0.64, 5: 0.16, 6: 0.01},
    7: {2: 0.05, 3: 0.23, 4: 0.59, 5: 0.12, 6: 0.01},
    8: {2: 0.08, 3: 0.31, 4: 0.52, 5: 0.09},
    9: {2: 0.10, 3: 0.38, 4: 0.45, 5: 0.06, 6: 0.01},
}

# When the card is not posted the slot is unknown; the league average across all
# nine is the honest stand-in, and the projection says the order was assumed.
UNKNOWN_SLOT_PA = {2: 0.04, 3: 0.19, 4: 0.56, 5: 0.20, 6: 0.02}


# A starter's strikeout rate falls sharply each time through the order --
# measured 0.2394, 0.2107, 0.1966 across the three passes, or 1.09x, 0.96x and
# 0.90x his overall rate. He faces roughly 2.4 passes, so applying one flat rate
# overstates the late plate appearances and inflates the total.
TIMES_THROUGH_FACTOR = {1: 1.0917, 2: 0.9607, 3: 0.8967, 4: 0.8967}

# Relievers strike out more than starters -- 0.2323 against 0.2193 -- so a league
# rate blended across both is the wrong baseline for a starting pitcher. Using
# it overstates a 22-batter start by about 0.12 strikeouts before any other
# adjustment.
STARTER_LEAGUE_FACTOR = 0.9758

# How far to pull each side's rate toward the league before combining them.
# log5 assumes the batter's and pitcher's rates are independent given the
# league; they are not quite, because an extreme record is partly extreme for
# having been built against soft opposition. Over twenty-two plate appearances
# that overshoot compounds, and the top quintile of starts was projected 0.72
# strikeouts high before this. Chosen by held-out log loss with an interior
# minimum, not by minimising bias -- bias alone would over-shrink and spoil the
# probabilities the report actually publishes.
MATCHUP_SHRINK = 0.75

# Held-out calibration on the game-level totals, fitted on 2023 and judged on
# 2024-2025. Home runs run about 5% high and a single factor fixes most of it.
#
# Hits are left alone deliberately. The sweep chose 1.00 because their bias is
# not a constant but a gradient -- +0.00 at the top of the order rising to +0.05
# at the ninth spot -- and no single multiplier addresses that. The cause is
# real rather than a defect: shrinking every rate toward the league mean must
# over-rate below-average hitters, and the bottom of the order is where they
# bat. A per-slot fudge would hide that instead of showing it.
CALIBRATION = {"hit": 1.00, "home_run": 0.98}


@dataclass
class CountProjection:
    """A full distribution over how many times something happens tonight."""

    expected: float = 0.0
    distribution: dict[int, float] = field(default_factory=dict)
    per_chance: float = 0.0
    expected_chances: float = 0.0
    evidence: int = 0
    note: str = ""

    def at_least(self, k: int) -> float:
        """Chance of k or more."""
        return float(sum(p for n, p in self.distribution.items() if n >= k))

    @property
    def none(self) -> float:
        return float(self.distribution.get(0, 0.0))


def _binomial(k: int, n: int, p: float) -> float:
    from math import comb

    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def count_distribution(
    rate: float, chances: dict[int, float], *, limit: int = 6
) -> CountProjection:
    """Distribution over successes, mixing binomials across possible chances.

    Plate appearances are not fixed before a game, so the count is a mixture:
    the chance of two hits is summed over the chance of getting four turns and
    the chance of getting five. Collapsing to a single expected number of turns
    would give the right mean and too narrow a spread, and the spread is the part
    a reader is being asked to judge.
    """
    rate = min(max(rate, 0.0), 1.0)
    total = sum(chances.values()) or 1.0
    out: dict[int, float] = {}

    for n, weight in chances.items():
        share = weight / total
        for k in range(0, min(int(n), limit) + 1):
            out[k] = out.get(k, 0.0) + share * _binomial(k, int(n), rate)

    # Prune first, then renormalise, then take the mean from what remains. The
    # published expected value has to be the mean of the published distribution:
    # computing it from the unpruned dict left the two disagreeing in the
    # seventh decimal, which is invisible and still wrong.
    kept = {k: v for k, v in sorted(out.items()) if v > 1e-6}
    mass = sum(kept.values()) or 1.0
    kept = {k: v / mass for k, v in kept.items()}

    expected = sum(k * p for k, p in kept.items())
    mean_chances = sum(int(n) * (w / total) for n, w in chances.items())
    return CountProjection(
        expected=float(expected),
        distribution={k: float(v) for k, v in kept.items()},
        per_chance=float(rate),
        expected_chances=float(mean_chances),
    )


def batter_projection(
    rates: RateModel,
    *,
    batter_id: int,
    pitcher_id: int,
    slot: int | None = None,
    platoon: dict[str, float] | None = None,
    park: dict[str, float] | None = None,
    stand: str = "R",
    throws: str = "R",
    home_team: str | None = None,
) -> CountProjection:
    """Expected total for one batter tonight, with the whole count distribution.

    The lineup slot is not decoration. A leadoff hitter gets 4.50 plate
    appearances against 3.47 for the ninth spot -- thirty percent more chances --
    so two hitters of identical quality have materially different totals
    depending on where the manager wrote them down.
    """
    rate = rates.matchup(batter_id, pitcher_id)
    factors = []
    if platoon:
        factors.append(platoon.get(f"{stand}{throws}", 1.0))
    if park and home_team:
        factors.append(park.get(home_team, 1.0))
    if factors:
        rate = adjust(rate, *factors)

    chances = SLOT_PA_DISTRIBUTION.get(int(slot), UNKNOWN_SLOT_PA) if slot else UNKNOWN_SLOT_PA
    projection = count_distribution(rate, chances)
    projection.evidence = rates.evidence(batter_id, side="batter")
    projection.note = (
        f"batting {slot}" if slot else "lineup not posted; league-average turns assumed"
    )
    return projection


def starter_strikeouts(
    rates: RateModel,
    *,
    pitcher_id: int,
    lineup_ids: list[int],
    expected_bf: float,
    platoon: dict[str, float] | None = None,
    stands: dict[int, str] | None = None,
    throws: str = "R",
    limit: int = 18,
) -> CountProjection:
    """Total strikeouts for the starting pitcher.

    Built batter by batter rather than from one blended rate, because a starter
    does not face an average lineup -- he faces the top of the order more often
    than the bottom, and a lineup with three high-strikeout hitters is worth more
    to him than its mean suggests.

    The count is a Poisson-binomial: each plate appearance has its own
    probability, so the distribution is the convolution of them all rather than a
    single binomial.
    """
    stands = stands or {}
    if not lineup_ids:
        rate = rates.pitcher_rate(pitcher_id) * STARTER_LEAGUE_FACTOR
        return count_distribution(rate, {int(round(expected_bf)): 1.0}, limit=limit)

    # Times through the order: whole passes, then a partial one at the top.
    passes, remainder = divmod(max(expected_bf, 0.0), len(lineup_ids))
    probabilities: list[float] = []
    def toward_league(value: float) -> float:
        return rates.league + MATCHUP_SHRINK * (value - rates.league)

    for index, batter in enumerate(lineup_ids):
        rate = log5(
            toward_league(rates.batter_rate(batter)),
            toward_league(rates.pitcher_rate(pitcher_id)),
            rates.league,
        )
        if platoon:
            rate = adjust(rate, platoon.get(f"{stands.get(batter, 'R')}{throws}", 1.0))
        # The league rate this was built from blends starters with relievers,
        # who strike out more; a starting pitcher sits below it.
        rate = adjust(rate, STARTER_LEAGUE_FACTOR)
        turns = int(passes) + (1 if index < int(round(remainder)) else 0)
        for turn in range(1, turns + 1):
            probabilities.append(
                adjust(rate, TIMES_THROUGH_FACTOR.get(turn, TIMES_THROUGH_FACTOR[3]))
            )

    # Convolve, which is exact and cheap at this size.
    distribution = np.zeros(len(probabilities) + 1)
    distribution[0] = 1.0
    for p in probabilities:
        distribution[1:] = distribution[1:] * (1 - p) + distribution[:-1] * p
        distribution[0] *= (1 - p)

    out = {k: float(v) for k, v in enumerate(distribution) if v > 1e-6 and k <= limit}
    expected = float(sum(k * v for k, v in out.items()))
    return CountProjection(
        expected=expected,
        distribution=out,
        per_chance=float(np.mean(probabilities)) if probabilities else 0.0,
        expected_chances=float(len(probabilities)),
        evidence=rates.evidence(pitcher_id, side="pitcher"),
        note=f"{len(probabilities)} batters faced assumed",
    )
