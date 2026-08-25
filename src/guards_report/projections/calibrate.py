"""Checking each market's projection against what actually happened.

Only the game-outcome model had ever been measured this way, which is why the
betting page can stake a moneyline and nothing else: without a record of how
often a stated probability comes true, there is no honest standard error, and a
stake is a claim about exactly that.

Every market here is measured the same way. Walk forward season by season,
fitting only on what came before, predict the season that follows, and compare
the stated probability against the realized frequency. Two rules make it honest:

**The prior is refitted per season.** The shipped props artifact is fitted on
every season including the ones being tested, so using it directly would let the
model grade itself on data it had already seen. Refitting costs under a second.

**Rates are read strictly before first pitch.** The as-of lookups use
`allow_exact_matches=False`, so a start can never contribute to the estimate
that predicts it. That single flag is the difference between a calibration
figure and a self-portrait.

Probabilities are measured at one fixed line per market rather than at the line
a book happened to post. A fixed line gives one observation per game, which
keeps them independent -- pricing the same game at four lines would inflate the
apparent sample fourfold while the underlying uncertainty stayed put.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from guards_report.projections import backtest, props

# The line each market is calibrated at. Chosen near the middle of the realized
# distribution so predicted probabilities spread across the range rather than
# bunching at one end.
STRIKEOUT_LINE = 4.5
TOTAL_LINE = 8.5

# Draws used where a distribution is easier to sample than to solve. Monte Carlo
# error at this count is about 1.1 points, comfortably under the ~1.5 point
# resolution floor the calibration itself can resolve, so a larger number would
# buy precision the measurement cannot use.
DRAWS = 2_000


@dataclass
class Calibrated:
    """One market's record against outcomes."""

    market: str
    line: float
    n: int = 0
    bins: list[dict] = field(default_factory=list)
    seasons: list[int] = field(default_factory=list)
    note: str = ""

    @property
    def measured(self) -> bool:
        return bool(self.bins)


def _lineups(season_plate: pd.DataFrame) -> pd.DataFrame:
    """The batters each starter actually faced, in order, first nine only.

    Reconstructed from the plate corpus rather than read from a lineup card,
    for the same reason the training code does it: a pinch-hitter appears as a
    consequence of how the game went and would not be on a card beforehand.
    """
    ordered = season_plate.sort_values(["game_pk", "at_bat_number"])
    faced = ordered.groupby(["game_pk", "pitcher"], sort=False)["batter"]
    rows = []
    for (game_pk, pitcher), batters in faced:
        seen: list[int] = []
        for batter in batters.tolist():
            if batter not in seen:
                seen.append(int(batter))
            if len(seen) == 9:
                break
        if len(seen) >= 5:
            rows.append({"game_pk": game_pk, "pitcher": pitcher,
                         "lineup": seen})
    return pd.DataFrame(rows)


def _as_of(running: pd.DataFrame, side: str) -> pd.DataFrame:
    """One row per player per date, sorted for an as-of merge."""
    if running.empty:
        return running
    return (running.sort_values("game_date")
            .rename(columns={side: "player", "rate": "as_of_rate"})
            [["player", "game_date", "as_of_rate"]])


def _merge_as_of(
    targets: pd.DataFrame, running: pd.DataFrame, *, key: str,
) -> pd.DataFrame:
    """Attach each player's rate as it stood strictly before the game.

    `allow_exact_matches=False` is the whole point. With it left on, a start
    contributes to the running estimate that predicts it, and every figure
    downstream measures the model against itself.
    """
    if running.empty or targets.empty:
        targets["as_of_rate"] = np.nan
        return targets

    left = targets.sort_values("game_date").copy()
    left["_player"] = left[key].astype("int64")
    right = running.copy()
    right["_player"] = right["player"].astype("int64")

    merged = pd.merge_asof(
        left, right[["_player", "game_date", "as_of_rate"]],
        on="game_date", by="_player",
        direction="backward", allow_exact_matches=False,
    )
    return merged.drop(columns=["_player"])


def strikeouts(
    plate: pd.DataFrame,
    starts: pd.DataFrame,
    *,
    seasons: list[int],
    starter_bf: float,
    line: float = STRIKEOUT_LINE,
) -> Calibrated:
    """How often a stated strikeout probability comes true.

    One observation per start: our chance of the starter clearing `line`,
    against whether he did.
    """
    predicted: list[float] = []
    realized: list[float] = []
    used: list[int] = []

    for season in seasons:
        before = plate[plate["season"] < season]
        current = plate[plate["season"] == season]
        if not len(before) or not len(current):
            continue

        prior = props.fit_rates(before, "strikeout")
        run_p = _as_of(props.running_rates_decayed(
            prior, current, side="pitcher"), "pitcher")
        run_b = _as_of(props.running_rates_decayed(
            prior, current, side="batter"), "batter")

        season_starts = starts[
            (starts["season"] == season) & (starts["gamesStarted"] == 1)
        ][["pitcher_id", "game_pk", "game_date", "strikeOuts"]].copy()
        if not len(season_starts):
            continue

        cards = _lineups(current)
        if cards.empty:
            continue
        season_starts = season_starts.merge(
            cards, left_on=["game_pk", "pitcher_id"],
            right_on=["game_pk", "pitcher"], how="inner")
        if not len(season_starts):
            continue

        season_starts["game_date"] = pd.to_datetime(season_starts["game_date"])
        run_p["game_date"] = pd.to_datetime(run_p["game_date"])
        run_b["game_date"] = pd.to_datetime(run_b["game_date"])

        season_starts = _merge_as_of(
            season_starts, run_p, key="pitcher_id").rename(
            columns={"as_of_rate": "pitcher_rate"})

        # Batter rates, one row per lineup slot, folded back into a per-start
        # list so the projection sees the same nine it would have on the night.
        slots = season_starts[["game_pk", "pitcher_id", "game_date", "lineup"]].explode(
            "lineup").rename(columns={"lineup": "batter"})
        slots["batter"] = slots["batter"].astype("int64")
        slots = _merge_as_of(slots, run_b, key="batter").rename(
            columns={"as_of_rate": "batter_rate"})
        by_start = slots.groupby(["game_pk", "pitcher_id"], sort=False).agg(
            batters=("batter", list), batter_rates=("batter_rate", list))
        season_starts = season_starts.merge(
            by_start, left_on=["game_pk", "pitcher_id"], right_index=True,
            how="left")

        for row in season_starts.itertuples():
            rates = getattr(row, "batter_rates", None)
            if rates is None or not isinstance(rates, list):
                continue
            live = props.RateModel(
                outcome="strikeout",
                league=prior.league,
                stabilisation=prior.stabilisation,
                batter={
                    int(b): float(r) if pd.notna(r) else prior.batter_rate(int(b))
                    for b, r in zip(row.batters, rates)
                },
                pitcher={
                    int(row.pitcher_id): (
                        float(row.pitcher_rate) if pd.notna(row.pitcher_rate)
                        else prior.pitcher_rate(int(row.pitcher_id))
                    )
                },
                batter_pa=prior.batter_pa,
                pitcher_pa=prior.pitcher_pa,
            )
            projection = props.starter_strikeouts(
                live,
                pitcher_id=int(row.pitcher_id),
                lineup_ids=[int(b) for b in row.batters],
                expected_bf=starter_bf,
            )
            chance = projection.at_least(int(line) + 1)
            predicted.append(float(chance))
            realized.append(1.0 if float(row.strikeOuts) > line else 0.0)

        used.append(int(season))

    if not predicted:
        return Calibrated(market="strikeouts", line=line,
                          note="no held-out starts to measure")

    return Calibrated(
        market="strikeouts",
        line=line,
        n=len(predicted),
        bins=backtest.calibration_bins(np.array(realized), np.array(predicted)),
        seasons=used,
    )


def totals(
    frames: dict[int, dict[str, Any]],
    *,
    alpha: float,
    line: float = TOTAL_LINE,
    seed: int = 20260824,
) -> Calibrated:
    """How often a stated total-runs probability comes true.

    `frames` maps a season to arrays of the two sides' expected runs and the
    realized total, produced by a walk-forward fit of the score model.

    Sampled rather than solved. Both sides are negative binomial and extra
    innings add runs to tied games, which is fiddly to convolve and trivial to
    draw -- and Monte Carlo error here sits an order below the resolution the
    calibration can report anyway.
    """
    rng = np.random.default_rng(seed)
    n = 1.0 / max(alpha, 1e-9)
    predicted: list[float] = []
    realized: list[float] = []
    used: list[int] = []

    for season, block in sorted(frames.items()):
        mu_home = np.asarray(block["mu_home"], dtype=float)
        mu_away = np.asarray(block["mu_away"], dtype=float)
        actual = np.asarray(block["total"], dtype=float)
        if not len(mu_home):
            continue

        def draw(mu: np.ndarray) -> np.ndarray:
            return rng.negative_binomial(
                n, n / (n + mu[:, None]), size=(len(mu), DRAWS))

        home, away = draw(mu_home), draw(mu_away)
        # Baseball has no ties: extra innings are a short additional frame,
        # matching how the report's own simulation resolves them.
        for _ in range(20):
            tied = home == away
            if not tied.any():
                break
            home = home + tied * draw(mu_home / 9.0)
            away = away + tied * draw(mu_away / 9.0)

        chance = ((home + away) > line).mean(axis=1)
        predicted.extend(chance.tolist())
        realized.extend((actual > line).astype(float).tolist())
        used.append(int(season))

    if not predicted:
        return Calibrated(market="total", line=line,
                          note="no held-out games to measure")

    return Calibrated(
        market="total",
        line=line,
        n=len(predicted),
        bins=backtest.calibration_bins(np.array(realized), np.array(predicted)),
        seasons=used,
    )
