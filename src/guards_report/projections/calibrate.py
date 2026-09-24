"""Checking each market's projection against what actually happened.

Only the game-outcome model had ever been measured this way, which is why the
betting page could stake a moneyline and nothing else: with no record of how
often a stated probability comes true there is no honest standard error, and a
stake is a claim about exactly that.

Every market is measured the same way -- walk forward season by season, fitting
only on what came before, predict the season that follows, then compare stated
probabilities against realized frequencies. Two rules keep it honest:

**The prior is refitted per season.** The shipped props artifact is fitted on
every season including the ones being tested, so using it directly would let the
model grade itself on data it had already seen. Refitting costs under a second.

**Rates are read strictly before first pitch.** The as-of merges use
`allow_exact_matches=False`, so a game can never contribute to the estimate that
predicts it. That one flag is the difference between a calibration figure and a
self-portrait.

Each market is measured at a single fixed line -- the number books most often
post -- which gives one observation per game and keeps them independent. Pricing
the same game at four lines would quadruple the apparent sample while the
underlying uncertainty stayed exactly where it was.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from guards_report.projections import backtest, model as model_module, props

# The lines each market is measured at.
#
# Measured per line rather than once, because the error is not the same at each
# of them. On strikeouts the model overstates by 1.8 points at 4.5 and
# understates by 3.5 at 6.5 and 8.5 -- the opposite direction and twice the
# size -- while the standard error doubles. A record taken at one line and
# applied to another therefore carries a bias pointing the wrong way, which is
# not extrapolation so much as the wrong answer.
#
# Each line's record stays independent: one observation per start per line.
# Correlation would only matter if they were pooled, and they are not.
STRIKEOUT_LINES = (3.5, 4.5, 5.5, 6.5, 7.5, 8.5)
TOTAL_LINES = (6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5)
FIRST5_TOTAL_LINES = (3.5, 4.0, 4.5, 5.0, 5.5)

# Signed from the home side, the way a book posts one: a home favorite is -1.5
# and has to win by two. Both signs are measured because they are not one bet
# seen twice. Covering -1.5 asks whether a good team wins comfortably; covering
# +1.5 asks whether a bad one stays close. Different questions, and nothing
# says the model is wrong by the same amount on each.
RUNLINE_LINES = (-2.5, -1.5, 1.5, 2.5)

# Books post hits and home runs at one number that matters, so these stay
# single. Two or more of either is a different bet, and is not priced.
HIT_LINE = 0.5
HOME_RUN_LINE = 0.5

# Kept as the default argument for the single-line callers.
STRIKEOUT_LINE = 4.5
TOTAL_LINE = 8.5

# Draws where a distribution is easier to sample than to solve. Monte Carlo
# error here is about 1.1 points, under the ~1.5 point resolution the
# calibration itself can report, so more would buy precision nothing can use.
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
    # A headline score beside the bins. The bins show WHERE a market is off;
    # these say whether it is worth anything at all, which nothing recorded
    # until now -- the props artifact shipped with an empty metrics block and
    # the page quoted a strikeout figure that was a literal in the template.
    log_loss: float | None = None
    baseline_log_loss: float | None = None
    brier: float | None = None
    mean_predicted: float | None = None
    actual_rate: float | None = None

    @property
    def measured(self) -> bool:
        return bool(self.bins)

    @property
    def gain(self) -> float | None:
        """How much better than knowing only the base rate."""
        if self.log_loss is None or self.baseline_log_loss is None:
            return None
        return self.baseline_log_loss - self.log_loss

    def as_dict(self) -> dict:
        return {
            "market": self.market, "line": self.line, "n": self.n,
            "bins": self.bins, "seasons": self.seasons, "note": self.note,
            "log_loss": self.log_loss,
            "baseline_log_loss": self.baseline_log_loss,
            "brier": self.brier, "mean_predicted": self.mean_predicted,
            "actual_rate": self.actual_rate,
        }


def _scored(market: str, line: float, realized, predicted,
            seasons: list[int]) -> "Calibrated":
    """One market's record: the bins, and a score against the base rate.

    The baseline is the held-out sample's own rate of the outcome -- the best a
    model can do knowing nothing about the game. A market that cannot beat it
    is not a projection, whatever its calibration looks like.
    """
    y = np.asarray(realized, dtype=float)
    p = np.clip(np.asarray(predicted, dtype=float), 1e-9, 1 - 1e-9)
    base = float(np.clip(y.mean(), 1e-9, 1 - 1e-9))
    return Calibrated(
        market=market, line=line, n=int(len(y)),
        bins=backtest.calibration_bins(y, p),
        seasons=seasons,
        log_loss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        baseline_log_loss=float(
            -(base * np.log(base) + (1 - base) * np.log(1 - base))),
        brier=float(np.mean((p - y) ** 2)),
        mean_predicted=float(p.mean()), actual_rate=float(y.mean()),
    )


def scoreboard(record: dict) -> list[dict]:
    """Every measured market, flattened, for a page to print.

    The stored record nests markets that were measured at several lines under
    `by_line`. This returns one row per market and line, newest measurement
    first in the order the markets are listed, with the headline score and the
    gain over knowing only the base rate.

    Markets measured before the score was added carry no figures and are left
    out rather than shown blank -- a market with nothing to report should not
    take a row on the page.
    """
    rows: list[dict] = []

    def add(entry: dict) -> None:
        if not isinstance(entry, dict) or entry.get("log_loss") is None:
            return
        rows.append({
            "market": entry.get("market"),
            "line": entry.get("line"),
            "n": entry.get("n"),
            "log_loss": entry.get("log_loss"),
            "baseline_log_loss": entry.get("baseline_log_loss"),
            "gain": (entry.get("baseline_log_loss") or 0.0)
            - (entry.get("log_loss") or 0.0),
            "mean_predicted": entry.get("mean_predicted"),
            "actual_rate": entry.get("actual_rate"),
            "seasons": entry.get("seasons") or [],
        })

    for entry in (record or {}).values():
        if not isinstance(entry, dict):
            continue
        if "by_line" in entry:
            for sub in (entry.get("by_line") or {}).values():
                add(sub)
        else:
            add(entry)
    return rows


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------

def _as_of(running: pd.DataFrame, side: str) -> pd.DataFrame:
    if running.empty:
        return pd.DataFrame(columns=["player", "game_date", "as_of_rate"])
    out = running.rename(columns={side: "player", "rate": "as_of_rate"})
    out = out[["player", "game_date", "as_of_rate"]].copy()
    # Normalized to nanoseconds. merge_asof refuses to join a datetime64[us]
    # against a datetime64[s], and the two frames come from parquet files
    # written by different code paths with different precisions.
    out["game_date"] = pd.to_datetime(out["game_date"]).astype("datetime64[ns]")
    out["player"] = out["player"].astype("int64")
    return out.sort_values("game_date")


def _merge_as_of(targets: pd.DataFrame, running: pd.DataFrame, *, key: str,
                 column: str) -> pd.DataFrame:
    """Attach each player's rate as it stood strictly before the game.

    `allow_exact_matches=False` is the whole point. Left on, a game contributes
    to the running estimate that predicts it and every figure downstream
    measures the model against itself.
    """
    targets = targets.copy()
    targets["game_date"] = pd.to_datetime(
        targets["game_date"]).astype("datetime64[ns]")
    targets = targets.sort_values("game_date")
    if running.empty:
        targets[column] = np.nan
        return targets
    targets["_player"] = targets[key].astype("int64")
    merged = pd.merge_asof(
        targets, running.rename(columns={"player": "_player"}),
        on="game_date", by="_player",
        direction="backward", allow_exact_matches=False,
    )
    return merged.rename(columns={"as_of_rate": column}).drop(columns=["_player"])


def _batter_games(season_plate: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """One row per batter per game: his slot, who started against him, and how
    many times the outcome happened."""
    ordered = season_plate.sort_values(["game_pk", "at_bat_number"])
    hit = props._flag(ordered, outcome)
    work = pd.DataFrame({
        "game_pk": ordered["game_pk"].to_numpy(),
        "game_date": pd.to_datetime(ordered["game_date"].to_numpy()),
        "batter": ordered["batter"].astype("int64").to_numpy(),
        "pitcher": ordered["pitcher"].astype("int64").to_numpy(),
        "batting_team": ordered["batting_team"].to_numpy(),
        "at_bat_number": ordered["at_bat_number"].to_numpy(),
        "stand": ordered["stand"].to_numpy(),
        "throws": ordered["p_throws"].to_numpy(),
        "hit": hit,
    })
    # The starter is whoever threw the side's first pitch of the game.
    starters = (work.sort_values("at_bat_number")
                .groupby(["game_pk", "batting_team"], as_index=False)
                .first()[["game_pk", "batting_team", "pitcher"]]
                .rename(columns={"pitcher": "starter"}))
    work = work.merge(starters, on=["game_pk", "batting_team"], how="left")

    # Batting order, as the order each hitter first came up.
    first = (work.groupby(["game_pk", "batting_team", "batter"], as_index=False)
             ["at_bat_number"].min())
    first["slot"] = (first.sort_values("at_bat_number")
                     .groupby(["game_pk", "batting_team"]).cumcount() + 1)

    totals = (work.groupby(["game_pk", "batting_team", "batter"], as_index=False)
              .agg(events=("hit", "sum"),
                   game_date=("game_date", "first"),
                   starter=("starter", "first"),
                   stand=("stand", "first"),
                   throws=("throws", "first")))
    return totals.merge(
        first[["game_pk", "batting_team", "batter", "slot"]],
        on=["game_pk", "batting_team", "batter"], how="left")


def _chance_weights(slots: np.ndarray) -> list[dict[int, float]]:
    return [
        props.SLOT_PA_DISTRIBUTION.get(int(s), props.UNKNOWN_SLOT_PA)
        if 1 <= int(s) <= 9 else props.UNKNOWN_SLOT_PA
        for s in slots
    ]


def _at_least_one(rates: np.ndarray, slots: np.ndarray) -> np.ndarray:
    """P(the outcome happens at least once), closed form.

    Summed over how many turns the slot actually gets rather than assuming a
    fixed four, because the number of chances is itself uncertain and is most
    of the spread on a 0.5 line.

        P(none) = sum_c P(c chances) * (1 - rate)^c
    """
    out = np.empty(len(rates), dtype=float)
    for index, (rate, weights) in enumerate(zip(rates, _chance_weights(slots))):
        rate = min(max(float(rate), 1e-9), 1 - 1e-9)
        none = sum(w * (1.0 - rate) ** c for c, w in weights.items())
        out[index] = 1.0 - none
    return out


# ---------------------------------------------------------------------------
# Batter counts: hits and home runs
# ---------------------------------------------------------------------------

def batter_counts(
    plate: pd.DataFrame,
    *,
    outcome: str,
    seasons: list[int],
    line: float = 0.5,
    platoon: dict[str, float] | None = None,
    calibration_factor: float = 1.0,
) -> Calibrated:
    """How often a stated hit or home run probability comes true."""
    predicted: list[np.ndarray] = []
    realized: list[np.ndarray] = []
    used: list[int] = []

    for season in seasons:
        before = plate[plate["season"] < season]
        current = plate[plate["season"] == season]
        if not len(before) or not len(current):
            continue

        prior = props.fit_rates(before, outcome)
        batters = _as_of(props.running_rates_decayed(
            prior, current, side="batter"), "batter")
        pitchers = _as_of(props.running_rates_decayed(
            prior, current, side="pitcher"), "pitcher")

        games = _batter_games(current, outcome)
        # The starting nine only. A substitute enters in the seventh and gets
        # one turn; the model assumes his slot's usual four, because on the
        # night it is projecting he is not in the lineup at all. Left in, those
        # rows are a tenth of the sample predicted at four times their real
        # chances, and they pulled every bin three to eight points low -- which
        # read as the model overstating hits and was this harness measuring
        # players it was never asked about.
        games = games[games["slot"].between(1, 9)]
        if games.empty:
            continue
        games = _merge_as_of(games, batters, key="batter", column="batter_rate")
        games = _merge_as_of(games, pitchers, key="starter", column="pitcher_rate")

        batter_rate = games["batter_rate"].fillna(prior.league).to_numpy(dtype=float)
        pitcher_rate = games["pitcher_rate"].fillna(prior.league).to_numpy(dtype=float)
        # The same adjustments the served projection applies. Measuring a
        # simpler model than the one that ships would produce a standard error
        # for a model nobody sees.
        stands = games["stand"].fillna("R").to_numpy()
        throws = games["throws"].fillna("R").to_numpy()
        rate = np.array([
            props.adjust(
                props.log5(b, p, prior.league),
                (platoon or {}).get(f"{str(s)[:1]}{str(th)[:1]}", 1.0),
                calibration_factor,
            )
            for b, p, s, th in zip(batter_rate, pitcher_rate, stands, throws)
        ])

        chance = _at_least_one(rate, games["slot"].fillna(9).to_numpy())
        predicted.append(chance)
        realized.append((games["events"].to_numpy(dtype=float) > line).astype(float))
        used.append(int(season))

    if not predicted:
        return Calibrated(market=outcome, line=line,
                          note="no held-out games to measure")
    return _scored(outcome, line, np.concatenate(realized),
                   np.concatenate(predicted), used)


# ---------------------------------------------------------------------------
# Starter strikeouts
# ---------------------------------------------------------------------------

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
        pitchers = _as_of(props.running_rates_decayed(
            prior, current, side="pitcher"), "pitcher")
        batters = _as_of(props.running_rates_decayed(
            prior, current, side="batter"), "batter")

        faced = _batter_games(current, "strikeout")
        if faced.empty:
            continue
        faced = _merge_as_of(faced, batters, key="batter", column="batter_rate")

        season_starts = starts[
            (starts["season"] == season) & (starts["gamesStarted"] == 1)
        ][["pitcher_id", "game_pk", "game_date", "strikeOuts"]].copy()
        season_starts["game_date"] = pd.to_datetime(season_starts["game_date"])
        season_starts = _merge_as_of(
            season_starts, pitchers, key="pitcher_id", column="pitcher_rate")

        # The nine he faced, in order, with their as-of rates.
        cards = (faced[faced["slot"] <= 9]
                 .sort_values(["game_pk", "starter", "slot"])
                 .groupby(["game_pk", "starter"], sort=False)
                 .agg(batters=("batter", list), rates=("batter_rate", list)))

        joined = season_starts.merge(
            cards, left_on=["game_pk", "pitcher_id"], right_index=True, how="inner")

        for row in joined.itertuples():
            rates = row.rates
            if not isinstance(rates, list) or len(rates) < 5:
                continue
            live = props.RateModel(
                outcome="strikeout", league=prior.league,
                stabilisation=prior.stabilisation,
                batter={
                    int(b): float(r) if pd.notna(r) else prior.batter_rate(int(b))
                    for b, r in zip(row.batters, rates)
                },
                pitcher={int(row.pitcher_id): (
                    float(row.pitcher_rate) if pd.notna(row.pitcher_rate)
                    else prior.pitcher_rate(int(row.pitcher_id)))},
                batter_pa=prior.batter_pa, pitcher_pa=prior.pitcher_pa,
            )
            projection = props.starter_strikeouts(
                live, pitcher_id=int(row.pitcher_id),
                lineup_ids=[int(b) for b in row.batters],
                expected_bf=starter_bf)
            predicted.append(float(projection.at_least(int(line) + 1)))
            realized.append(1.0 if float(row.strikeOuts) > line else 0.0)
        used.append(int(season))

    if not predicted:
        return Calibrated(market="strikeout", line=line,
                          note="no held-out starts to measure")
    return _scored("strikeout", line, np.array(realized),
                   np.array(predicted), used)


# ---------------------------------------------------------------------------
# First five
# ---------------------------------------------------------------------------

def first_five(
    frames: dict[int, dict[str, Any]], *, alpha: float,
) -> Calibrated:
    """How often a stated first-five probability comes true.

    Ties leave the denominator rather than being handed to a side, matching how
    the main line is settled: a tie after five is a push, not a loss.
    """
    predicted: list[float] = []
    realized: list[float] = []
    used: list[int] = []

    from guards_report.projections import first5 as f5_module

    for season, block in sorted(frames.items()):
        mu_home = np.asarray(block["mu_home"], dtype=float)
        mu_away = np.asarray(block["mu_away"], dtype=float)
        home_win = np.asarray(block["home_win"], dtype=float)
        tie = np.asarray(block["tie"], dtype=float)
        if not len(mu_home):
            continue

        outcome = f5_module.outcome_probabilities(mu_home, mu_away, alpha=alpha)
        p_home = np.asarray(outcome["home_leads"], dtype=float)
        p_away = np.asarray(outcome["away_leads"], dtype=float)
        live = p_home + p_away
        chance = np.where(live > 0, p_home / np.maximum(live, 1e-9), 0.5)

        played = tie < 0.5      # pushes are not a result either way
        predicted.extend(chance[played].tolist())
        realized.extend(home_win[played].tolist())
        used.append(int(season))

    if not predicted:
        return Calibrated(market="first_five", line=0.0,
                          note="no held-out games to measure")
    return _scored("first_five", 0.0, np.array(realized),
                   np.array(predicted), used)


# ---------------------------------------------------------------------------
# Game totals
# ---------------------------------------------------------------------------

def totals(
    frames: dict[int, dict[str, Any]], *, alpha: float,
    line: float = TOTAL_LINE, seed: int = 20260825,
    extra_innings: bool = True, market: str = "total",
) -> Calibrated:
    """How often a stated total-runs probability comes true.

    Sampled rather than solved: both sides are negative binomial and extra
    innings add runs to tied games, which is fiddly to convolve and trivial to
    draw. Monte Carlo error here sits below what the calibration can report.
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

        if extra_innings:
            # The same half-innings the report prices from, rules and all. This
            # used to draw two independent game totals and redraw ties, which
            # measured a model the page had stopped serving -- and the home
            # side's runs are censored by the score, so independent draws put
            # the total about two points high.
            home, away = model_module.simulate_counts(
                mu_home, mu_away, draws=DRAWS, rng=rng)
        else:
            # Five innings are always played out by both sides, so there is no
            # ninth-inning rule to apply and no tie to resolve: a first-five
            # total is a bet on exactly the number both clubs reached.
            def draw(mu: np.ndarray) -> np.ndarray:
                return rng.negative_binomial(
                    n, n / (n + mu[:, None]), size=(len(mu), DRAWS))

            home, away = draw(mu_home), draw(mu_away)

        # A total landing exactly on a whole number is a push: refunded, not
        # lost. The serving path has always priced it that way; this did not,
        # which is why 9 and 9.5 came back with identical figures to five
        # decimals -- the measurement was grading a bet nobody can place, and
        # the staking sigma read off it described the wrong wager.
        #
        # Half-point lines cannot push, so nothing below changes them.
        drawn = home + away
        over = (drawn > line).mean(axis=1)
        pushed = (drawn == line).mean(axis=1)
        live = np.clip(1.0 - pushed, 1e-9, None)
        played = actual != line
        predicted.extend((over / live)[played].tolist())
        realized.extend((actual[played] > line).astype(float).tolist())
        used.append(int(season))

    if not predicted:
        return Calibrated(market=market, line=line,
                          note="no held-out games to measure")
    return _scored(market, line, np.array(realized),
                   np.array(predicted), used)


def runline(
    frames: dict[int, dict[str, Any]], *, alpha: float,
    line: float = -1.5, seed: int = 20260826,
) -> Calibrated:
    """How often a stated run-line probability comes true.

    Measured on the margin rather than borrowed from the total. Both fall out
    of the same score model, but a different read of one model is not the same
    question, and that assumption is exactly what went wrong on strikeouts,
    where the error ran from -5.4 points at one line to +4.0 at another.

    `line` is signed from the home side, as posted: -1.5 means the home team
    must win by two, so the cover condition is `margin > -line`.
    """
    rng = np.random.default_rng(seed)
    n = 1.0 / max(alpha, 1e-9)
    predicted: list[float] = []
    realized: list[float] = []
    used: list[int] = []

    for season, block in sorted(frames.items()):
        mu_home = np.asarray(block["mu_home"], dtype=float)
        mu_away = np.asarray(block["mu_away"], dtype=float)
        margin = np.asarray(block["margin"], dtype=float)
        if not len(mu_home):
            continue

        # The margin comes out of the same simulation the report prices from.
        home, away = model_module.simulate_counts(
            mu_home, mu_away, draws=DRAWS, rng=rng)

        # A whole-number line pushes on an exact margin, and a push is neither
        # a win nor a loss. Books post halves so this is rare, but scoring one
        # as a loss would understate the side that pushed by all of it.
        live = margin != -line
        predicted.extend(((home - away) > -line).mean(axis=1)[live].tolist())
        realized.extend((margin[live] > -line).astype(float).tolist())
        used.append(int(season))

    if not predicted:
        return Calibrated(market="runline", line=line,
                          note="no held-out games to measure")
    return _scored("runline", line, np.array(realized),
                   np.array(predicted), used)
