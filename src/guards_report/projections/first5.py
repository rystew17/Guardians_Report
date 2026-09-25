"""First five innings — the score, and the three-way result derived from it.

Modelled as runs per side rather than as a classifier, because the outcome is
three-way: 15.0% of games are level after five, where a full game has no ties at
all. Simulating two run distributions produces all three probabilities at once
and guarantees they agree with each other, which a separate win classifier and
score model would not.

**The starter is almost the whole game here.** Measured across the corpus he
faces 92.3% of the batters who come up in the first five innings, and in 72.4%
of games he faces every one of them. He goes 19.39 batters, which is 2.15 times
through a nine-man order -- so the third-time-through penalty that shapes a full
game barely applies, and a nine-inning line describes a different question than
the one being asked.

That is why this carries its own starter block. A season line includes the
innings where a pitcher tired and was pulled; the first-five line covers roughly
two turns through the order, and the two correlate only 0.40 to 0.49, so most of
what it says is not in the season figures.

The scoring share is stable enough to use as a fixed offset: the first five
innings hold 56.5% of a game's runs, and that holds from 0.560 in low-scoring
games to 0.576 in high-scoring ones. A shorter game is not a differently-shaped
game, just a smaller one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Runs scored in innings one through five as a share of the full game, measured
# at 5.100 of 8.99. Flat across scoring levels, which is what licenses using it
# as an offset rather than refitting the whole run environment.
FIRST5_SHARE = 5.100 / 8.99

# Overdispersion for five innings, chosen by held-out sweep with an interior
# minimum. Higher than the full game's 0.275 because one big inning is a larger
# share of a shorter game.
FIRST5_ALPHA = 0.45

# How often a side is blanked through five, as a law against its own
# projected runs.
#
# A negative binomial gets the mean and the spread of a five-inning score
# right -- 5.0079 against 4.9982 and sd 3.3273 against 3.3303 over 11,605
# held-out games -- and the shutout wrong. A side is held scoreless 20.38%
# of the time and the distribution says 18.06%, and the missing mass lands
# on one run and two. That is what leaned every posted line: the over ran
# 1.14 points light at 5 and 1.11 at 5.5.
#
# Four other explanations were measured first and none survived. The
# per-side game rate factor that fixed the full-game markets is 0.0555 here
# and exactly zero in three of seven seasons. The covariance between the two
# sides is -0.0063, so there is no shared park-and-weather factor to find.
# Drawing five negative-binomial half-innings instead of one negative
# binomial over five changes nothing. Putting the measured half-inning
# hurdle underneath that moves the bias 0.69 points to 0.66 and log loss the
# wrong way -- because conditioned on a game's rate, five independent
# half-innings are blanked 17.5% of the time. Compounding the pooled
# half-inning figure appears to work only because pooling smuggles in the
# variation between games; within a game the innings are not independent,
# and the same pitcher is the reason.
#
# So the shape is measured where the bet settles. Fitted on 28,846
# team-halves from 2015-2021, never on the 2022-26 window the market is
# graded against, across ten bands of projected runs:
#
#     logit P(blanked) = +0.1167 - 1.5999 * log(mu)
#
# holding to 1.09 points at worst. It takes the shutout rate from 0.1806 to
# 0.1987 against a realised 0.2038, and the mean |bias| across the five
# posted lines from 0.69 points to 0.66 with log loss very slightly better.
# A small gain, and it is the only one of five candidates that was a gain at
# all.
BLANK_INTERCEPT = 0.1167
BLANK_SLOPE = -1.5999

# Runs past which a five-inning score is not worth enumerating. The most
# either side has managed through five in the corpus is well inside this.
MAX_FIVE_INNING_RUNS = 16

# Starts before a pitcher's first-five history says anything. Below this the
# columns are left null rather than imputed, so the model learns from the
# `known` flag instead of from a number nobody measured.
MIN_PRIOR_STARTS = 5

STARTER_COLUMNS = ("opp_f5_ra", "opp_f5_bf")


def blank_chance(mu):
    """Measured chance a side is held scoreless through five."""
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-6)
    return 1.0 / (1.0 + np.exp(-(BLANK_INTERCEPT + BLANK_SLOPE * np.log(mu))))


def _nb_pmf(lam, alpha, kmax):
    from scipy.special import gammaln

    k = 1.0 / alpha
    lam = np.maximum(np.asarray(lam, dtype=float), 1e-9)[:, None]
    counts = np.arange(kmax + 1)[None, :]
    log = (gammaln(counts + k) - gammaln(counts + 1) - gammaln(k)
           + k * np.log(k / (k + lam)) + counts * np.log(lam / (k + lam)))
    return np.exp(log)


def score_pmf(mu, *, alpha: float = FIRST5_ALPHA,
              kmax: int = MAX_FIVE_INNING_RUNS):
    """Five-inning runs for one side: measured shutout, the rest negative binomial.

    The zero comes from `blank_chance`. The positive part is a zero-truncated
    negative binomial whose parameter is moved until the whole distribution has
    the mean it was handed, so correcting the shape never quietly changes the
    run environment -- the same mean-preserving construction the strikeout
    distribution uses.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    p0 = blank_chance(mu)
    target = mu / np.maximum(1.0 - p0, 1e-9)

    counts = np.arange(1, kmax + 1)
    low = np.full_like(mu, 1e-4)
    high = np.maximum(mu * 8.0, 1.0)
    for _ in range(60):
        mid = 0.5 * (low + high)
        pmf = _nb_pmf(mid, alpha, kmax)
        mean_positive = (pmf[:, 1:] @ counts) / np.maximum(1.0 - pmf[:, 0], 1e-12)
        below = mean_positive < target
        low = np.where(below, mid, low)
        high = np.where(below, high, mid)

    pmf = _nb_pmf(0.5 * (low + high), alpha, kmax)
    out = np.empty_like(pmf)
    out[:, 0] = p0
    out[:, 1:] = (pmf[:, 1:] / np.maximum(1.0 - pmf[:, 0:1], 1e-12)
                  * (1.0 - p0)[:, None])
    return out / out.sum(axis=1, keepdims=True)


def draw_scores(mu, *, draws: int, rng, alpha: float = FIRST5_ALPHA,
                chunk: int = 400):
    """Sample five-inning runs for each entry of `mu`.

    Shared by the calibration and the page on purpose. Drawn separately they
    drifted once already -- the measurement played ties out and the page did
    not -- and a record describing a model nobody serves is worse than no
    record.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    kmax = MAX_FIVE_INNING_RUNS
    out = np.empty((len(mu), draws), dtype=np.int64)
    for start in range(0, len(mu), chunk):
        block = mu[start:start + chunk]
        cdf = np.cumsum(score_pmf(block, alpha=alpha, kmax=kmax), axis=1)
        cdf[:, -1] = 1.0
        uniform = rng.random((len(block), draws))
        out[start:start + chunk] = (
            uniform[:, :, None] > cdf[:, None, :]).sum(axis=2)
    return np.clip(out, 0, kmax)


def build_dataset(pitch_corpus: pd.DataFrame) -> pd.DataFrame:
    """Runs through five innings for both clubs, one row per game."""
    early = pitch_corpus[pitch_corpus["inning"] <= 5]
    sides = early.groupby(["game_pk", "batting_team"], as_index=False).agg(
        f5_runs=("post_bat_score", "max")
    )
    meta = early[
        ["game_pk", "season", "game_date", "home_team", "away_team"]
    ].drop_duplicates("game_pk")

    frame = (
        meta.merge(
            sides.rename(columns={"batting_team": "home_team", "f5_runs": "home_f5"}),
            on=["game_pk", "home_team"], how="inner",
        ).merge(
            sides.rename(columns={"batting_team": "away_team", "f5_runs": "away_f5"}),
            on=["game_pk", "away_team"], how="inner",
        )
    )
    frame["f5_margin"] = frame["home_f5"] - frame["away_f5"]
    frame["f5_home_win"] = (frame["f5_margin"] > 0).astype(int)
    frame["f5_tie"] = (frame["f5_margin"] == 0).astype(int)
    return frame


def starter_history(pitch_corpus: pd.DataFrame) -> pd.DataFrame:
    """Each starter's first-five record, as of the start before this one.

    Runs allowed and batters faced inside five innings, averaged over his prior
    starts. The shift is the usual guarantee: row i carries starts 0..i-1, so
    the feature cannot contain the game it predicts.
    """
    plate = pitch_corpus[
        pitch_corpus["events"].notna() & (pitch_corpus["events"] != "")
    ]
    opener = (
        plate.sort_values(["game_pk", "batting_team", "inning"])
        .groupby(["game_pk", "batting_team"]).head(1)
        [["game_pk", "batting_team", "pitcher"]]
        .rename(columns={"pitcher": "starter"})
    )
    early = plate[plate["inning"] <= 5].merge(
        opener, on=["game_pk", "batting_team"], how="left"
    )
    own = early[early["pitcher"] == early["starter"]]

    per_start = own.groupby(
        ["starter", "game_pk", "game_date", "season"], as_index=False
    ).agg(
        bf=("events", "size"),
        end_score=("post_bat_score", "max"),
        start_score=("bat_score", "min"),
    )
    per_start["f5_runs"] = (
        per_start["end_score"] - per_start["start_score"]
    ).clip(lower=0)
    per_start = per_start.sort_values(["starter", "game_date", "game_pk"])

    grouped = per_start.groupby("starter", sort=False)
    prior_runs = grouped["f5_runs"].cumsum() - per_start["f5_runs"]
    prior_bf = grouped["bf"].cumsum() - per_start["bf"]
    prior_starts = grouped.cumcount()

    per_start["sp_f5_ra"] = prior_runs / prior_starts.replace(0, np.nan)
    per_start["sp_f5_bf"] = prior_bf / prior_starts.replace(0, np.nan)
    per_start["sp_f5_starts"] = prior_starts
    thin = prior_starts < MIN_PRIOR_STARTS
    per_start.loc[thin, ["sp_f5_ra", "sp_f5_bf"]] = np.nan

    # `game_date` travels with the row. It is what lets the live lookup take a
    # starter's line as of the day being projected rather than the last row in
    # the file, and it is how the refresh knows whether this table has fallen
    # behind the pitch corpus it was derived from -- neither of which is
    # answerable from the starter and game_pk alone.
    return per_start[
        ["starter", "game_pk", "game_date", "sp_f5_ra", "sp_f5_bf", "sp_f5_starts"]
    ]


def add_features(data: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Attach the opposing starter's first-five line to each team-game."""
    # The date is dropped rather than suffixed away: `data` carries its own
    # `game_date`, and a merge that produced `game_date_x`/`game_date_y` would
    # break every later reader of this frame for no gain -- the join key is the
    # game, so the two dates are the same date anyway.
    joined = data.merge(
        history.drop(columns=["game_date"], errors="ignore")
               .rename(columns={"starter": "opp_starter_id"}),
        on=["opp_starter_id", "game_pk"], how="left",
    ).rename(columns={"sp_f5_ra": "opp_f5_ra", "sp_f5_bf": "opp_f5_bf"})
    joined["f5_line_known"] = joined["opp_f5_ra"].notna().astype(int)
    return joined


def outcome_probabilities(
    home_mu: np.ndarray, away_mu: np.ndarray, *, alpha: float = FIRST5_ALPHA,
    draws: int = 20_000, seed: int = 11,
) -> dict[str, np.ndarray]:
    """Three-way result by simulation, plus the margin distribution.

    Ties are kept rather than resolved. A full game has none -- extra innings are
    played until somebody leads -- but a first-five result genuinely can be
    level, and collapsing that into one side or the other would misstate one
    outcome in six.
    """
    rng = np.random.default_rng(seed)
    # `draw_scores` rather than a bare negative binomial, so the three-way
    # result and the first-five total come out of one distribution. A side is
    # blanked through five more often than a negative binomial allows, and the
    # three-way result is mostly a question about exactly that.
    home = draw_scores(home_mu, draws=draws, rng=rng, alpha=alpha)
    away = draw_scores(away_mu, draws=draws, rng=rng, alpha=alpha)
    margin = home - away
    return {
        "home_leads": (margin > 0).mean(axis=1),
        "tied": (margin == 0).mean(axis=1),
        "away_leads": (margin < 0).mean(axis=1),
        "expected_home": home.mean(axis=1),
        "expected_away": away.mean(axis=1),
        "expected_total": (home + away).mean(axis=1),
    }
