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

# Starts before a pitcher's first-five history says anything. Below this the
# columns are left null rather than imputed, so the model learns from the
# `known` flag instead of from a number nobody measured.
MIN_PRIOR_STARTS = 5

STARTER_COLUMNS = ("opp_f5_ra", "opp_f5_bf")


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
    n = 1.0 / alpha
    home = rng.negative_binomial(n, n / (n + np.asarray(home_mu)[:, None]), size=(len(home_mu), draws))
    away = rng.negative_binomial(n, n / (n + np.asarray(away_mu)[:, None]), size=(len(away_mu), draws))
    margin = home - away
    return {
        "home_leads": (margin > 0).mean(axis=1),
        "tied": (margin == 0).mean(axis=1),
        "away_leads": (margin < 0).mean(axis=1),
        "expected_home": home.mean(axis=1),
        "expected_away": away.mean(axis=1),
        "expected_total": (home + away).mean(axis=1),
    }
