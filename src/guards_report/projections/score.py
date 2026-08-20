"""Model B — expected runs per side.

Predicts the runs one team scores in one game, so every game contributes two
rows: the home side batting against the away staff, and the reverse. Modelling
it per side rather than as a margin is what lets the same fitted model produce a
full score distribution instead of a point estimate.

Independence between the two sides is not assumed on faith. Measured across
7,303 games, the correlation between home and away runs is **r = −0.0015** --
close enough to zero that a joint structure (bivariate Poisson, copula) would
add parameters to capture nothing.

Sequencing follows the agreed plan: fit Poisson first as the reference, test the
dispersion of *its* residuals, and only then decide whether the negative
binomial's extra parameter is earned. The marginal measurement already suggests
it will be (var/mean = 2.14, and conditioning on team identity did not reduce
it), but a summary statistic is not the same evidence as a fitted model's
residuals.

The run environment is entered as an **offset**, not a fitted season effect. A
season dummy absorbs each year cleanly in backtest and is then useless live: in
April there is no coefficient for the season yet. A trailing league average is
available at prediction time and leak-free by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# How much history the trailing league-average run level uses. Long enough to be
# stable, short enough to track a genuine shift in the run environment (measured
# range 8.50-9.66 R/G across seasons, F = 13.3, p = 1.2e-23).
LEAGUE_WINDOW_GAMES = 600


def league_run_level(corpus_frame) -> pd.Series:
    """Trailing league runs per team-game, as-of each game.

    Shifted before the rolling mean so a game never contributes to the
    environment used to predict it.
    """
    per_side = (corpus_frame["home_runs"] + corpus_frame["away_runs"]) / 2.0
    trailing = per_side.shift().rolling(LEAGUE_WINDOW_GAMES, min_periods=50).mean()
    # Before enough history accumulates, fall back to the running mean rather
    # than a constant guess.
    return trailing.fillna(per_side.shift().expanding().mean()).fillna(4.5)


def team_offense(corpus_frame) -> pd.DataFrame:
    """Season-to-date runs scored per game, per team, as-of."""
    sides = []
    for side, other in (("home", "away"), ("away", "home")):
        block = corpus_frame[[
            "game_pk", "game_date", "season", f"{side}_team_id", f"{side}_runs",
        ]].rename(columns={f"{side}_team_id": "team_id", f"{side}_runs": "runs"})
        sides.append(block)

    long = pd.concat(sides, ignore_index=True)
    long = long.sort_values(["team_id", "season", "game_date"]).reset_index(drop=True)
    grouped = long.groupby(["team_id", "season"], sort=False)

    prior_runs = grouped["runs"].cumsum() - long["runs"]
    prior_games = grouped.cumcount()

    long["off_rpg"] = prior_runs / prior_games.replace(0, np.nan)
    long["off_games"] = prior_games
    # Under 15 games the rate is mostly noise.
    long.loc[prior_games < 15, "off_rpg"] = np.nan
    return long[["game_pk", "team_id", "off_rpg", "off_games"]]


def build_dataset(
    corpus_frame, feature_frame, asof, pen_table=None,
    elo_ratings=None, od_expected=None,
) -> pd.DataFrame:
    """One row per team-game: what this side scored, and against what.

    `feature_frame` supplies the per-game columns already assembled for Model A
    (park factor above all); `asof` supplies starter quality; `pen_table` the
    optional bullpen availability. Everything is oriented from the *batting*
    team's point of view, so the opposing staff's columns describe what it faced.
    """
    corpus_frame = corpus_frame.reset_index(drop=True)
    league = league_run_level(corpus_frame)
    offense = team_offense(corpus_frame)

    park = feature_frame[["game_pk", "park_factor"]].drop_duplicates("game_pk")

    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        block = corpus_frame[[
            "game_pk", "game_date", "season", "venue_id",
            f"{side}_team_id", f"{other}_team_id",
            f"{side}_runs", f"{other}_starter_id",
        ]].rename(columns={
            f"{side}_team_id": "team_id",
            f"{other}_team_id": "opp_team_id",
            f"{side}_runs": "runs",
            f"{other}_starter_id": "opp_starter_id",
        })
        block["is_home"] = int(side == "home")
        block["league_rpg"] = league.to_numpy()

        # Team strength from the rating systems. Model A gets most of its signal
        # from Elo (+0.29 standardised); without it the run model was inferring
        # the same thing from runs-per-game alone, which is a far weaker proxy.
        if elo_ratings is not None:
            own, opp = elo_ratings if side == "home" else elo_ratings[::-1]
            block["elo_own"] = own
            block["elo_opp"] = opp
            block["elo_diff"] = own - opp
        if od_expected is not None:
            # off_def already predicts each side's runs directly, which is this
            # model's target -- so it enters as a prior mean rather than as a
            # generic strength score.
            key = "exp_home_runs" if side == "home" else "exp_away_runs"
            block["od_exp_runs"] = od_expected[key]

        rows.append(block)

    data = pd.concat(rows, ignore_index=True)

    # Batting side's own offensive form.
    data = data.merge(offense, on=["game_pk", "team_id"], how="left")
    # Opposing side's defensive form, reusing the same table from the other view.
    data = data.merge(
        offense.rename(columns={"team_id": "opp_team_id", "off_rpg": "opp_off_rpg"})
        .drop(columns=["off_games"]),
        on=["game_pk", "opp_team_id"], how="left",
    )

    # Opposing starter.
    starter = asof.rename(columns={"pitcher_id": "opp_starter_id"})
    keep = ["opp_starter_id", "game_pk", "fip", "k_pct", "bb_pct", "ip_per_start", "ip_prior"]
    data = data.merge(
        starter[keep].rename(columns={
            "fip": "opp_sp_fip", "k_pct": "opp_sp_k", "bb_pct": "opp_sp_bb",
            "ip_per_start": "opp_sp_ip", "ip_prior": "opp_sp_prior",
        }),
        on=["opp_starter_id", "game_pk"], how="left",
    )

    if pen_table is not None:
        data = data.merge(
            pen_table.rename(columns={
                "team_id": "opp_team_id",
                "pen_available_fip": "opp_pen_fip",
                "pen_top3_fip": "opp_pen_top3",
            })[["game_pk", "opp_team_id", "opp_pen_fip", "opp_pen_top3"]],
            on=["game_pk", "opp_team_id"], how="left",
        )

    data = data.merge(park, on="game_pk", how="left")
    data["park_factor"] = data["park_factor"].fillna(1.0)
    return data


# Predictors, all as-of and all oriented from the batting team's side.
SCORE_COLUMNS = [
    "off_rpg",        # this offence's form
    "opp_off_rpg",    # the opposing club's offence, a proxy for its overall quality
    "opp_sp_fip",     # who is on the mound
    "opp_sp_k",
    "opp_sp_bb",
    "opp_sp_ip",
    "park_factor",
    "is_home",
]

# Team strength, added after the first fit showed the mean model was the weak
# half: NB2 wrapped a well-calibrated distribution around a mean that had no
# access to the ratings Model A relies on.
STRENGTH_COLUMNS = ["elo_diff", "od_exp_runs"]

# The bullpen block failed in Model A because Elo had already priced it. Runs
# allowed late is where it should actually bite, with no Elo competing for the
# same variance.
PEN_COLUMNS = ["opp_pen_fip"]


def design(data: pd.DataFrame, columns: list[str]):
    """Model matrix, response and offset, with missing predictors mean-imputed.

    The offset carries the trailing league run level on the log scale, which is
    what makes the fitted coefficients relative to the run environment rather
    than absolute.
    """
    frame = data.dropna(subset=["runs"]).copy()
    X = frame[columns].to_numpy(dtype=float)
    means = np.nanmean(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(means, idx[1])

    y = frame["runs"].to_numpy(dtype=float)
    offset = np.log(np.clip(frame["league_rpg"].to_numpy(dtype=float), 0.5, None))
    return X, y, offset, frame


def dispersion(y: np.ndarray, mu: np.ndarray, n_params: int) -> float:
    """Pearson dispersion statistic: chi-square over residual degrees of freedom.

    Equals 1 when the Poisson variance assumption holds. Above 1 means the data
    vary more than Poisson allows, and the model's intervals are too narrow --
    which for a score projection means claiming more certainty about a blowout
    than the data support.
    """
    resid = (y - mu) ** 2 / np.clip(mu, 1e-9, None)
    return float(resid.sum() / (len(y) - n_params))
