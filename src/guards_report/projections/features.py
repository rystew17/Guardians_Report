"""Feature assembly for the game-outcome models.

One rule governs this module: **a feature for a game on date D may use only data
from games played before D.** Every function here is written so that obeying it
is the path of least resistance, because leakage does not announce itself -- it
shows up as a backtest that looks unusually good, which is the one result nobody
is inclined to interrogate.

Three mechanisms enforce it:

* Elo is sequential, so its rating before a game is leak-free by construction.
* Starter statistics come from `pitchers.as_of_table`, which shifts each row to
  the totals *before* that start.
* Park factors are computed from **prior seasons only**, never the season in
  progress.

Signs are normalized so that a larger value always favours the home side. This
is not cosmetic: it makes a fitted coefficient's sign interpretable at a glance,
and a sign that comes out backwards is then an obvious signal that something is
wrong rather than a detail buried in the encoding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from guards_report.projections import elo

# Starter metrics carried into the model.
#
# Two runs-allowed measures were tested and both dropped, for different reasons.
# ERA contributed nothing next to FIP (p = 0.308) and its earned/unearned split
# is an official scorer's judgement rather than a property of the game. RA/9
# looked promising univariately (r = +0.081) but took a *negative* coefficient
# in the fitted model -- and that survived orthogonalising it against FIP, with
# VIF at 1.89 ruling out collinearity as the cause. What is left of RA/9 once
# FIP is accounted for is mostly luck, and luck reverts: a starter who has
# allowed more runs than his peripherals warrant is due to allow fewer.
#
# Dropping it improved log loss (0.67687 -> 0.67668) and left every coefficient
# interpretable. The defensive signal RA/9 was standing in for is measured
# properly at team level in the defense block instead.
STARTER_METRICS = ("fip", "k_pct", "bb_pct", "ip_per_start")

MIN_PRIOR_IP = 10.0


def park_factors(corpus_frame) -> pd.DataFrame:
    """Runs scored at each venue relative to league average, from prior seasons.

    Deliberately excludes the season being predicted. A park factor computed
    from the current season would carry the results of games not yet played --
    including, at the extreme, the game being predicted.

    Parks with too little history fall back to neutral (1.0) rather than to a
    noisy estimate from a handful of games.
    """
    per_game = corpus_frame.assign(
        total_runs=corpus_frame["home_runs"] + corpus_frame["away_runs"]
    )
    by_venue = (
        per_game.groupby(["season", "venue_id"])
        .agg(runs=("total_runs", "mean"), games=("total_runs", "size"))
        .reset_index()
    )
    league = per_game.groupby("season")["total_runs"].mean().rename("league_runs")
    by_venue = by_venue.join(league, on="season")
    by_venue["raw_factor"] = by_venue["runs"] / by_venue["league_runs"]

    # Shift so each season sees only what came before it, then expand backwards
    # through the seasons already played.
    by_venue = by_venue.sort_values(["venue_id", "season"])
    grouped = by_venue.groupby("venue_id")
    by_venue["prior_runs"] = (
        grouped["runs"].apply(lambda s: s.shift().expanding().mean()).to_numpy()
    )
    by_venue["prior_games"] = (
        grouped["games"].apply(lambda s: s.shift().expanding().sum()).to_numpy()
    )
    by_venue["prior_league"] = (
        grouped["league_runs"].apply(lambda s: s.shift().expanding().mean()).to_numpy()
    )

    factor = by_venue["prior_runs"] / by_venue["prior_league"]
    # Under ~150 games of history the estimate is mostly noise; neutral is the
    # honest answer.
    factor = factor.where(by_venue["prior_games"] >= 150, 1.0)
    by_venue["park_factor"] = factor.fillna(1.0)

    return by_venue[["season", "venue_id", "park_factor"]]


def team_rest(corpus_frame) -> pd.DataFrame:
    """Days since each team's previous game, per game.

    Returned long (one row per team-game) so the caller joins it once per side
    and cannot accidentally align the home value onto the away column.
    """
    long = pd.concat([
        corpus_frame[["game_pk", "game_date", "season", "home_team_id"]]
        .rename(columns={"home_team_id": "team_id"}),
        corpus_frame[["game_pk", "game_date", "season", "away_team_id"]]
        .rename(columns={"away_team_id": "team_id"}),
    ]).sort_values(["team_id", "season", "game_date"])

    previous = long.groupby(["team_id", "season"])["game_date"].shift()
    long["team_rest_days"] = (
        pd.to_datetime(long["game_date"]) - pd.to_datetime(previous)
    ).dt.days
    # First game of a season: capped rather than null. Spring rest is long but
    # not unboundedly so, and an unbounded value would dominate any scaling.
    long["team_rest_days"] = long["team_rest_days"].fillna(6).clip(upper=6)

    return long[["game_pk", "team_id", "team_rest_days"]]


def starter_rest(logs) -> pd.DataFrame:
    """Days since each pitcher's previous appearance."""
    ordered = logs.sort_values(["pitcher_id", "season", "game_date"]).copy()
    previous = ordered.groupby(["pitcher_id", "season"])["game_date"].shift()
    ordered["starter_rest_days"] = (
        pd.to_datetime(ordered["game_date"]) - pd.to_datetime(previous)
    ).dt.days
    # Five days is a standard turn; unknown rest gets the modal value rather
    # than a flag the model would over-read.
    ordered["starter_rest_days"] = ordered["starter_rest_days"].fillna(5).clip(1, 10)
    return ordered[["pitcher_id", "game_pk", "starter_rest_days"]]


def build_core(corpus_frame, logs, asof, *, elo_params: elo.EloParams) -> pd.DataFrame:
    """The Core feature block: Elo, starter quality, rest, park.

    Every column is expressed so that a higher value favours the home team.
    Rows where either starter lacks the minimum prior workload are kept, with
    their starter columns null -- dropping them would bias the sample toward
    established pitchers and quietly remove April.
    """
    frame = corpus_frame.reset_index(drop=True).copy()

    # --- Elo: sequential, so the value at row i used only games before i ------
    frame["elo_prob"] = elo.run(frame, elo_params)
    # Logit form gives the linear models something additive to work with, rather
    # than asking them to relearn the logistic transform Elo already applied.
    clipped = np.clip(frame["elo_prob"].to_numpy(), 1e-6, 1 - 1e-6)
    frame["elo_logit"] = np.log(clipped / (1 - clipped))

    # --- starter quality, home minus away ------------------------------------
    home = asof.add_prefix("h_").rename(
        columns={"h_pitcher_id": "home_starter_id", "h_game_pk": "game_pk"}
    )
    away = asof.add_prefix("a_").rename(
        columns={"a_pitcher_id": "away_starter_id", "a_game_pk": "game_pk"}
    )
    frame = frame.merge(home, on=["home_starter_id", "game_pk"], how="left")
    frame = frame.merge(away, on=["away_starter_id", "game_pk"], how="left")

    for metric in STARTER_METRICS:
        h, a = frame[f"h_{metric}"], frame[f"a_{metric}"]
        # For FIP and BB% a *lower* value is better, so away-minus-home points
        # toward the home side. K% and innings run the other way.
        frame[f"sp_{metric}"] = (a - h) if metric not in ("k_pct", "ip_per_start") else (h - a)

    thin = (frame["h_ip_prior"] < MIN_PRIOR_IP) | (frame["a_ip_prior"] < MIN_PRIOR_IP)
    for metric in STARTER_METRICS:
        frame.loc[thin, f"sp_{metric}"] = np.nan
    frame["starter_known"] = (~thin).astype(int)

    # --- rest ----------------------------------------------------------------
    rest = team_rest(corpus_frame)
    frame = frame.merge(
        rest.rename(columns={"team_id": "home_team_id", "team_rest_days": "h_rest"}),
        on=["game_pk", "home_team_id"], how="left",
    ).merge(
        rest.rename(columns={"team_id": "away_team_id", "team_rest_days": "a_rest"}),
        on=["game_pk", "away_team_id"], how="left",
    )
    frame["team_rest_diff"] = frame["h_rest"] - frame["a_rest"]

    sp_rest = starter_rest(logs)
    frame = frame.merge(
        sp_rest.rename(columns={"pitcher_id": "home_starter_id",
                                "starter_rest_days": "h_sp_rest"}),
        on=["game_pk", "home_starter_id"], how="left",
    ).merge(
        sp_rest.rename(columns={"pitcher_id": "away_starter_id",
                                "starter_rest_days": "a_sp_rest"}),
        on=["game_pk", "away_starter_id"], how="left",
    )
    frame["sp_rest_diff"] = frame["h_sp_rest"] - frame["a_sp_rest"]

    # --- park ----------------------------------------------------------------
    frame = frame.merge(park_factors(corpus_frame), on=["season", "venue_id"], how="left")
    frame["park_factor"] = frame["park_factor"].fillna(1.0)

    return frame


CORE_COLUMNS = (
    ["elo_logit"]
    + [f"sp_{m}" for m in STARTER_METRICS]
    + ["starter_known", "team_rest_diff", "sp_rest_diff", "park_factor"]
)


def design_matrix(frame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Model matrix and target, with missing starter values mean-imputed.

    Imputation happens here rather than in the feature builder so the null stays
    visible upstream, and `starter_known` rides alongside so the model can tell
    an imputed value from a measured one instead of treating them alike.
    """
    # `to_numpy` hands back a read-only view when the frame is a single
    # dtype block, so the imputation below raises rather than filling. It
    # has never fired in production because the feature frames carry mixed
    # dtypes and pandas copies those -- which means this worked by luck,
    # and would have broken the day a caller passed a uniformly float
    # frame. The copy is cheap and removes the coincidence.
    X = np.array(frame[columns].to_numpy(dtype=float), copy=True)
    means = np.nanmean(X, axis=0)
    indices = np.where(np.isnan(X))
    X[indices] = np.take(means, indices[1])
    return X, frame["home_win"].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Block 2 -- bullpen
# ---------------------------------------------------------------------------

BULLPEN_METRICS = ("bp_ra9", "bp_outs_l3", "bp_share")


def bullpen_table(corpus_frame, logs) -> pd.DataFrame:
    """Per team-game bullpen workload and effectiveness, as-of.

    Derived rather than fetched. The starter game logs cover only pitchers who
    started at least once, so pure relievers are absent from them -- but the
    bullpen's line is exactly the team's line minus the starter's, and both of
    those are already on hand:

        bullpen runs = (runs the team allowed) - (runs its starter allowed)
        bullpen outs = (team pitching outs)    - (starter outs)

    Runs are exact. Outs carry one approximation: a team's pitchers record three
    outs per inning the opposition bats, and the home side's last half-inning is
    cut short when it wins at home. That understates home bullpen outs slightly.
    The bias is small, applies only to home-team rows, and matters less than it
    might because the feature is used as a *difference* between two bullpens
    that are both measured the same way.
    """
    starters = (
        logs[logs["gamesStarted"] > 0][["pitcher_id", "game_pk", "outs", "runs"]]
        .rename(columns={"outs": "sp_outs", "runs": "sp_runs"})
    )

    sides = []
    for side, other in (("home", "away"), ("away", "home")):
        block = corpus_frame[[
            "game_pk", "game_date", "season", "innings",
            f"{side}_team_id", f"{side}_starter_id", f"{other}_runs",
        ]].rename(columns={
            f"{side}_team_id": "team_id",
            f"{side}_starter_id": "pitcher_id",
            f"{other}_runs": "team_runs_allowed",
        })
        block["is_home"] = int(side == "home")
        sides.append(block)

    long = pd.concat(sides, ignore_index=True)
    long = long.merge(starters, on=["pitcher_id", "game_pk"], how="left")

    long["team_outs"] = long["innings"].fillna(9) * 3.0
    long["bp_outs"] = (long["team_outs"] - long["sp_outs"]).clip(lower=0)
    long["bp_runs"] = (long["team_runs_allowed"] - long["sp_runs"]).clip(lower=0)

    long = long.sort_values(["team_id", "season", "game_date"]).reset_index(drop=True)
    grouped = long.groupby(["team_id", "season"], sort=False)

    # Shift so each row sees only games already played.
    prior_runs = grouped["bp_runs"].cumsum() - long["bp_runs"]
    prior_outs = grouped["bp_outs"].cumsum() - long["bp_outs"]
    prior_team = grouped["team_outs"].cumsum() - long["team_outs"]

    innings = (prior_outs / 3.0).replace(0, np.nan)
    long["bp_ra9"] = prior_runs * 9.0 / innings
    long["bp_share"] = prior_outs / prior_team.replace(0, np.nan)

    # Fatigue: bullpen outs recorded in the three days before this game. Uses a
    # date window rather than a game count because an off-day is exactly the
    # thing that makes a bullpen fresh again.
    dates = pd.to_datetime(long["game_date"])
    long["_d"] = dates
    fatigue = np.zeros(len(long))
    for _, block in long.groupby(["team_id", "season"], sort=False):
        d = block["_d"].to_numpy()
        outs = block["bp_outs"].to_numpy()
        idx = block.index.to_numpy()
        for i in range(len(block)):
            window = (d >= d[i] - np.timedelta64(3, "D")) & (d < d[i])
            fatigue[idx[i]] = outs[window].sum()
    long["bp_outs_l3"] = fatigue

    thin = prior_outs < 30      # under ten innings the rate is noise
    long.loc[thin, ["bp_ra9", "bp_share"]] = np.nan

    return long[["game_pk", "team_id", "bp_ra9", "bp_outs_l3", "bp_share"]]


def add_bullpen(frame, corpus_frame, logs) -> pd.DataFrame:
    """Attach bullpen differentials, signed so higher favours the home side."""
    table = bullpen_table(corpus_frame, logs)

    frame = frame.merge(
        table.rename(columns={"team_id": "home_team_id",
                              **{m: f"h_{m}" for m in BULLPEN_METRICS}}),
        on=["game_pk", "home_team_id"], how="left",
    ).merge(
        table.rename(columns={"team_id": "away_team_id",
                              **{m: f"a_{m}" for m in BULLPEN_METRICS}}),
        on=["game_pk", "away_team_id"], how="left",
    )

    # Lower runs allowed is better, and a *more tired* opposing bullpen helps
    # the home side -- so both are away-minus-home.
    frame["bp_ra9_diff"] = frame["a_bp_ra9"] - frame["h_bp_ra9"]
    frame["bp_fatigue_diff"] = frame["a_bp_outs_l3"] - frame["h_bp_outs_l3"]
    # Share is neither good nor bad on its own; it describes how heavily a club
    # leans on its pen, which interacts with fatigue.
    frame["bp_share_diff"] = frame["h_bp_share"] - frame["a_bp_share"]
    return frame


BULLPEN_COLUMNS = ["bp_ra9_diff", "bp_fatigue_diff", "bp_share_diff"]
