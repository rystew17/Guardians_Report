"""Bullpen availability — who can actually pitch tonight, and how good are they.

The first attempt at a bullpen block used season-to-date bullpen runs allowed
and contributed nothing (−0.00006 log loss, p = 0.61). The diagnosis is that an
aggregate destroys the signal: a pen averaging 4.00 with its two leverage arms
rested and the same pen with both burned are the same number, and they are not
remotely the same proposition. If the good arms are unavailable, what is left is
the arm the manager has been avoiding using.

So this module models the pen as a roster of individuals with quality and
availability, and asks what is left tonight:

* **who is on it** — pitchers who have relieved for this club recently
* **how good each is** — as-of FIP and K%, built only from prior appearances
* **who can go** — recent workload, since three days in a row rules an arm out
* **what remains** — the quality of the arms that can actually be used

`pen_depth_gap` is the feature that encodes the specific failure mode: the
distance between the best available arm and the worst one likely to be needed.
A club with a rested closer and a rested long man has a small gap; a club down
to its last arm has a large one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# An arm is treated as unavailable when recent workload makes a manager
# unlikely to use it. These thresholds mirror the availability logic already
# used in the live report (metrics/team_context.py) rather than inventing a
# second standard.
BACK_TO_BACK_OUTS = 3        # pitched yesterday at all
HEAVY_3DAY_OUTS = 9          # three innings across three days
RECENT_WINDOW_DAYS = 3

# Below this workload a reliever's rate stats are noise, and he is treated as
# an unknown rather than assigned a misleading number.
MIN_PRIOR_OUTS = 30


def reliever_pool(logs, corpus_frame) -> pd.DataFrame:
    """Every relief appearance, tagged with the team the pitcher was on.

    Team identity is not in the pitcher logs, so it is recovered by joining the
    appearance to the game and reading off which side he was on.
    """
    relief = logs[logs["gamesStarted"] == 0].copy()

    sides = corpus_frame[["game_pk", "home_team_id", "away_team_id"]]
    relief = relief.merge(sides, on="game_pk", how="inner")
    relief["team_id"] = np.where(
        relief["is_home"], relief["home_team_id"], relief["away_team_id"]
    )
    return relief.drop(columns=["home_team_id", "away_team_id"])


def _as_of_quality(relief: pd.DataFrame) -> pd.DataFrame:
    """Each reliever's rate stats from his appearances *before* each outing."""
    relief = relief.sort_values(["pitcher_id", "season", "game_date"]).copy()
    grouped = relief.groupby(["pitcher_id", "season"], sort=False)

    counts = ["outs", "runs", "homeRuns", "baseOnBalls", "strikeOuts",
              "battersFaced", "hitByPitch"]
    prior = grouped[counts].cumsum() - relief[counts]

    innings = (prior["outs"] / 3.0).replace(0, np.nan)
    faced = prior["battersFaced"].replace(0, np.nan)

    relief["rp_fip"] = (
        13 * prior["homeRuns"]
        + 3 * (prior["baseOnBalls"] + prior["hitByPitch"])
        - 2 * prior["strikeOuts"]
    ) / innings
    relief["rp_k_pct"] = prior["strikeOuts"] / faced
    relief["rp_outs_prior"] = prior["outs"]

    thin = prior["outs"] < MIN_PRIOR_OUTS
    relief.loc[thin, ["rp_fip", "rp_k_pct"]] = np.nan

    # Put FIP on the ERA scale so magnitudes are comparable and the numbers mean
    # something to a reader.
    #
    # The shift comes from seasons already finished, never from the season being
    # described. The first version averaged the whole season, so a game in April
    # carried a constant computed from September -- a season-wide peek at the
    # very season a walk-forward fold is holding out. That was harmless while
    # nothing fitted these columns, and a leak the moment something did.
    #
    # The earliest season has nothing before it and is left unshifted rather
    # than shifted by itself, which would reintroduce exactly the same peek for
    # that season.
    for season in sorted(relief["season"].unique()):
        earlier = relief.loc[relief["season"] < season, "rp_fip"]
        if not len(earlier):
            continue
        reference = earlier.mean(skipna=True)
        if not np.isfinite(reference):
            continue
        mask = relief["season"] == season
        relief.loc[mask, "rp_fip"] = relief.loc[mask, "rp_fip"] + (3.10 - reference)

    return relief


def availability_table(logs, corpus_frame) -> pd.DataFrame:
    """Per team-game: how many arms are available, and how good they are.

    For each game, the club's pen is the set of pitchers who have relieved for
    it in the previous 30 days. Each is marked available or not from his
    workload in the three days before, and the quality figures are computed over
    the available subset only.
    """
    relief = _as_of_quality(reliever_pool(logs, corpus_frame))
    relief["date"] = pd.to_datetime(relief["game_date"])

    # Team-game slots we need a row for.
    slots = pd.concat([
        corpus_frame[["game_pk", "game_date", "season", "home_team_id"]]
        .rename(columns={"home_team_id": "team_id"}),
        corpus_frame[["game_pk", "game_date", "season", "away_team_id"]]
        .rename(columns={"away_team_id": "team_id"}),
    ], ignore_index=True)
    slots["date"] = pd.to_datetime(slots["game_date"])

    out_rows = []
    by_team = {key: block for key, block in relief.groupby(["team_id", "season"], sort=False)}

    for (team_id, season), block in slots.groupby(["team_id", "season"], sort=False):
        pool = by_team.get((team_id, season))
        if pool is None:
            continue

        pool_dates = pool["date"].to_numpy()
        pids = pool["pitcher_id"].to_numpy()
        outs = pool["outs"].to_numpy()
        fips = pool["rp_fip"].to_numpy()

        for game_pk, when in zip(block["game_pk"].to_numpy(), block["date"].to_numpy()):
            # The pen as it stood: anyone who relieved in the last 30 days.
            member = (pool_dates < when) & (pool_dates >= when - np.timedelta64(30, "D"))
            if not member.any():
                continue

            recent = pool_dates >= when - np.timedelta64(RECENT_WINDOW_DAYS, "D")
            yesterday = pool_dates >= when - np.timedelta64(1, "D")

            quality: dict[int, float] = {}
            burden: dict[int, tuple[float, float]] = {}
            for pid, o, fip, d in zip(
                pids[member], outs[member], fips[member], pool_dates[member]
            ):
                if not np.isnan(fip):
                    quality[pid] = fip
                three, one = burden.get(pid, (0.0, 0.0))
                if d >= when - np.timedelta64(RECENT_WINDOW_DAYS, "D"):
                    three += o
                if d >= when - np.timedelta64(1, "D"):
                    one += o
                burden[pid] = (three, one)

            available = [
                q for pid, q in quality.items()
                if not (
                    burden.get(pid, (0, 0))[1] >= BACK_TO_BACK_OUTS
                    and burden.get(pid, (0, 0))[0] >= HEAVY_3DAY_OUTS
                )
            ]
            if len(available) < 2:
                available = list(quality.values())

            if not available:
                continue

            arms = np.sort(np.array(available))
            out_rows.append({
                "game_pk": game_pk,
                "team_id": team_id,
                "pen_n_available": len(arms),
                "pen_available_fip": float(arms.mean()),
                # The arm you want in a tight spot.
                "pen_best_fip": float(arms[0]),
                # The one you are stuck with if the game goes long -- the
                # scenario an average cannot see.
                "pen_worst_fip": float(arms[-1]),
                "pen_depth_gap": float(arms[-1] - arms[0]),
                "pen_top3_fip": float(arms[: min(3, len(arms))].mean()),
            })

    return pd.DataFrame(out_rows)


PEN_METRICS = (
    "pen_n_available", "pen_available_fip", "pen_best_fip",
    "pen_worst_fip", "pen_depth_gap", "pen_top3_fip",
)


def add_features(frame, logs, corpus_frame) -> pd.DataFrame:
    """Attach availability differentials, signed so higher favours the home side."""
    table = availability_table(logs, corpus_frame)

    # No relief history anywhere means the table comes back with no columns at
    # all, and merging on `game_pk` against it raises. That is opening day --
    # every club has a bullpen and none of them has used it yet. The honest
    # answer is that the feature is unknown, which is what the design matrix
    # already handles by imputing; raising would lose the whole game instead.
    if table.empty or "game_pk" not in table.columns:
        for name in PEN_COLUMNS:
            frame[name] = np.nan
        return frame

    frame = frame.merge(
        table.rename(columns={"team_id": "home_team_id",
                              **{m: f"h_{m}" for m in PEN_METRICS}}),
        on=["game_pk", "home_team_id"], how="left",
    ).merge(
        table.rename(columns={"team_id": "away_team_id",
                              **{m: f"a_{m}" for m in PEN_METRICS}}),
        on=["game_pk", "away_team_id"], how="left",
    )

    # Lower FIP is better, so away-minus-home favours the home side.
    frame["pen_avail_fip_diff"] = frame["a_pen_available_fip"] - frame["h_pen_available_fip"]
    frame["pen_best_fip_diff"] = frame["a_pen_best_fip"] - frame["h_pen_best_fip"]
    frame["pen_top3_fip_diff"] = frame["a_pen_top3_fip"] - frame["h_pen_top3_fip"]
    # A club with more arms left has the advantage.
    frame["pen_n_diff"] = frame["h_pen_n_available"] - frame["a_pen_n_available"]
    # A *smaller* drop-off is better, so away-minus-home again.
    frame["pen_depth_gap_diff"] = frame["a_pen_depth_gap"] - frame["h_pen_depth_gap"]
    return frame


PEN_COLUMNS = [
    "pen_avail_fip_diff", "pen_best_fip_diff", "pen_top3_fip_diff",
    "pen_n_diff", "pen_depth_gap_diff",
]
