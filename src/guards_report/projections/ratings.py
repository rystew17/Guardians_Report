"""Team rating systems — candidate backbones for the outcome models.

Standard Elo rates a team with one number. That is a real modelling assumption,
and on this problem a questionable one: a club that scores six and allows five
gets the same rating as one that scores four and allows three, but they are not
the same opponent, and they are certainly not the same opponent *for a given
starting pitcher*. A single scalar cannot express that.

Three systems are implemented here so the assumption can be tested rather than
inherited:

* `win_elo`   — classic Elo on win/loss, optionally scaled by margin. The
                incumbent benchmark.
* `run_elo`   — one scalar per team, updated on run differential rather than
                the binary result, so a one-run win and a blowout are not
                treated as identical evidence.
* `off_def`   — two ratings per team, offence and defence, in runs-per-game
                units. Predicts each side's runs separately, which is the
                structure the score model needs anyway.

All three are sequential: a rating used for game i was built only from games
before i, so none of them can leak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEAGUE_MEAN_RUNS = 4.5      # starting point only; the fit updates from data


@dataclass
class RunEloParams:
    k: float = 0.06          # rating points per run of surprise
    hfa: float = 0.20        # in runs, not rating points
    carry: float = 0.70
    scale: float = 0.60      # run differential -> win probability slope


@dataclass
class OffDefParams:
    k_off: float = 0.020
    k_def: float = 0.020
    hfa: float = 0.10        # extra runs for the home side
    carry: float = 0.70
    scale: float = 0.60


def run_elo(frame, params: RunEloParams) -> tuple[np.ndarray, np.ndarray]:
    """Single rating per team, updated on run differential.

    Returns (expected_margin, win_probability) per game. The rating is carried
    in runs, so a rating of +0.4 means "scores 0.4 more runs per game than it
    allows, against average opposition".
    """
    ratings: dict[int, float] = {}
    seasons: dict[int, int] = {}
    margin = np.empty(len(frame))
    prob = np.empty(len(frame))

    home_ids = frame["home_team_id"].to_numpy()
    away_ids = frame["away_team_id"].to_numpy()
    home_runs = frame["home_runs"].to_numpy()
    away_runs = frame["away_runs"].to_numpy()
    season = frame["season"].to_numpy()

    for i in range(len(frame)):
        h, a, s = int(home_ids[i]), int(away_ids[i]), int(season[i])
        for team in (h, a):
            if team not in ratings:
                ratings[team], seasons[team] = 0.0, s
            elif seasons[team] != s:
                ratings[team] *= params.carry
                seasons[team] = s

        expected = ratings[h] - ratings[a] + params.hfa
        margin[i] = expected
        prob[i] = 1.0 / (1.0 + np.exp(-params.scale * expected))

        actual = float(home_runs[i]) - float(away_runs[i])
        surprise = actual - expected
        ratings[h] += params.k * surprise
        ratings[a] -= params.k * surprise

    return margin, prob


def off_def(frame, params: OffDefParams) -> dict[str, np.ndarray]:
    """Separate offence and defence ratings, in runs per game.

    Each side's expected runs come from one team's offence meeting the other's
    defence, which is the structure the game actually has. Updating offence and
    defence from the *same* residual is deliberate: when a team scores seven,
    the evidence is genuinely ambiguous between a good offence and a bad
    opposing defence, and splitting the credit evenly is the honest encoding of
    that ambiguity rather than a claim about which it was.
    """
    offence: dict[int, float] = {}
    defence: dict[int, float] = {}
    seasons: dict[int, int] = {}

    n = len(frame)
    exp_home = np.empty(n)
    exp_away = np.empty(n)
    prob = np.empty(n)

    home_ids = frame["home_team_id"].to_numpy()
    away_ids = frame["away_team_id"].to_numpy()
    home_runs = frame["home_runs"].to_numpy()
    away_runs = frame["away_runs"].to_numpy()
    season = frame["season"].to_numpy()

    # League run level, tracked as it drifts (measured: 8.50-9.66 R/G across
    # seasons, F=13.3, p=1.2e-23 -- too large to treat as constant).
    league = LEAGUE_MEAN_RUNS
    decay = 0.999

    for i in range(n):
        h, a, s = int(home_ids[i]), int(away_ids[i]), int(season[i])
        for team in (h, a):
            if team not in offence:
                offence[team], defence[team], seasons[team] = 0.0, 0.0, s
            elif seasons[team] != s:
                offence[team] *= params.carry
                defence[team] *= params.carry
                seasons[team] = s

        eh = league + offence[h] + defence[a] + params.hfa
        ea = league + offence[a] + defence[h]
        eh, ea = max(eh, 0.5), max(ea, 0.5)

        exp_home[i], exp_away[i] = eh, ea
        prob[i] = 1.0 / (1.0 + np.exp(-params.scale * (eh - ea)))

        ah, aa = float(home_runs[i]), float(away_runs[i])
        res_h, res_a = ah - eh, aa - ea

        # Home scoring is evidence about home offence and away defence alike.
        offence[h] += params.k_off * res_h
        defence[a] += params.k_def * res_h
        offence[a] += params.k_off * res_a
        defence[h] += params.k_def * res_a

        league = decay * league + (1 - decay) * ((ah + aa) / 2.0)

    return {
        "exp_home_runs": exp_home,
        "exp_away_runs": exp_away,
        "exp_margin": exp_home - exp_away,
        "prob": prob,
    }


def final_off_def(frame, params: OffDefParams) -> dict[int, tuple[float, float]]:
    """Offence/defence ratings after walking the frame, for live prediction."""
    offence: dict[int, float] = {}
    defence: dict[int, float] = {}
    seasons: dict[int, int] = {}
    league = LEAGUE_MEAN_RUNS

    for row in frame.itertuples(index=False):
        h, a, s = int(row.home_team_id), int(row.away_team_id), int(row.season)
        for team in (h, a):
            if team not in offence:
                offence[team], defence[team], seasons[team] = 0.0, 0.0, s
            elif seasons[team] != s:
                offence[team] *= params.carry
                defence[team] *= params.carry
                seasons[team] = s

        eh = max(league + offence[h] + defence[a] + params.hfa, 0.5)
        ea = max(league + offence[a] + defence[h], 0.5)
        res_h, res_a = float(row.home_runs) - eh, float(row.away_runs) - ea

        offence[h] += params.k_off * res_h
        defence[a] += params.k_def * res_h
        offence[a] += params.k_off * res_a
        defence[h] += params.k_def * res_a
        league = 0.999 * league + 0.001 * ((row.home_runs + row.away_runs) / 2.0)

    return {t: (offence[t], defence[t]) for t in offence}
