"""Elo ratings — the benchmark every win-probability model has to beat.

Elo is the right benchmark rather than a strawman. It is sequential, so it is
leak-free by construction: a game's prediction uses only ratings built from
games already played. It needs no feature engineering, carries four parameters,
and in published work on baseball it is difficult to beat by much. If a fitted
model cannot clear it, that is worth knowing before any effort goes into tuning.

Every parameter here is a choice, so none of them is hard-coded to folklore --
they are fitted on training seasons and the chosen values reported.

Formulation
-----------
    expected_home = 1 / (1 + 10 ** (-(R_home + hfa - R_away) / 400))
    R_home += k * mov_multiplier * (won - expected_home)

`mov_multiplier` optionally scales the update by margin of victory, damped so a
blowout does not count linearly (a 12-run win is not four times the evidence of
a 3-run win). Between seasons ratings regress toward the league mean by
`1 - carry`, which encodes roster turnover.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

MEAN_RATING = 1500.0


@dataclass
class EloParams:
    """The four things that define an Elo system.

    Defaults are starting points for the search, not recommendations -- they are
    replaced by fitted values before anything is reported.
    """

    k: float = 4.0
    hfa: float = 24.0          # in rating points, not probability
    # Season-to-season carryover. A roster-aware replacement for it was built
    # and tested, and could not be shown to help.
    #
    # The motivation was an outside benchmark: FiveThirtyEight's model beat
    # ours by 1.49 points while its team-only Elo merely matched ours, and the
    # gap was widest in April and May (+2.27) -- the shape of a better
    # preseason prior, which theirs had and this has never had.
    #
    # So one was built from the 40-man roster the day before opening day (MLB's
    # API), each player valued by the ridge talent fit on EARLIER seasons and
    # weighted by last season's workload. It is a real season-level signal: it
    # predicts a club's win rate better than last season's record (r = +0.587
    # against +0.576) and explains part of what that record misses (r = +0.175,
    # p = 0.007), lifting R^2 from 0.332 to 0.382.
    #
    # It does not convert into game-level edge. Held out 2022-26, paired per
    # game, in three forms: blended into the season-start rating at weights
    # 0.25/0.5/0.75/1.0 (primary w = 0.5: z = -1.37; w = 1.0: z = -2.30), as a
    # flat feature (z = -0.32), and as a feature fading over a club's first
    # thirty games (z = -0.86). Every form lost a little log loss.
    #
    # Worth reading as "too small to detect" rather than "absent": accuracy
    # moved the right way in all three (+0.19, +0.12, +0.29 in April-May, and
    # nowhere else), which is where the mechanism says it should. April and May
    # hold about 1,500 held-out games, so the standard error on that split is
    # near half a point and a quarter-point effect is invisible either way.
    #
    # The scale was not the problem: one standard deviation of the prior is
    # 36.6 rating points against the 33.9 that Elo's own carry spreads clubs
    # over. If a projection feed is ever bought, the scaffolding to carry it is
    # the roster weighting and this blend, and the test to repeat is this one.
    carry: float = 0.75        # season-to-season rating carryover
    mov: bool = True           # scale updates by margin of victory

    def as_dict(self) -> dict[str, Any]:
        return {"k": self.k, "hfa": self.hfa, "carry": self.carry, "mov": self.mov}


def _expected(rating_home: float, rating_away: float, hfa: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(rating_home + hfa - rating_away) / 400.0))


def _mov_multiplier(run_diff: int, rating_diff: float) -> float:
    """Damped margin-of-victory scaling.

    The log form keeps a 10-run win from carrying three times the weight of a
    3-run win. The rating-difference denominator is the standard correction for
    autocorrelation: a strong team beating a weak one by six runs is less
    informative than the same margin between equals, and without the correction
    good teams inflate without bound.
    """
    return float(np.log(abs(run_diff) + 1.0) * (2.2 / (rating_diff * 0.001 + 2.2)))


def run(frame, params: EloParams) -> np.ndarray:
    """Walk the corpus forward, returning each game's pre-game home win probability.

    The returned array aligns with `frame` row for row. Every value is computed
    from ratings that existed *before* that game was played, which is what makes
    Elo leak-free without any as-of bookkeeping.
    """
    ratings: dict[int, float] = {}
    seasons: dict[int, int] = {}
    out = np.empty(len(frame), dtype=float)

    home_ids = frame["home_team_id"].to_numpy()
    away_ids = frame["away_team_id"].to_numpy()
    home_runs = frame["home_runs"].to_numpy()
    away_runs = frame["away_runs"].to_numpy()
    home_win = frame["home_win"].to_numpy()
    season = frame["season"].to_numpy()

    for i in range(len(frame)):
        h, a, s = int(home_ids[i]), int(away_ids[i]), int(season[i])

        # New season: regress toward the mean before the first game is scored.
        for team in (h, a):
            if team not in ratings:
                ratings[team] = MEAN_RATING
                seasons[team] = s
            elif seasons[team] != s:
                ratings[team] = MEAN_RATING + params.carry * (
                    ratings[team] - MEAN_RATING
                )
                seasons[team] = s

        expected = _expected(ratings[h], ratings[a], params.hfa)
        out[i] = expected

        result = float(home_win[i])
        rating_diff = (ratings[h] + params.hfa) - ratings[a]
        if not result:
            rating_diff = -rating_diff

        shift = params.k * (result - expected)
        if params.mov:
            shift *= _mov_multiplier(
                int(home_runs[i]) - int(away_runs[i]), rating_diff
            )

        ratings[h] += shift
        ratings[a] -= shift

    return out


@dataclass
class EloState:
    """Final ratings, for predicting games that have not been played yet."""

    ratings: dict[int, float] = field(default_factory=dict)
    params: EloParams = field(default_factory=EloParams)

    def probability(self, home_team_id: int, away_team_id: int) -> float:
        return _expected(
            self.ratings.get(home_team_id, MEAN_RATING),
            self.ratings.get(away_team_id, MEAN_RATING),
            self.params.hfa,
        )


def fit_state(frame, params: EloParams) -> EloState:
    """Run the corpus through and keep the ratings it ends on."""
    ratings: dict[int, float] = {}
    seasons: dict[int, int] = {}

    for row in frame.itertuples(index=False):
        h, a, s = int(row.home_team_id), int(row.away_team_id), int(row.season)
        for team in (h, a):
            if team not in ratings:
                ratings[team] = MEAN_RATING
                seasons[team] = s
            elif seasons[team] != s:
                ratings[team] = MEAN_RATING + params.carry * (ratings[team] - MEAN_RATING)
                seasons[team] = s

        expected = _expected(ratings[h], ratings[a], params.hfa)
        result = float(row.home_win)
        rating_diff = (ratings[h] + params.hfa) - ratings[a]
        if not result:
            rating_diff = -rating_diff

        shift = params.k * (result - expected)
        if params.mov:
            shift *= _mov_multiplier(int(row.home_runs) - int(row.away_runs), rating_diff)
        ratings[h] += shift
        ratings[a] -= shift

    return EloState(ratings=ratings, params=params)


def ratings_per_game(frame, params: EloParams) -> tuple[np.ndarray, np.ndarray]:
    """Each side's rating *before* the game, aligned to `frame`.

    `run` returns the win probability the two ratings imply, which is what a
    binary model wants. A per-side run model needs the ratings themselves, so
    the batting team's quality and its opponent's can enter separately rather
    than only as a difference.
    """
    ratings: dict[int, float] = {}
    seasons: dict[int, int] = {}
    home_out = np.empty(len(frame))
    away_out = np.empty(len(frame))

    home_ids = frame["home_team_id"].to_numpy()
    away_ids = frame["away_team_id"].to_numpy()
    home_runs = frame["home_runs"].to_numpy()
    away_runs = frame["away_runs"].to_numpy()
    home_win = frame["home_win"].to_numpy()
    season = frame["season"].to_numpy()

    for i in range(len(frame)):
        h, a, s = int(home_ids[i]), int(away_ids[i]), int(season[i])
        for team in (h, a):
            if team not in ratings:
                ratings[team], seasons[team] = MEAN_RATING, s
            elif seasons[team] != s:
                ratings[team] = MEAN_RATING + params.carry * (ratings[team] - MEAN_RATING)
                seasons[team] = s

        home_out[i], away_out[i] = ratings[h], ratings[a]

        expected = _expected(ratings[h], ratings[a], params.hfa)
        result = float(home_win[i])
        rating_diff = (ratings[h] + params.hfa) - ratings[a]
        if not result:
            rating_diff = -rating_diff

        shift = params.k * (result - expected)
        if params.mov:
            shift *= _mov_multiplier(int(home_runs[i]) - int(away_runs[i]), rating_diff)
        ratings[h] += shift
        ratings[a] -= shift

    return home_out, away_out
