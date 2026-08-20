"""Bring the stored ratings up to date before projecting.

The fitted artifact carries two very different kinds of thing. The
**coefficients** are stable -- they describe how starter quality and team
strength convert into runs and wins, and they only change when the model is
refitted. The **ratings** are state, and they go out of date the moment another
game is played.

Shipping both frozen together was a design error: an artifact fitted through
September 2025 rated every club on last year's team, which for an August game is
worse than useless because it looks authoritative.

So the coefficients stay pinned to the fit, and the ratings are walked forward
through the current season's finished games each time a report is built. One
schedule request covers it, the games are already cached by the corpus loader
after the first call, and the update is the same sequential arithmetic the
training run used -- so a rating refreshed here is identical to one that would
have come from refitting.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from guards_report.projections import corpus, elo, ratings, score
from guards_report.projections.model import OutcomeModel


def current_season_games(season: int, *, cache_dir: Path):
    """Finished games of the season so far.

    Deliberately re-fetched rather than read from cache: an in-progress season
    gains games daily, and a stale parquet would reintroduce exactly the problem
    this module exists to fix.
    """
    rows, _ = corpus.fetch_season(season)
    import pandas as pd

    frame = pd.DataFrame(rows, columns=list(corpus.COLUMNS))
    if frame.empty:
        return frame
    frame = frame.drop_duplicates(subset="game_pk", keep="first")
    return frame.query("game_type == 'R'").sort_values(
        ["game_date", "game_pk"]
    ).reset_index(drop=True)


def refresh(model: OutcomeModel, *, on: date, cache_dir: Path) -> tuple[OutcomeModel, dict]:
    """Ratings advanced to `on`, leaving every fitted coefficient untouched.

    Returns the updated model and a note of what was applied, so the report can
    state how current its ratings are rather than implying they are live.
    """
    through = date.fromisoformat(model.corpus_through)
    info = {
        "fitted_through": model.corpus_through,
        "games_applied": 0,
        "ratings_through": model.corpus_through,
        "seasons_advanced": [],
    }
    if on <= through:
        return model, info

    import pandas as pd

    elo_state = elo.EloState(
        ratings={int(k): float(v) for k, v in model.elo_ratings.items()},
        params=elo.EloParams(**model.elo_params),
    )
    od = {int(k): (float(o), float(d)) for k, (o, d) in model.off_def.items()}
    od_params = ratings.OffDefParams(0.010, 0.010, 0.16, 0.70, 0.40)

    applied = 0
    league = model.league_rpg
    carried = 0

    for season in range(through.year, on.year + 1):
        if season > through.year:
            # Roster turnover: regress toward the mean exactly as the training
            # run does at a season boundary. Applied per boundary crossed and
            # before that season's games, whether or not any have been played --
            # on opening day the regression is the whole update.
            carry = elo_state.params.carry
            for team in list(elo_state.ratings):
                elo_state.ratings[team] = elo.MEAN_RATING + carry * (
                    elo_state.ratings[team] - elo.MEAN_RATING
                )
            for team in list(od):
                offence, defence = od[team]
                od[team] = (offence * od_params.carry, defence * od_params.carry)
            carried += 1

        games = current_season_games(season, cache_dir=cache_dir)
        if games.empty:
            continue
        # Only games finished strictly before the date being projected.
        games = games[
            (pd.to_datetime(games["game_date"]).dt.date > through)
            & (pd.to_datetime(games["game_date"]).dt.date < on)
        ]
        if games.empty:
            continue

        for row in games.itertuples(index=False):
            home, away = int(row.home_team_id), int(row.away_team_id)
            for team in (home, away):
                elo_state.ratings.setdefault(team, elo.MEAN_RATING)
                od.setdefault(team, (0.0, 0.0))

            expected = elo_state.probability(home, away)
            result = float(row.home_win)
            spread = (elo_state.ratings[home] + elo_state.params.hfa) - elo_state.ratings[away]
            if not result:
                spread = -spread
            shift = elo_state.params.k * (result - expected)
            if elo_state.params.mov:
                shift *= elo._mov_multiplier(
                    int(row.home_runs) - int(row.away_runs), spread
                )
            elo_state.ratings[home] += shift
            elo_state.ratings[away] -= shift

            expected_home = max(league + od[home][0] + od[away][1] + od_params.hfa, 0.5)
            expected_away = max(league + od[away][0] + od[home][1], 0.5)
            residual_home = float(row.home_runs) - expected_home
            residual_away = float(row.away_runs) - expected_away

            od[home] = (od[home][0] + od_params.k_off * residual_home,
                        od[home][1] + od_params.k_def * residual_away)
            od[away] = (od[away][0] + od_params.k_off * residual_away,
                        od[away][1] + od_params.k_def * residual_home)

            league = 0.999 * league + 0.001 * ((row.home_runs + row.away_runs) / 2.0)
            applied += 1

        info["seasons_advanced"].append(int(season))
        info["ratings_through"] = str(games["game_date"].max())

    info["games_applied"] = applied
    info["seasons_carried"] = carried
    if applied or carried:
        model.elo_ratings = {str(k): float(v) for k, v in elo_state.ratings.items()}
        model.off_def = {str(k): [float(o), float(d)] for k, (o, d) in od.items()}
        model.league_rpg = float(league)

    return model, info
