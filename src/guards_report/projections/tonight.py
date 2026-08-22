"""Assemble every projection for one game.

Four models with four artifacts, one call. Kept separate from `predict.py`,
which owns the full-game win and score models, because these have their own
failure modes and a report should lose one section rather than the page: a
missing props artifact should quieten the player table, not stop a win
probability from rendering.

Every estimate here is carried forward from a stored prior through the plate
appearances of the current season, using only those strictly before the date
being projected. That is the same computation the models were fitted with, which
is the property that makes the served numbers mean what the held-out figures
say they mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from guards_report.projections import first5 as f5_module
from guards_report.projections import props as props_module


@dataclass
class PlayerProp:
    """One batter's expected totals tonight."""

    player_id: int
    name: str = ""
    slot: int | None = None
    hits: dict[str, Any] = field(default_factory=dict)
    home_runs: dict[str, Any] = field(default_factory=dict)
    evidence: int = 0

    @property
    def thin(self) -> bool:
        """Whether the estimate rests on too little to be worth printing."""
        return self.evidence < 100


@dataclass
class StarterProp:
    """A starting pitcher's strikeout total."""

    player_id: int
    name: str = ""
    expected: float = 0.0
    distribution: dict[int, float] = field(default_factory=dict)
    batters_faced: float = 0.0
    evidence: int = 0

    def at_least(self, k: int) -> float:
        return float(sum(p for n, p in self.distribution.items() if n >= k))

    @property
    def line(self) -> float:
        """The half-integer total closest to an even split, as a book would set it."""
        for k in range(0, 20):
            if self.at_least(k + 1) < 0.5:
                return k + 0.5
        return 9.5


@dataclass
class FirstFive:
    """The three-way first-five result and the score behind it."""

    home_leads: float = 0.0
    tied: float = 0.0
    away_leads: float = 0.0
    expected_home: float = 0.0
    expected_away: float = 0.0
    expected_total: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def favourite(self) -> str:
        return "home" if self.home_leads >= self.away_leads else "away"

    # Share of a game's runs that fall in the first five innings, measured at
    # 0.565 across the corpus and flat from low-scoring games to high.
    expected_share: float = 0.565
    full_game_total: float | None = None

    @property
    def coherent(self) -> bool:
        """Sums to one, and agrees with the full-game model about the scoring.

        Summing to one is arithmetic and cannot fail. The check that can is
        whether this model and the score model describe the same game: they are
        fitted independently, and if the first five hold 70% of the runs the
        full-game model expects, one of them is extrapolating. The tolerance is
        wide because a genuine pitching matchup does shift the share.
        """
        if abs(self.home_leads + self.tied + self.away_leads - 1.0) >= 0.01:
            return False
        if not self.full_game_total:
            return True
        return abs(self.share_of_game - self.expected_share) < 0.08

    @property
    def share_of_game(self) -> float:
        """What fraction of the projected runs this model puts in five innings."""
        if not self.full_game_total:
            return self.expected_share
        return self.expected_total / self.full_game_total


def _as_of_rates(
    prior: props_module.RateModel, plate: pd.DataFrame, *, on: date, side: str
) -> dict[int, float]:
    """Each player's rate carried forward through this season, before `on`."""
    current = plate[plate["game_date"] < on]
    if current.empty:
        return {}
    running = props_module.running_rates_decayed(prior, current, side=side)
    if running.empty:
        return {}
    latest = running.sort_values("game_date").groupby(side).last()
    return {int(k): float(v) for k, v in latest["rate"].items()}


def batter_props(
    artifact: Any,
    plate: pd.DataFrame,
    *,
    on: date,
    lineup: list[int],
    names: dict[int, str],
    opposing_starter: int | None,
    opposing_throws: str,
    stands: dict[int, str],
    home_team: str,
) -> list[PlayerProp]:
    """Hit and home-run totals for one club's posted nine.

    Returns an empty list when no card has been posted. A projection for nine
    players the manager has not named is a guess about the lineup dressed as a
    statement about the hitters, and the report already has a place to say the
    card is not out yet.
    """
    if not lineup or opposing_starter is None:
        return []

    out: list[PlayerProp] = []
    rates = {o: artifact.rate_model(o) for o in ("hit", "home_run")}
    as_of = {
        o: _as_of_rates(rates[o], plate, on=on, side="batter")
        for o in ("hit", "home_run")
    }
    pitcher_as_of = {
        o: _as_of_rates(rates[o], plate, on=on, side="pitcher")
        for o in ("hit", "home_run")
    }

    for index, batter in enumerate(lineup[:9], start=1):
        prop = PlayerProp(player_id=int(batter), name=names.get(int(batter), ""), slot=index)
        for outcome, target in (("hit", "hits"), ("home_run", "home_runs")):
            model = rates[outcome]
            batter_rate = as_of[outcome].get(int(batter), model.batter_rate(batter))
            pitcher_rate = pitcher_as_of[outcome].get(
                int(opposing_starter), model.pitcher_rate(opposing_starter)
            )
            rate = props_module.log5(batter_rate, pitcher_rate, model.league)
            rate = props_module.adjust(
                rate,
                artifact.platoon.get(outcome, {}).get(
                    f"{stands.get(int(batter), 'R')}{opposing_throws}", 1.0
                ),
                artifact.park.get(outcome, {}).get(home_team, 1.0),
                props_module.CALIBRATION.get(outcome, 1.0),
            )
            chances = props_module.SLOT_PA_DISTRIBUTION.get(
                index, props_module.UNKNOWN_SLOT_PA
            )
            projection = props_module.count_distribution(rate, chances)
            setattr(prop, target, {
                "expected": projection.expected,
                "at_least_one": projection.at_least(1),
                "at_least_two": projection.at_least(2),
                "per_pa": projection.per_chance,
                "chances": projection.expected_chances,
                "distribution": projection.distribution,
            })
        prop.evidence = rates["hit"].evidence(batter, side="batter")
        out.append(prop)
    return out


def starter_strikeouts(
    artifact: Any,
    plate: pd.DataFrame,
    *,
    on: date,
    pitcher_id: int,
    name: str,
    opposing_lineup: list[int],
    stands: dict[int, str],
    throws: str,
    expected_bf: float | None = None,
) -> StarterProp | None:
    """Strikeout total for one starter against the nine he will face."""
    if pitcher_id is None:
        return None

    model = artifact.rate_model("strikeout")
    batter_as_of = _as_of_rates(model, plate, on=on, side="batter")
    pitcher_as_of = _as_of_rates(model, plate, on=on, side="pitcher")

    # A local copy so the as-of rates are used rather than the stored prior.
    live = props_module.RateModel(
        outcome="strikeout",
        league=model.league,
        stabilisation=model.stabilisation,
        batter={**model.batter, **batter_as_of},
        pitcher={**model.pitcher, **pitcher_as_of},
        batter_pa=model.batter_pa,
        pitcher_pa=model.pitcher_pa,
        through=str(on),
    )

    projection = props_module.starter_strikeouts(
        live,
        pitcher_id=int(pitcher_id),
        lineup_ids=[int(b) for b in (opposing_lineup or [])],
        expected_bf=expected_bf or artifact.starter_bf_mean,
        platoon=artifact.platoon.get("strikeout"),
        stands=stands,
        throws=throws,
    )
    return StarterProp(
        player_id=int(pitcher_id),
        name=name,
        expected=projection.expected,
        distribution=projection.distribution,
        batters_faced=projection.expected_chances,
        evidence=model.evidence(pitcher_id, side="pitcher"),
    )


def starter_first_five(history: pd.DataFrame, pitcher_id: int, *, on: date) -> dict:
    """A starter's most recent first-five line, as of the date being projected.

    The stored table is keyed by the start it precedes, so the live lookup takes
    his latest row and nothing after it. Returns empty when he has too little
    history, which the model reads through `f5_line_known` rather than through a
    number invented to fill the gap.
    """
    if history is None or pitcher_id is None or history.empty:
        return {}
    rows = history[history["starter"] == int(pitcher_id)]
    if "game_date" in rows.columns:
        rows = rows[pd.to_datetime(rows["game_date"]).dt.date < on]
    rows = rows.dropna(subset=["sp_f5_ra"])
    if rows.empty:
        return {}
    latest = rows.iloc[-1]
    return {
        "opp_f5_ra": float(latest["sp_f5_ra"]),
        "opp_f5_bf": float(latest["sp_f5_bf"]),
        "f5_line_known": 1.0,
    }


def first_five(artifact: Any, features: dict[str, float]) -> FirstFive | None:
    """Runs through five for both sides, and the three-way result.

    `features` carries one entry per column for each side, keyed `home_` and
    `away_`. Anything missing falls back to the fitted mean, matching how every
    other model in this project handles absence.
    """
    if artifact is None or not artifact.columns:
        return None

    def expected(prefix: str) -> float:
        linear = 0.0
        for column, coefficient, mean in zip(
            artifact.columns, artifact.coef, artifact.mean
        ):
            value = features.get(f"{prefix}{column}")
            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = mean
            linear += coefficient * value
        offset = np.log(max(features.get("league_rpg", 4.5) * artifact.share, 0.3))
        return float(np.exp(linear + offset))

    home_mu, away_mu = expected("home_"), expected("away_")
    outcome = f5_module.outcome_probabilities(
        np.array([home_mu]), np.array([away_mu]), alpha=artifact.alpha
    )
    return FirstFive(
        home_leads=float(outcome["home_leads"][0]),
        tied=float(outcome["tied"][0]),
        away_leads=float(outcome["away_leads"][0]),
        expected_home=float(outcome["expected_home"][0]),
        expected_away=float(outcome["expected_away"][0]),
        expected_total=float(outcome["expected_total"][0]),
        metrics=artifact.metrics,
    )
