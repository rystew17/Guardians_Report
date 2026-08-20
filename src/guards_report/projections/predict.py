"""Project a game that has not been played.

The backtest could look every answer up in the corpus. A live projection cannot,
so this module assembles the same feature vector from what is knowable before
first pitch: the stored rating state, the probable starters' season to date, and
the park.

Everything degrades rather than fails. A missing starter, an unrated team, an
unfamiliar park -- each falls back to the fitted mean for that column, and the
projection says which inputs were missing. A report that silently substitutes a
league-average pitcher for an unnamed starter is worse than one that says the
starter is unannounced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np

from guards_report.projections.model import OutcomeModel

MEAN_ELO = 1500.0

# What each model input is called on the page. The model's own column names are
# fine in code and useless to a reader.
FEATURE_LABELS = {
    "elo_logit": "Team rating",
    "sp_k_pct": "Starter strikeout rate",
    "sp_ip_per_start": "Starter innings per start",
    "sp_fip": "Starter FIP",
    "sp_bb_pct": "Starter walk rate",
    "sp_rest_diff": "Starter rest",
    "team_rest_diff": "Team rest",
    "starter_known": "Starter announced",
    "park_factor": "Park",
}


@dataclass
class SideInputs:
    """What the models were told about one club."""

    team_id: int
    team: str
    elo: float
    offense_rating: float
    defense_rating: float
    starter_name: str = ""
    starter_fip: float | None = None
    starter_k_pct: float | None = None
    starter_bb_pct: float | None = None
    starter_ip_per_start: float | None = None
    starter_innings: float | None = None
    offense_rpg: float | None = None


@dataclass
class Projection:
    """A finished projection, with the inputs that produced it."""

    home: SideInputs
    away: SideInputs
    park_factor: float = 1.0
    league_rpg: float = 4.5

    win_probability: float = 0.5          # Model A, direct
    implied_win_probability: float = 0.5  # Model B, simulated
    score: dict[str, Any] = field(default_factory=dict)

    missing: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    model_fitted_at: str = ""
    # How current the ratings are: which games were applied on top of the fit.
    ratings_note: dict[str, Any] = field(default_factory=dict)
    # Per-feature push on the log-odds, and where tonight sits in the model's
    # own historical spread. Both exist so the page can show why, and how
    # unusual, rather than only what.
    contributions: list[dict] = field(default_factory=list)
    confidence_percentile: float = 50.0
    reference: dict[str, Any] = field(default_factory=dict)

    @property
    def coherent(self) -> bool:
        """Whether the two models agree closely enough to publish both.

        They were built independently and cross-checked at fit time (they agreed
        to +0.00029 log loss). A wide disagreement on a specific game means one
        of them is extrapolating, and the report should say so rather than pick
        the friendlier number.
        """
        return abs(self.win_probability - self.implied_win_probability) < 0.10

    @property
    def favourite(self) -> str:
        return self.home.team if self.win_probability >= 0.5 else self.away.team

    @property
    def confidence(self) -> float:
        return max(self.win_probability, 1 - self.win_probability)


def _starter_features(box: Any) -> dict[str, float | None]:
    """Pull the as-of starter line out of a report PlayerBox.

    The report already computes exactly these figures for its own pages, from
    the same game logs and with the same as-of rule, so the projection reuses
    them rather than deriving a second version that could disagree.
    """
    if box is None:
        return {}
    season = getattr(box, "season", {}) or {}

    # The report stores outs rather than innings, because thirds of an inning do
    # not survive being written as a decimal. Divide here rather than reading the
    # display field, which is formatted for people (5.2 means five and two
    # thirds) and would be wrong as arithmetic.
    outs = season.get("outs")
    innings = outs / 3.0 if outs is not None else None
    starts = season.get("gamesStarted") or season.get("games")

    return {
        "fip": season.get("fip"),
        "k_pct": season.get("kPct"),
        "bb_pct": season.get("bbPct"),
        # Left as None when either half is missing. Falling back to zero here
        # would hand the model a pitcher who records no outs, which it would
        # read as the worst starter in the corpus rather than as an unknown.
        "ip_per_start": (innings / starts) if innings is not None and starts else None,
        "innings": innings,
    }


def _side(model: OutcomeModel, team_id: int, team: str, box: Any) -> SideInputs:
    offense, defense = model.off_def.get(str(team_id), [0.0, 0.0])
    stats = _starter_features(box)
    return SideInputs(
        team_id=team_id,
        team=team,
        elo=model.elo_ratings.get(str(team_id), MEAN_ELO),
        offense_rating=float(offense),
        defense_rating=float(defense),
        starter_name=getattr(box, "name", "") if box is not None else "",
        starter_fip=stats.get("fip"),
        starter_k_pct=stats.get("k_pct"),
        starter_bb_pct=stats.get("bb_pct"),
        starter_ip_per_start=stats.get("ip_per_start"),
        starter_innings=stats.get("innings"),
    )


def _clean(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(number) else number


def project(
    model: OutcomeModel,
    *,
    home_team_id: int,
    away_team_id: int,
    home_team: str,
    away_team: str,
    venue_id: int | None = None,
    home_starter: Any = None,
    away_starter: Any = None,
    home_offense_rpg: float | None = None,
    away_offense_rpg: float | None = None,
) -> Projection:
    """Run both models against one upcoming game."""
    home = _side(model, home_team_id, home_team, home_starter)
    away = _side(model, away_team_id, away_team, away_starter)
    home.offense_rpg = _clean(home_offense_rpg)
    away.offense_rpg = _clean(away_offense_rpg)

    park = model.park_factors.get(str(venue_id), 1.0) if venue_id else 1.0

    missing: list[str] = []
    if str(home_team_id) not in model.elo_ratings:
        missing.append(f"no rating for {home_team}")
    if str(away_team_id) not in model.elo_ratings:
        missing.append(f"no rating for {away_team}")
    for side in (home, away):
        if _clean(side.starter_fip) is None:
            missing.append(
                f"{side.team} starter"
                + (f" ({side.starter_name})" if side.starter_name else "")
                + " has no season line yet"
            )

    # --- Model A: the Core block, home minus away ------------------------
    elo_diff = home.elo + model.elo_params.get("hfa", 24.0) - away.elo
    elo_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    elo_logit = float(np.log(elo_prob / (1 - elo_prob)))

    def diff(attr: str, higher_is_better: bool) -> float | None:
        h, a = _clean(getattr(home, attr)), _clean(getattr(away, attr))
        if h is None or a is None:
            return None
        return (h - a) if higher_is_better else (a - h)

    win_features = {
        "elo_logit": elo_logit,
        "sp_fip": diff("starter_fip", False),
        "sp_k_pct": diff("starter_k_pct", True),
        "sp_bb_pct": diff("starter_bb_pct", False),
        "sp_ip_per_start": diff("starter_ip_per_start", True),
        "starter_known": 1.0 if not any("starter" in m for m in missing) else 0.0,
        "team_rest_diff": 0.0,
        "sp_rest_diff": 0.0,
        "park_factor": park,
    }
    win_probability = model.win_probability(win_features)
    contributions = model.win_contributions(win_features)

    # --- Model B: expected runs per side ---------------------------------
    def score_features(batting: SideInputs, fielding: SideInputs, is_home: int) -> dict:
        return {
            "off_rpg": batting.offense_rpg,
            "opp_off_rpg": fielding.offense_rpg,
            "opp_sp_fip": _clean(fielding.starter_fip),
            "opp_sp_k": _clean(fielding.starter_k_pct),
            "opp_sp_bb": _clean(fielding.starter_bb_pct),
            "opp_sp_ip": _clean(fielding.starter_ip_per_start),
            "park_factor": park,
            "is_home": float(is_home),
            "elo_diff": batting.elo - fielding.elo,
            # off_def predicts this side's runs directly.
            "od_exp_runs": (
                model.league_rpg + batting.offense_rating + fielding.defense_rating
                + (0.16 if is_home else 0.0)
            ),
            "sp_known": 1.0 if _clean(fielding.starter_fip) is not None else 0.0,
        }

    simulation = model.simulate(
        score_features(home, away, 1), score_features(away, home, 0)
    )

    return Projection(
        home=home,
        away=away,
        park_factor=park,
        league_rpg=model.league_rpg,
        win_probability=win_probability,
        implied_win_probability=simulation["home_win_probability"],
        score=simulation,
        missing=missing,
        contributions=contributions,
        confidence_percentile=model.percentile_of(win_probability),
        reference=model.reference,
        metrics=model.metrics,
        model_fitted_at=model.fitted_at,
    )


def read_of(projection: Projection) -> dict[str, Any]:
    """The one sentence the decomposition supports, assembled from its numbers.

    The waterfall shows what moved the probability, but the comparison a reader
    should take away is between its two largest terms, and that comparison
    changes every night. Some games are won on the rating -- one club is simply
    better -- and some are a coin flip between two ordinary teams where the
    starter is doing all the work. Those read very differently and look almost
    identical as a bar chart.

    Every branch here is a threshold on measured quantities. No text is
    generated; the sentence is selected by arithmetic, in the same way the rest
    of the report's prose is.
    """
    rows = [r for r in (projection.contributions or []) if abs(r["contribution"]) > 1e-4]
    if not rows:
        return {}

    favoured = projection.favourite
    top = rows[0]
    second = rows[1] if len(rows) > 1 else None
    label = FEATURE_LABELS.get(top["name"], top["name"])

    total = sum(abs(r["contribution"]) for r in rows) or 1.0
    top_share = abs(top["contribution"]) / total

    # Does the rating lead, or does something about tonight override it?
    rating = next((r for r in rows if r["name"] == "elo_logit"), None)
    starters = [r for r in rows if r["name"].startswith("sp_")]
    starter_push = sum(r["contribution"] for r in starters)
    rating_push = rating["contribution"] if rating else 0.0

    if second is not None and abs(second["contribution"]) >= 0.85 * abs(
        top["contribution"]
    ):
        shape = "shared"
        second_label = FEATURE_LABELS.get(second["name"], second["name"])
        sentence = (
            f"{label.lower()} and {second_label.lower()} carry "
            f"{favoured}’s edge in almost equal measure"
        )
    elif top_share >= 0.55:
        shape = "dominant"
        sentence = f"nearly all of {favoured}’s edge comes from {label.lower()}"
    else:
        shape = "led"
        sentence = f"{label.lower()} is the largest single factor"

    # The genuinely notable case: tonight's pitching matchup outweighing the
    # standing difference between the clubs.
    if abs(starter_push) > abs(rating_push) and starters:
        note = (
            "the starters matter more here than the gap between the clubs does"
        )
    elif rating is not None and abs(rating_push) >= 0.6 * total:
        note = "this is a difference in team quality more than a matchup"
    else:
        note = ""

    return {
        "sentence": sentence,
        "note": note,
        # How many starter bars the chart draws, so the prose can say which
        # bars it is adding up rather than quoting a total that appears
        # nowhere on the chart.
        "starter_count": len(starters),
        "shape": shape,
        "top_label": label,
        "top_share": top_share,
        "starter_push": starter_push,
        "rating_push": rating_push,
        "favourite": favoured,
    }
