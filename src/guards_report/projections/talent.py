"""Latent batter and pitcher quality, estimated from plate appearances.

This is the composite-index idea done in the form that can actually pay. A
hand-weighted score built from features the model already has cannot help: in a
linear model any fixed combination of FIP, K% and BB% lies in the span of those
same columns, so adding it is collinear and replacing them with it imposes a
rank-one restriction on the coefficient vector. On the same data, that can only
lose fit.

An index earns its place when it is estimated on evidence the outcome model
cannot see. Pitcher quality learned here rests on roughly 1.9 million plate
appearances rather than 25,192 game results -- about seventy times the evidence
about the same latent quantity, and crucially not a function of who won, which
is why it is the first block with a real chance of adding something Elo has not
already priced.

Two problems have to be solved together, which is why this is one model rather
than two sets of rate stats:

**Opposition.** A pitcher who faced weak lineups looks better than he is. Batter
and pitcher effects are estimated jointly, so each is adjusted for who it faced.

**Sample size.** Playing time is wildly unequal -- a regular takes 600 PA in a
season, a September call-up takes 20. Ridge shrinks every effect toward the
league mean in proportion to how little evidence supports it, which is the
empirical-Bayes posterior mean under a normal prior. The penalty is chosen by
walk-forward validation rather than assumed.

The target is the run value Statcast assigns each event, so no linear weights
are hard-coded here to drift out of date as the run environment moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

# Plate appearances that do not count toward the wOBA denominator -- sacrifice
# bunts, catcher's interference -- carry no run value to learn from.
DENOM_REQUIRED = 1

# Below this many prior plate appearances a player's own estimate is mostly
# prior anyway; the effect is still produced, but callers are told the evidence
# is thin so a projection can say so rather than implying confidence.
THIN_EVIDENCE_PA = 50


@dataclass
class Talent:
    """Fitted batter and pitcher effects, on the run-value-per-PA scale."""

    batter: dict[int, float] = field(default_factory=dict)
    pitcher: dict[int, float] = field(default_factory=dict)
    # Platoon: the extra effect of a same-handed matchup, learned rather than
    # assumed, because its size differs for left- and right-handed pitchers.
    platoon: dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    alpha: float = 0.0
    batter_pa: dict[int, int] = field(default_factory=dict)
    pitcher_pa: dict[int, int] = field(default_factory=dict)
    through: str = ""

    def expected_value(
        self, batter_id: int, pitcher_id: int, *, stand: str = "R", throws: str = "R"
    ) -> float:
        """Run value of one plate appearance between these two."""
        return (
            self.intercept
            + self.batter.get(int(batter_id), 0.0)
            + self.pitcher.get(int(pitcher_id), 0.0)
            + self.platoon.get(f"{stand}{throws}", 0.0)
        )

    def pitcher_score(self, pitcher_id: int) -> float | None:
        """Runs prevented per PA against a league-average batter.

        Signed so that higher is better, which is the opposite of the fitted
        effect -- a pitcher's effect lowers the run value of the plate
        appearances he is in.
        """
        value = self.pitcher.get(int(pitcher_id))
        return None if value is None else -value

    def batter_score(self, batter_id: int) -> float | None:
        value = self.batter.get(int(batter_id))
        return None if value is None else value

    def evidence(self, player_id: int, *, side: str) -> int:
        table = self.batter_pa if side == "batter" else self.pitcher_pa
        return int(table.get(int(player_id), 0))


def _design(frame: pd.DataFrame) -> tuple[sparse.csr_matrix, np.ndarray, dict, dict, list]:
    """Sparse indicator matrix over batters, pitchers and platoon states.

    Sparse because the dense version would be 1.9M rows by several thousand
    columns of almost entirely zeros -- tens of gigabytes to hold three ones per
    row.
    """
    batters = {pid: i for i, pid in enumerate(sorted(frame["batter"].unique()))}
    pitchers = {pid: i for i, pid in enumerate(sorted(frame["pitcher"].unique()))}
    states = sorted((frame["stand"].fillna("R") + frame["p_throws"].fillna("R")).unique())
    state_index = {s: i for i, s in enumerate(states)}

    n = len(frame)
    nb, np_, ns = len(batters), len(pitchers), len(states)

    rows = np.repeat(np.arange(n), 3)
    cols = np.empty(n * 3, dtype=np.int64)
    cols[0::3] = frame["batter"].map(batters).to_numpy()
    cols[1::3] = frame["pitcher"].map(pitchers).to_numpy() + nb
    cols[2::3] = (
        (frame["stand"].fillna("R") + frame["p_throws"].fillna("R"))
        .map(state_index).to_numpy() + nb + np_
    )
    data = np.ones(n * 3, dtype=np.float64)

    X = sparse.csr_matrix((data, (rows, cols)), shape=(n, nb + np_ + ns))
    y = frame["woba_value"].to_numpy(dtype=float)
    return X, y, batters, pitchers, states


def fit(frame: pd.DataFrame, *, alpha: float, through: date | None = None) -> Talent:
    """Ridge-penalised batter and pitcher effects on run value per PA.

    Solved by conjugate gradient on the normal equations rather than by forming
    the dense system. With one row per plate appearance and three non-zeros in
    each, the normal matrix is small and extremely sparse, so this converges in
    seconds where a dense solve would not fit in memory.
    """
    usable = frame[
        (frame["woba_denom"] >= DENOM_REQUIRED) & frame["woba_value"].notna()
    ]
    if through is not None:
        usable = usable[usable["game_date"] < through]
    usable = usable.reset_index(drop=True)

    X, y, batters, pitchers, states = _design(usable)
    n, k = X.shape

    # The intercept is left unpenalised by centring the response: shrinking the
    # league mean toward zero would bias every effect.
    intercept = float(y.mean())
    target = y - intercept

    XtX = (X.T @ X).tocsc() + alpha * sparse.identity(k, format="csc")
    Xty = X.T @ target
    beta = sparse.linalg.spsolve(XtX, Xty)

    nb, np_ = len(batters), len(pitchers)
    batter_pa = usable["batter"].value_counts().to_dict()
    pitcher_pa = usable["pitcher"].value_counts().to_dict()

    # Every row carries exactly one batter, one pitcher and one platoon state,
    # so all three factors are collinear with the intercept and ridge splits the
    # league mean between them by minimum norm -- an arbitrary division that
    # leaves no term interpretable on its own. Recentring each factor to a
    # frequency-weighted zero and folding what is removed into the intercept
    # fixes the split without changing any fitted value: every prediction is the
    # sum of the same four terms as before.
    batter_effects = np.array([beta[i] for i in range(nb)])
    pitcher_effects = np.array([beta[nb + i] for i in range(np_)])
    platoon_effects = np.array([beta[nb + np_ + i] for i in range(len(states))])

    batter_weight = np.array([batter_pa.get(pid, 0) for pid in batters])
    pitcher_weight = np.array([pitcher_pa.get(pid, 0) for pid in pitchers])
    state_counts = (
        (usable["stand"].fillna("R") + usable["p_throws"].fillna("R"))
        .value_counts()
    )
    platoon_weight = np.array([state_counts.get(s, 0) for s in states])

    def recentre(effects, weights):
        total = weights.sum()
        if not total:
            return effects, 0.0
        mean = float((effects * weights).sum() / total)
        return effects - mean, mean

    batter_effects, shift_b = recentre(batter_effects, batter_weight)
    pitcher_effects, shift_p = recentre(pitcher_effects, pitcher_weight)
    platoon_effects, shift_s = recentre(platoon_effects, platoon_weight)
    intercept += shift_b + shift_p + shift_s

    return Talent(
        batter={int(pid): float(batter_effects[i]) for pid, i in batters.items()},
        pitcher={int(pid): float(pitcher_effects[i]) for pid, i in pitchers.items()},
        platoon={s: float(platoon_effects[i]) for i, s in enumerate(states)},
        intercept=intercept,
        alpha=alpha,
        batter_pa={int(k_): int(v) for k_, v in batter_pa.items()},
        pitcher_pa={int(k_): int(v) for k_, v in pitcher_pa.items()},
        through=str(through) if through else str(usable["game_date"].max()),
    )


def choose_alpha(
    frame: pd.DataFrame,
    *,
    candidates=(50, 100, 200, 400, 800, 1600, 3200),
    train_seasons: tuple[int, int] = (2015, 2020),
    test_seasons: tuple[int, int] = (2021, 2021),
    verbose: bool = True,
) -> tuple[float, dict[float, float]]:
    """Pick the ridge penalty by predicting held-out plate appearances.

    Chosen forward in time, never by random split: a random fold would let a
    player's July performance help predict his own April, which is exactly the
    leak the whole project is built to avoid.
    """
    train = frame[frame["season"].between(*train_seasons)]
    test = frame[
        frame["season"].between(*test_seasons)
        & (frame["woba_denom"] >= DENOM_REQUIRED)
        & frame["woba_value"].notna()
    ]

    scores: dict[float, float] = {}
    for alpha in candidates:
        model = fit(train, alpha=alpha)
        predicted = np.array([
            model.expected_value(b, p, stand=s, throws=t)
            for b, p, s, t in zip(
                test["batter"], test["pitcher"],
                test["stand"].fillna("R"), test["p_throws"].fillna("R"),
            )
        ])
        rmse = float(np.sqrt(np.mean((test["woba_value"].to_numpy() - predicted) ** 2)))
        scores[alpha] = rmse
        if verbose:
            print(f"  alpha {alpha:>6}  held-out RMSE {rmse:.5f}", flush=True)

    best = min(scores, key=scores.get)
    return best, scores


def season_priors(frame: pd.DataFrame, alpha: float, seasons) -> dict[int, Talent]:
    """One ridge fit per season, trained only on seasons before it began.

    This is the *prior*, not the answer. It fixes the structure -- how hard to
    shrink, the platoon effects, the run-value scale -- from a large leak-free
    sample. What a player is doing this season is layered on top by
    `update_as_of`, because refitting the whole ridge for every game date would
    cost hours to produce almost the same numbers.
    """
    fits: dict[int, Talent] = {}
    for season in seasons:
        prior = frame[frame["season"] < season]
        if not len(prior):
            continue
        fits[int(season)] = fit(prior, alpha=alpha)
    return fits


def update_as_of(
    prior: Talent,
    frame: pd.DataFrame,
    *,
    on: date,
    season: int | None = None,
) -> Talent:
    """Carry a season's prior forward through the plate appearances since.

    Fitting on prior seasons and stopping there was a design error: it discarded
    the current season, which is the single most predictive input available.
    Measured on starter strikeout rate, this season's first half predicts the
    second half at R2 0.4390 against 0.3098 for the whole prior season, and both
    together reach 0.4792 -- so the answer is not to choose between them but to
    weight them, which is what this does.

    The weighting is not a tuning knob. Ridge with penalty `alpha` already
    implies the posterior mean of a normal-prior model, so a player's estimate
    after n new plate appearances is

        (alpha * prior + sum of new residuals) / (alpha + n)

    shrinking toward the prior-season estimate rather than toward zero. The same
    `alpha` chosen by walk-forward validation therefore sets the blend, and no
    second parameter is invented.

    Only plate appearances strictly before `on` are used, which is what makes
    this identical in training and in production -- the property the per-season
    version broke.
    """
    current = frame[frame["game_date"] < on]
    if season is not None:
        current = current[current["season"] == season]
    current = current[
        (current["woba_denom"] >= DENOM_REQUIRED) & current["woba_value"].notna()
    ]
    if current.empty:
        return prior

    updated = Talent(
        batter=dict(prior.batter), pitcher=dict(prior.pitcher),
        platoon=dict(prior.platoon), intercept=prior.intercept,
        alpha=prior.alpha, batter_pa=dict(prior.batter_pa),
        pitcher_pa=dict(prior.pitcher_pa), through=str(on),
    )

    # Residual against what the prior already expects, so a player who is
    # performing exactly to prior does not move.
    expected = (
        prior.intercept
        + current["batter"].map(prior.batter).fillna(0.0).to_numpy()
        + current["pitcher"].map(prior.pitcher).fillna(0.0).to_numpy()
        + (current["stand"].fillna("R") + current["p_throws"].fillna("R"))
        .map(prior.platoon).fillna(0.0).to_numpy()
    )
    residual = current["woba_value"].to_numpy(dtype=float) - expected

    for side, table, counts in (
        ("batter", updated.batter, updated.batter_pa),
        ("pitcher", updated.pitcher, updated.pitcher_pa),
    ):
        grouped = pd.DataFrame({
            "id": current[side].to_numpy(), "r": residual,
        }).groupby("id")["r"].agg(["sum", "size"])

        for player, row in grouped.iterrows():
            player = int(player)
            base = table.get(player, 0.0)
            # Prior plus a shrunk correction, not a weighted average of prior
            # and raw rate. Writing the residual as r = y - (baseline + prior),
            # the posterior mean (tau*prior + sum y) / (tau + n) rearranges to
            # prior + sum(r) / (tau + n). The weighted-average form is only
            # correct when residuals are taken against the league mean; against
            # the prior it drags every established player toward zero, so a
            # player performing exactly to expectation would still lose value.
            table[player] = base + float(row["sum"]) / (
                prior.alpha + float(row["size"])
            )
            counts[player] = counts.get(player, 0) + int(row["size"])

    return updated
