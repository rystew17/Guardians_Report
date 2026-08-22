"""The first five evaluators.

Chosen to exercise different machinery rather than to cover the most ground: a
rate model, a change-point search, the pitch corpus, a per-pitch aggregation and
a team-level comparison. If five well-chosen findings do not read better than
the model's paragraph, forty will not close the gap, and that comparison is the
point of building only five.

Each is a pure function from computed figures to zero or more findings, which is
the practical payoff over a prompt: "does a hitter with this line trigger the
pull-heavy finding" is a unit test, and the equivalent question about a prompt
is not.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from guards_report.insight.types import Finding, Reference


def _reference(values: np.ndarray, population: str) -> Reference:
    """League distribution for one metric, measured not assumed."""
    clean = np.asarray([v for v in values if v is not None and np.isfinite(v)])
    if len(clean) < 2:
        return Reference(mean=0.0, sd=1.0, population=population, n=len(clean))
    return Reference(
        mean=float(clean.mean()), sd=float(clean.std()) or 1.0,
        population=population, n=len(clean),
    )


# ---------------------------------------------------------------------------
# 1. Overall rate quality — how good is this hitter, on one measured axis
# ---------------------------------------------------------------------------

# Whether a higher rate is good news for this side. Without it, `direction` is
# just the sign of a z-score and says nothing about whether the finding is a
# strength or a weakness -- a high strikeout rate is excellent for a pitcher and
# the reverse for a hitter, and the contrast detector depends on telling them
# apart.
POLARITY = {
    ("batter", "hit"): +1, ("batter", "home_run"): +1, ("batter", "strikeout"): -1,
    ("pitcher", "hit"): -1, ("pitcher", "home_run"): -1, ("pitcher", "strikeout"): +1,
}


def rate_quality(
    *, player_id: int, rate: float, evidence: int, reference: Reference,
    stabilisation: int, outcome: str, side: str = "batter",
) -> list[Finding]:
    """A player's rate against the league, shrunk by how much evidence backs it.

    The shrinkage is what makes this printable. Ranked on the raw rate, the page
    would fill with twenty-at-bat hot streaks, because the largest deviations
    are always found in the smallest samples.
    """
    family = {"hit": "contact", "home_run": "power", "strikeout": "discipline"}
    # Signed so that positive always means "better than the reference", whatever
    # the metric. The raw rate is kept in `detail` for the sentence to print.
    sign = POLARITY.get((side, outcome), 1)
    return [Finding(
        subject=int(player_id),
        subject_kind=side,
        code=f"{side[:3]}.rate.{outcome}",
        family=family.get(outcome, outcome),
        kind="skill" if sign > 0 else "skill",
        value=sign * float(rate),
        reference=Reference(
            mean=sign * reference.mean, sd=reference.sd,
            population=reference.population, n=reference.n,
        ),
        evidence=int(evidence),
        stabilisation=int(stabilisation),
        detail={"outcome": outcome, "rate": float(rate)},
    )]


# ---------------------------------------------------------------------------
# 2. Trend — and, honestly, for how long
# ---------------------------------------------------------------------------

def longest_significant_window(
    series: pd.Series, *, baseline: float, min_length: int = 15,
    max_length: int = 120, alpha: float = 0.05,
) -> tuple[int, float, float] | None:
    """The longest recent stretch that differs from a player's own baseline.

    "Trending up -- for how long?" invites the obvious wrong answer. Trying L5,
    L10, L15 and L30 and reporting the most dramatic is the same
    multiple-comparisons trap through the back door, and it will always find
    something.

    Scanning from the longest window down and stopping at the first that clears
    a corrected threshold answers the question actually asked, and returns
    nothing when nothing qualifies -- which is most players most nights, and is
    the correct answer.
    """
    from scipy import stats

    values = series.dropna().to_numpy(dtype=float)
    if len(values) < min_length:
        return None

    # One test per candidate window, so the threshold is corrected for how many
    # were tried rather than for the one that happened to look best.
    candidates = range(min(max_length, len(values)), min_length - 1, -1)
    corrected = alpha / max(len(list(candidates)), 1)

    for length in range(min(max_length, len(values)), min_length - 1, -1):
        window = values[-length:]
        if window.std() == 0:
            continue
        t, p = stats.ttest_1samp(window, baseline)
        if p < corrected:
            return length, float(window.mean()), float(p)
    return None


def form_trend(
    *, player_id: int, series: pd.Series, baseline: float, reference: Reference,
    side: str = "batter",
) -> list[Finding]:
    """A change in form, reported with the window that actually supports it."""
    found = longest_significant_window(series, baseline=baseline)
    if found is None:
        return []

    length, mean, p = found
    return [Finding(
        subject=int(player_id),
        subject_kind=side,
        code=f"{side[:3]}.trend.window",
        family="trend",
        kind="trend",
        value=float(mean),
        reference=reference,
        # The window is the evidence, and a longer one is worth more.
        evidence=int(length),
        stabilisation=30,
        detail={"games": int(length), "baseline": float(baseline), "p": float(p)},
    )]


# ---------------------------------------------------------------------------
# 3. A pitcher's best offering — needs the pitch corpus
# ---------------------------------------------------------------------------

def best_pitch(
    *, pitcher_id: int, pitches: pd.DataFrame, league: pd.DataFrame,
    minimum: int = 150,
) -> list[Finding]:
    """The offering a pitcher gets the most out of, against league for its type.

    Judged within pitch type rather than across. A .280 expected wOBA on a
    slider means something different from the same figure on a four-seamer, and
    comparing them directly would simply rank pitch types.
    """
    if pitches.empty or "pitch_name" not in pitches.columns:
        return []

    thrown = pitches[pitches["pitcher"] == int(pitcher_id)]
    if len(thrown) < minimum:
        return []

    league_by_type = (
        league.groupby("pitch_name")["estimated_woba_using_speedangle"]
        .agg(["mean", "std", "size"])
    )
    rows = thrown.groupby("pitch_name").agg(
        value=("estimated_woba_using_speedangle", "mean"),
        n=("estimated_woba_using_speedangle", "size"),
    )
    rows = rows[rows["n"] >= 40]
    if rows.empty:
        return []

    findings = []
    for pitch_name, row in rows.iterrows():
        if pitch_name not in league_by_type.index:
            continue
        stats_row = league_by_type.loc[pitch_name]
        reference = Reference(
            mean=float(stats_row["mean"]),
            sd=float(stats_row["std"]) or 1.0,
            population=f"league_{pitch_name}",
            n=int(stats_row["size"]),
        )
        findings.append(Finding(
            subject=int(pitcher_id),
            subject_kind="pitcher",
            code="pit.arsenal.pitch",
            family=f"arsenal_{pitch_name}",
            kind="skill",
            # Signed so that better is positive: a pitcher wants a low expected
            # wOBA against, and the sign has to say so or the ranking inverts.
            value=-float(row["value"]),
            reference=Reference(
                mean=-reference.mean, sd=reference.sd,
                population=reference.population, n=reference.n,
            ),
            evidence=int(row["n"]),
            stabilisation=100,
            detail={"pitch": str(pitch_name), "xwoba": float(row["value"]),
                    "league": float(stats_row["mean"]), "thrown": int(row["n"]),
                    "usage": float(row["n"] / len(thrown))},
        ))
    return findings


# ---------------------------------------------------------------------------
# 4. Times through the order — where a starter starts to go
# ---------------------------------------------------------------------------

def order_penalty(
    *, pitcher_id: int, pitches: pd.DataFrame, league_decline: float,
    minimum: int = 200,
) -> list[Finding]:
    """How much worse a starter gets the third time a lineup sees him.

    The league falls from a .239 strikeout rate on the first pass to .197 on the
    third. A pitcher who falls off faster than that is a different proposition
    late, and it is not visible in any season line.
    """
    if pitches.empty or "n_thruorder_pitcher" not in pitches.columns:
        return []

    own = pitches[
        (pitches["pitcher"] == int(pitcher_id))
        & pitches["events"].notna() & (pitches["events"] != "")
    ]
    if len(own) < minimum:
        return []

    struck = own["events"].isin({"strikeout", "strikeout_double_play"})
    by_pass = struck.groupby(own["n_thruorder_pitcher"]).agg(["mean", "size"])
    first = by_pass.loc[1] if 1 in by_pass.index else None
    later = by_pass.loc[3] if 3 in by_pass.index else (
        by_pass.loc[2] if 2 in by_pass.index else None
    )
    if first is None or later is None or first["size"] < 60 or later["size"] < 40:
        return []

    decline = float(first["mean"] - later["mean"])
    return [Finding(
        subject=int(pitcher_id),
        subject_kind="pitcher",
        code="pit.weak.order",
        family="durability",
        kind="weakness",
        # Signed so a steeper-than-league fall is negative.
        value=-(decline - league_decline),
        reference=Reference(mean=0.0, sd=0.05, population="league_order_decline"),
        evidence=int(later["size"]),
        stabilisation=120,
        detail={"first_pass": float(first["mean"]), "late_pass": float(later["mean"]),
                "decline": decline, "league_decline": float(league_decline)},
    )]


# ---------------------------------------------------------------------------
# 5. The starting matchup, at team level
# ---------------------------------------------------------------------------

def starter_edge(
    *, home_talent: float | None, away_talent: float | None,
    home_name: str, away_name: str, reference: Reference,
) -> list[Finding]:
    """Which club's starter is better, on the latent scale the models use."""
    if home_talent is None or away_talent is None:
        return []

    # Talent is a run value: lower is better for a pitcher, so the difference is
    # signed to make a home edge positive.
    edge = float(away_talent - home_talent)
    return [Finding(
        subject=0,
        subject_kind="game",
        code="game.starters",
        family="matchup",
        kind="matchup",
        value=edge,
        reference=reference,
        evidence=1,
        stabilisation=0,
        detail={"home": home_name, "away": away_name,
                "home_talent": float(home_talent), "away_talent": float(away_talent)},
    )]


# ---------------------------------------------------------------------------
# 6. Full arsenal — best and worst offering, with whiff and usage
# ---------------------------------------------------------------------------

# A swing that misses. Statcast records the outcome of every pitch, so a whiff
# rate is a count over swings rather than a leaderboard figure that only exists
# at season granularity.
SWINGS = frozenset({
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
    "hit_into_play", "foul_bunt", "missed_bunt",
})
MISSES = frozenset({"swinging_strike", "swinging_strike_blocked", "missed_bunt"})


def arsenal(
    *, pitcher_id: int, pitches: pd.DataFrame, league: pd.DataFrame,
    minimum: int = 150, per_pitch: int = 40,
) -> list[Finding]:
    """Every offering a pitcher throws often enough to judge.

    Emits one finding per pitch type rather than picking a winner, so selection
    and contrast detection can decide what is worth saying. A pitcher whose best
    and worst offerings are both extreme is a more interesting subject than one
    who is uniformly good, and that only becomes visible with both on the table.

    Judged within pitch type. A .280 expected wOBA on a slider is not the same
    achievement as on a four-seamer, and comparing across types would rank pitch
    types rather than pitchers.
    """
    if pitches.empty or "pitch_name" not in pitches.columns:
        return []

    thrown = pitches[pitches["pitcher"] == int(pitcher_id)]
    if len(thrown) < minimum:
        return []

    def summarise(frame: pd.DataFrame) -> pd.DataFrame:
        swings = frame["description"].isin(SWINGS)
        misses = frame["description"].isin(MISSES)
        return pd.DataFrame({
            "xwoba": frame.groupby("pitch_name")["estimated_woba_using_speedangle"].mean(),
            "n": frame.groupby("pitch_name").size(),
            "swings": swings.groupby(frame["pitch_name"]).sum(),
            "misses": misses.groupby(frame["pitch_name"]).sum(),
        })

    mine = summarise(thrown)
    theirs = summarise(league)
    mine = mine[mine["n"] >= per_pitch]
    if mine.empty:
        return []

    findings: list[Finding] = []
    for pitch_name, row in mine.iterrows():
        if pitch_name not in theirs.index:
            continue
        reference_row = theirs.loc[pitch_name]
        # Spread of the league's per-pitcher figures, not of individual pitches,
        # since the subject here is a pitcher rather than a pitch.
        spread = float(
            league[league["pitch_name"] == pitch_name]
            .groupby("pitcher")["estimated_woba_using_speedangle"].mean().std()
        ) or 0.05

        whiff = float(row["misses"] / row["swings"]) if row["swings"] else 0.0
        league_whiff = (
            float(reference_row["misses"] / reference_row["swings"])
            if reference_row["swings"] else 0.0
        )
        findings.append(Finding(
            subject=int(pitcher_id),
            subject_kind="pitcher",
            code="pit.arsenal.pitch",
            # Family per pitch type, so selection can pair a good one against a
            # bad one instead of treating the arsenal as a single trait.
            family=f"arsenal_{pitch_name}",
            kind="skill",
            # Lower expected wOBA is better, so negate to make positive good.
            value=-float(row["xwoba"]),
            reference=Reference(
                mean=-float(reference_row["xwoba"]), sd=spread,
                population=f"league_{pitch_name}", n=int(reference_row["n"]),
            ),
            evidence=int(row["n"]),
            stabilisation=120,
            inferential=False,
            detail={
                "pitch": str(pitch_name),
                "xwoba": float(row["xwoba"]),
                "league": float(reference_row["xwoba"]),
                "thrown": int(row["n"]),
                "usage": float(row["n"] / len(thrown)),
                "whiff": whiff,
                "league_whiff": league_whiff,
            },
        ))
    return findings


def no_track_record(
    *, player_id: int, evidence: int, side: str = "pitcher", minimum: int = 40,
) -> list[Finding]:
    """That there is nothing to say, said deliberately.

    A player with no recent record is not an average player, and filling the gap
    with a league prior would assert something nobody measured. The absence is
    itself the most useful thing on the line.
    """
    if evidence >= minimum:
        return []
    return [Finding(
        subject=int(player_id),
        subject_kind=side,
        code=f"{side[:3]}.absent",
        family="evidence",
        kind="weakness",
        value=0.0,
        reference=Reference(mean=0.0, sd=1.0, population="none"),
        evidence=0,
        stabilisation=1,
        inferential=False,
        detail={"seen": int(evidence)},
    )]
