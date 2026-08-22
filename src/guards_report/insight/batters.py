"""Batter evaluators.

Pitchers were the easy half: an arsenal is a small set of named things, each
with an obvious comparison. A hitter has no equivalent, which is why the model
reaches for platoon splits and percentile rankings when describing one -- there
is no single frame that says what kind of hitter he is.

So this covers four different frames, and lets selection decide which is worth
saying tonight:

* **what he does** -- reaching, power, strikeouts, against the league
* **how he does it** -- the batted-ball profile, which separates a slugger from
  a contact hitter even when their rates look alike
* **what he is doing lately** -- a change-point search rather than a fixed window
* **what he cannot handle** -- the pitch type and the platoon side that beat him

Most of these are descriptive rather than inferential. "Pulls 47% of his fly
balls" is a fact about a hitter, not a claim that 47% is remarkable, so it is not
gated behind a multiple-comparisons floor -- the distinction Phase B found the
hard way.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from guards_report.insight.evaluators import POLARITY
from guards_report.insight.types import Finding, Reference

# Statcast's contact classification. Six is a barrel: the exit-velocity and
# launch-angle combination that has historically produced at least a .500
# average and 1.500 slugging.
BARREL = 6.0

SWINGS = frozenset({
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
    "hit_into_play", "foul_bunt", "missed_bunt",
})
MISSES = frozenset({"swinging_strike", "swinging_strike_blocked", "missed_bunt"})

# Zones 11 through 14 are outside the strike zone; 1 through 9 are inside it.
OUT_OF_ZONE = frozenset({11.0, 12.0, 13.0, 14.0})


def _finding(
    *, batter: int, code: str, family: str, kind: str, value: float,
    reference: Reference, evidence: int, stabilisation: int,
    detail: dict, inferential: bool = False,
) -> Finding:
    return Finding(
        subject=int(batter), subject_kind="batter", code=code, family=family,
        kind=kind, value=float(value), reference=reference,
        evidence=int(evidence), stabilisation=int(stabilisation),
        inferential=inferential, detail=detail,
    )


def _ref(series: pd.Series, population: str) -> Reference:
    clean = series.dropna()
    if len(clean) < 2:
        return Reference(mean=0.0, sd=1.0, population=population, n=len(clean))
    return Reference(
        mean=float(clean.mean()), sd=float(clean.std()) or 1.0,
        population=population, n=len(clean),
    )


def batted_ball_profile(
    *, batter_id: int, plate: pd.DataFrame, league_profile: pd.DataFrame,
    minimum: int = 60,
) -> list[Finding]:
    """How he makes contact — the frame that separates two hitters with one line.

    A slugger and a contact hitter can carry the same batting average and be
    completely different problems. Ground-ball rate, barrel rate and average
    exit velocity say which is which, and none of them are in a rate line.
    """
    own = plate[plate["batter"] == int(batter_id)]
    contact = own[own["bb_type"].notna() & (own["bb_type"] != "")]
    if len(contact) < minimum:
        return []

    findings: list[Finding] = []

    shares = contact["bb_type"].value_counts(normalize=True)
    for bb_type, family, label in (
        ("ground_ball", "trajectory", "ground balls"),
        ("fly_ball", "trajectory", "fly balls"),
        ("line_drive", "trajectory", "line drives"),
    ):
        if bb_type not in shares or f"{bb_type}_share" not in league_profile:
            continue
        findings.append(_finding(
            batter=batter_id, code="bat.profile.trajectory", family=family,
            kind="skill", value=float(shares[bb_type]),
            reference=_ref(league_profile[f"{bb_type}_share"], "qualified_batters"),
            evidence=len(contact), stabilisation=120,
            detail={"share": float(shares[bb_type]), "label": label},
        ))

    barrels = contact["launch_speed_angle"].eq(BARREL)
    if barrels.notna().sum() >= minimum:
        findings.append(_finding(
            batter=batter_id, code="bat.profile.barrel", family="power",
            kind="skill", value=float(barrels.mean()),
            reference=_ref(league_profile["barrel_rate"], "qualified_batters"),
            evidence=len(contact), stabilisation=150,
            detail={"rate": float(barrels.mean())},
        ))

    exit_velocity = contact["launch_speed"].dropna()
    if len(exit_velocity) >= minimum:
        findings.append(_finding(
            batter=batter_id, code="bat.profile.exit_velocity", family="contact_quality",
            kind="skill", value=float(exit_velocity.mean()),
            reference=_ref(league_profile["exit_velocity"], "qualified_batters"),
            evidence=len(exit_velocity), stabilisation=80,
            detail={"velocity": float(exit_velocity.mean())},
        ))

    return findings


def spray_tendency(
    *, batter_id: int, plate: pd.DataFrame, league_pull: pd.Series,
    minimum: int = 50,
) -> list[Finding]:
    """Whether he pulls the ball, which decides where a defence stands.

    Spray angle is computed from the landing coordinates and signed by
    handedness, so "pull" means the same thing for a left-handed hitter as a
    right-handed one rather than meaning "toward left field".
    """
    own = plate[(plate["batter"] == int(batter_id)) & plate["hc_x"].notna()]
    if len(own) < minimum:
        return []

    # Statcast's coordinate origin sits behind the plate; this converts to an
    # angle where negative is toward left field.
    angle = np.degrees(np.arctan2(
        own["hc_x"].to_numpy() - 125.42,
        198.27 - own["hc_y"].to_numpy(),
    ))
    stand = own["stand"].fillna("R").to_numpy()
    # A right-handed hitter pulls to the left; flip so positive is always pull.
    pull_side = np.where(stand == "R", -angle, angle)
    pull_rate = float((pull_side > 15).mean())

    return [_finding(
        batter=batter_id, code="bat.profile.spray", family="spray",
        kind="skill", value=pull_rate,
        reference=_ref(league_pull, "qualified_batters"),
        evidence=len(own), stabilisation=100,
        detail={"pull": pull_rate},
    )]


def plate_discipline(
    *, batter_id: int, plate: pd.DataFrame, league_chase: pd.Series,
    league_whiff: pd.Series, minimum: int = 200,
) -> list[Finding]:
    """Whether he swings at pitches he should not, and whether he connects.

    Chase rate is the clearest single read on approach and it stabilises fast,
    so it survives shrinkage where a batting average would not.
    """
    own = plate[plate["batter"] == int(batter_id)]
    if len(own) < minimum:
        return []

    outside = own[own["zone"].isin(OUT_OF_ZONE)]
    findings: list[Finding] = []
    if len(outside) >= 60:
        chased = outside["description"].isin(SWINGS)
        findings.append(_finding(
            batter=batter_id, code="bat.approach.chase", family="discipline",
            kind="skill",
            # Chasing is bad, so negate to keep positive meaning better.
            value=-float(chased.mean()),
            reference=Reference(
                mean=-float(league_chase.mean()), sd=float(league_chase.std()) or 0.05,
                population="qualified_batters", n=len(league_chase),
            ),
            evidence=len(outside), stabilisation=80,
            detail={"chase": float(chased.mean())},
        ))

    swings = own[own["description"].isin(SWINGS)]
    if len(swings) >= 100:
        missed = swings["description"].isin(MISSES)
        findings.append(_finding(
            batter=batter_id, code="bat.approach.whiff", family="whiff",
            kind="skill", value=-float(missed.mean()),
            reference=Reference(
                mean=-float(league_whiff.mean()), sd=float(league_whiff.std()) or 0.05,
                population="qualified_batters", n=len(league_whiff),
            ),
            evidence=len(swings), stabilisation=100,
            detail={"whiff": float(missed.mean())},
        ))
    return findings


def pitch_type_weakness(
    *, batter_id: int, plate: pd.DataFrame, league: pd.DataFrame,
    minimum: int = 90,
) -> list[Finding]:
    """What he handles and what beats him, by pitch type.

    The mirror of a pitcher's arsenal, and the finding a scouting report exists
    to surface: knowing a hitter cannot cover a breaking ball is what a pitcher
    does something with.
    """
    own = plate[
        (plate["batter"] == int(batter_id))
        & plate["pitch_name"].notna() & (plate["pitch_name"] != "")
    ]
    if own.empty:
        return []

    league_by_type = league.groupby("pitch_name")["estimated_woba_using_speedangle"].agg(
        ["mean", "std", "size"]
    )
    mine = own.groupby("pitch_name").agg(
        xwoba=("estimated_woba_using_speedangle", "mean"),
        n=("estimated_woba_using_speedangle", "size"),
    )
    mine = mine[mine["n"] >= minimum]

    findings: list[Finding] = []
    for pitch_name, row in mine.iterrows():
        if pitch_name not in league_by_type.index:
            continue
        reference = league_by_type.loc[pitch_name]
        findings.append(_finding(
            batter=batter_id, code="bat.approach.pitchtype",
            family=f"vs_{pitch_name}", kind="skill",
            value=float(row["xwoba"]),
            reference=Reference(
                mean=float(reference["mean"]), sd=float(reference["std"]) or 0.05,
                population=f"league_{pitch_name}", n=int(reference["size"]),
            ),
            evidence=int(row["n"]), stabilisation=120,
            detail={"pitch": str(pitch_name), "xwoba": float(row["xwoba"]),
                    "league": float(reference["mean"]), "seen": int(row["n"])},
        ))
    return findings


def platoon_split(
    *, batter_id: int, plate: pd.DataFrame, league_gap: pd.Series,
    minimum: int = 80,
) -> list[Finding]:
    """A gap between what he does to left-handers and right-handers.

    Inferential, unlike most of this file: claiming a hitter has a platoon
    problem is a claim about a difference, and differences over small samples are
    where false findings come from.
    """
    own = plate[plate["batter"] == int(batter_id)]
    ended = own[own["events"].notna() & (own["events"] != "")]
    if len(ended) < minimum:
        return []

    by_hand = ended.groupby(ended["p_throws"].fillna("R")).agg(
        xwoba=("estimated_woba_using_speedangle", "mean"),
        n=("estimated_woba_using_speedangle", "size"),
    )
    if not {"L", "R"}.issubset(set(by_hand.index)):
        return []
    if by_hand.loc["L", "n"] < 30 or by_hand.loc["R", "n"] < 30:
        return []

    gap = float(by_hand.loc["R", "xwoba"] - by_hand.loc["L", "xwoba"])
    return [_finding(
        batter=batter_id, code="bat.split.platoon", family="platoon",
        kind="contrast", value=abs(gap),
        reference=Reference(
            mean=float(league_gap.abs().mean()), sd=float(league_gap.std()) or 0.05,
            population="qualified_batters", n=len(league_gap),
        ),
        evidence=int(min(by_hand.loc["L", "n"], by_hand.loc["R", "n"])),
        stabilisation=120, inferential=True,
        detail={"vs_left": float(by_hand.loc["L", "xwoba"]),
                "vs_right": float(by_hand.loc["R", "xwoba"]),
                "stronger": "right-handers" if gap > 0 else "left-handers",
                "gap": abs(gap)},
    )]


def results_versus_contact(
    *, batter_id: int, plate: pd.DataFrame, league_gap: pd.Series,
    minimum: int = 120,
) -> list[Finding]:
    """The gap between what he has earned and what he has got.

    Expected wOBA scores a batted ball by how balls hit that hard at that angle
    usually do; actual wOBA scores what happened to this one. A wide gap is the
    most useful thing a scouting report can say about a hitter whose line looks
    settled, because it is the part most likely to move.
    """
    own = plate[plate["batter"] == int(batter_id)]
    ended = own[own["events"].notna() & (own["events"] != "")]
    ended = ended[ended["estimated_woba_using_speedangle"].notna()]
    if len(ended) < minimum:
        return []

    actual = float(ended["woba_value"].mean())
    expected = float(ended["estimated_woba_using_speedangle"].mean())
    gap = actual - expected

    return [_finding(
        batter=batter_id, code="bat.luck.gap", family="regression",
        kind="trend", value=gap,
        reference=Reference(
            mean=float(league_gap.mean()), sd=float(league_gap.std()) or 0.03,
            population="qualified_batters", n=len(league_gap),
        ),
        evidence=len(ended), stabilisation=200, inferential=True,
        detail={"actual": actual, "expected": expected, "gap": gap,
                "direction": "outrunning" if gap > 0 else "behind"},
    )]


def league_profile(plate: pd.DataFrame, *, minimum: int = 150) -> pd.DataFrame:
    """Per-batter reference figures, so every comparison has a real population.

    Rebuilt from the season being played rather than fixed, because a constant
    threshold goes stale as the run environment moves -- which has already
    happened twice in this project.
    """
    ended = plate[plate["events"].notna() & (plate["events"] != "")]
    counts = ended.groupby("batter").size()
    qualified = counts[counts >= minimum].index

    own = plate[plate["batter"].isin(qualified)]
    contact = own[own["bb_type"].notna() & (own["bb_type"] != "")]

    # Cross-tabulate then divide by the row total. Normalising inside a grouped
    # apply leaves a duplicated index level that will not align back.
    counts = (
        contact.groupby(["batter", "bb_type"]).size().unstack(fill_value=0)
    )
    shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    shares.columns = [f"{c}_share" for c in shares.columns]

    frame = shares.copy()
    frame["barrel_rate"] = (
        contact.assign(barrel=contact["launch_speed_angle"].eq(BARREL))
        .groupby("batter")["barrel"].mean()
    )
    frame["exit_velocity"] = contact.groupby("batter")["launch_speed"].mean()

    swings = own[own["description"].isin(SWINGS)]
    frame["whiff"] = (
        swings.assign(missed=swings["description"].isin(MISSES))
        .groupby("batter")["missed"].mean()
    )
    outside = own[own["zone"].isin(OUT_OF_ZONE)]
    frame["chase"] = (
        outside.assign(swung=outside["description"].isin(SWINGS))
        .groupby("batter")["swung"].mean()
    )

    landed = own[own["hc_x"].notna()]
    if len(landed):
        angle = np.degrees(np.arctan2(
            landed["hc_x"].to_numpy() - 125.42, 198.27 - landed["hc_y"].to_numpy()
        ))
        stand = landed["stand"].fillna("R").to_numpy()
        landed = landed.assign(pull=np.where(stand == "R", -angle, angle) > 15)
        frame["pull"] = landed.groupby("batter")["pull"].mean()

    ended_q = ended[ended["batter"].isin(qualified)]
    ended_q = ended_q[ended_q["estimated_woba_using_speedangle"].notna()]
    frame["luck_gap"] = (
        ended_q.groupby("batter")["woba_value"].mean()
        - ended_q.groupby("batter")["estimated_woba_using_speedangle"].mean()
    )

    by_hand = ended_q.groupby(["batter", ended_q["p_throws"].fillna("R")])[
        "estimated_woba_using_speedangle"
    ].mean().unstack()
    if {"L", "R"}.issubset(set(by_hand.columns)):
        frame["platoon_gap"] = by_hand["R"] - by_hand["L"]

    return frame
