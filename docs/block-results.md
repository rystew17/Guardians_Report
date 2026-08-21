# Feature block results — game outcome

Every block tested against the Core block on a walk-forward across 2018–2026,
19,822 held-out games. Recorded whatever the answer was.

## Incumbent

**Core block**, 9 features: `elo_logit`, `sp_fip`, `sp_k_pct`, `sp_bb_pct`,
`sp_ip_per_start`, `starter_known`, `team_rest_diff`, `sp_rest_diff`,
`park_factor`.

Pooled held-out log loss **0.67543**, accuracy **0.5758**. Elo alone: 0.67836.

## Results

| Block | Log loss | Δ | Seasons | Verdict |
|---|---|---|---|---|
| Core (incumbent) | 0.67543 | — | — | — |
| + starter talent (PA ridge) | 0.67548 | +0.00004 | 5/9 | **fails** |
| + career-to-date starter line | 0.67538 | −0.00005 | 5/9 | **fails** |
| + lineup value | 0.67504 | −0.00039 | 6/9 | **fails** (p = 0.13–0.19) |
| + lineup + talent | 0.67517 | −0.00026 | 3/9 | **fails** |
| Elo + talent, no season line | 0.67614 | +0.00071 | — | **fails** |

Earlier blocks, same protocol: ERA (p = 0.308 beside FIP), RA/9 (wrong sign,
survived orthogonalisation), bullpen aggregate (p = 0.614), bullpen availability
(p = 0.141), offence/defence ratings, run-Elo, recent form (partial r +0.003
against SE 0.006), team defence proxy.

**Nine blocks tested. None has beaten the Core block.**

## Why

Everything correlates with Elo, because Elo is built from the outcomes those
things cause:

| Feature | r with `elo_logit` |
|---|---|
| Lineup value differential | **+0.6513** |
| L30 run differential | +0.7789 |
| Starter talent differential | +0.4353 |

A club's lineup quality is most of what makes it win, and Elo is a summary of
winning. By the time a hitter is good enough to matter, his club's rating has
already recorded it.

## The methodological finding

**Partial correlation is a screening test, not a block test.**

Used to *kill* a block it is valid: a feature with no partial correlation cannot
help. Used to *promote* one it is misleading.

| Block | Partial r vs Core | Block test |
|---|---|---|
| Lineup value | +0.0299 (**4.4 SE**) | fails, p = 0.13 |
| Starter talent, full corpus | +0.0209 (2.9 SE) | fails |
| Starter talent, 2016 only | +0.0759 (3.4 SE) | overstated 3.6× |

Partial correlation holds the incumbent coefficients fixed. The block test
refits them, and the incumbents absorb the new information by re-weighting. A
feature can carry real signal *at the current weights* and still be redundant
once the model is allowed to adjust.

The 2016 mini-test also overstated by 3.6×. One season of prior against one of
test settles that machinery runs; it settles nothing about magnitude.

## Where the evidence pointed, and why that was a clue

Starter talent's partial correlation by evidence available at first pitch:

| Evidence | n | Partial r |
|---|---|---|
| <100 PA | 4,567 | +0.0604 (4.1 SE) |
| 100–300 PA | 9,429 | +0.0102 (1.0 SE) |
| 300+ PA | 5,696 | −0.0020 (0.2 SE) |

Backwards. More evidence should mean a better estimate and more signal. All of
it sitting in the thinnest samples meant the talent model was not contributing
talent — it was contributing *prior-season information*, which the Core block
lacks because `as_of_table` groups by `(pitcher, season)`.

Testing that directly with a career-to-date line: also fails. So it was not the
wrong vehicle for prior-season information; prior-season information does not
help either.

## Score model

Directionally different. Held-out negative log likelihood per team-game:

| Variant | Value | Δ |
|---|---|---|
| Model B (incumbent) | 2.45623 | — |
| + opposing starter talent | 2.45602 | −0.00020 |
| + own lineup value | 2.45539 | −0.00084 |
| + both | 2.45516 | **−0.00107** |

Larger than anything the win model showed, and **not yet significance-tested** —
which is exactly the mistake made twice above. Test before believing.

## Conclusion

The win model is at a practical ceiling for team-level features. The remaining
gap to a book is not a feature we have failed to engineer; it is injury
information and a market aggregating private knowledge.

The pitch corpus should be pointed where Elo is not the incumbent: the score
model, and projections 3–5 (player hit, player home run, starter strikeouts),
which are plate-appearance questions with no rating already summarising them.

---

# Improving Elo rather than competing with it

The framing above — every block measured against Elo as an incumbent — misses an
option. Elo updates on actual runs, which carry sequencing luck (whether hits
arrived with runners on) and fielding luck (whether hard contact found a glove).
Expected wOBA removes both. Feeding the rating a cleaner signal should let it
converge faster, and it costs no new model.

Expected runs per team-game are built from `estimated_woba_using_speedangle` on
contact and the actual value otherwise — a walk is a walk, and substituting an
expectation there adds noise rather than removing it. Rescaled so the league mean
matches actual runs per team-game.

| | sd |
|---|---|
| Actual margin | 4.490 |
| Expected margin | 1.796 |
| Correlation between them | 0.758 |

## Result

Held-out 2018–2026, K re-tuned separately for each signal so the comparison is
like for like (K fitted for a margin with three times the spread is simply the
wrong step size).

| Weight on expected margin | Held-out log loss | Seasons better | Clustered p |
|---|---|---|---|
| 0.00 — pure actual | 0.67680 | — | — |
| 0.15 | 0.67672 | 8/9 | 0.006 |
| **0.30** | **0.67668** | **8/9** | **0.040** |
| 0.50 | 0.67678 | 6/9 | 0.823 |
| 0.70 | 0.67704 | 5/9 | 0.241 |
| 1.00 — pure expected | 0.67740 | 2/9 | 0.058 |

**A blend beats pure actual and pure expected loses.** That shape is the finding.

## Why pure expected fails

Expected runs here are built from the *batting* side's contact quality, so they
measure offence alone. Actual runs are offence minus defence. Replacing the
margin entirely discards the defensive half in order to remove sequencing noise,
and the defensive half is worth more. A partial weight keeps both.

## Why it is not adopted

The gain at the best weight is **−0.00012 log loss**. The Core block beats Elo by
0.00293, so this is roughly four percent of that, and it is smaller than the
season-to-season variation in every measurement on this page.

Against it: the Elo path would gain a dependency on the pitch corpus, and the
in-season refresh — currently one schedule request and some arithmetic — would
have to compute expected runs for every new game before it could update a rating.
That is a real complication of the one path that must run for every report.

Recorded as validated in direction and not worth shipping. If the expected-runs
table ever becomes a dependency of the refresh path for another reason, revisit:
the improvement is free at that point.

## What would be worth trying

The failure of pure expected points at the fix. An expected margin that included
the *fielding* side — expected runs allowed, not just expected runs scored —
would not discard defence, and might justify a larger weight. That needs a
defensive model this project does not have, and the flagged team-defence block
was itself measured as collinear with Elo, so the prospect is not encouraging.
