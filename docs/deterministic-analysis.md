# Deterministic analysis — replacing the LLM layer

**Status:** design note, not scheduled. Filed 2026-08-20.

## The proposal

Generate the player and game commentary from computed criteria rather than from a
language model. Every sentence becomes the output of a threshold on a measured
quantity, so the same game produces the same report, forever, with no tokens
spent and nothing to verify after the fact.

This restores the project's founding constraint in the one place it was relaxed.
`analysis/` is currently the only layer permitted to reach a model, and
`test_no_llm_in_pipeline.py` exists to keep that boundary from spreading. If the
prose is computed, the boundary disappears and the test becomes an invariant over
the whole codebase.

## What already exists

`analysis/digest.py` is half of this. It already selects which players are worth
writing about (`select_subjects`) and assembles their numbers into a structured
record (`pitcher_digest`, `batter_digest`, `_windows`, `_situational`,
`_percentiles`). That work is model-agnostic — today the digest is serialised
into a prompt, but nothing about it requires a prompt.

What is missing is everything between "here are 200 numbers about this player"
and "here are the two sentences worth reading."

## The part that is actually hard

Templating is not the difficult half. Anyone can write

    f"{name} is hitting {avg} over his last {n} games."

and produce fifty-three of them. The difficulty is **selection**: out of roughly
forty computable criteria per player, knowing which two matter tonight. That is
the entire intellectual content of a scouting note, and it is what a language
model is currently doing implicitly.

Encoding it means being explicit about three things a model never has to state.

### 1. Significance is not magnitude

"Batting .312" is a number. "Batting .312 against an expected .240 given his
contact quality" is a finding. Every criterion needs a reference distribution and
a distance from it — a percentile or a z-score — not a raw value with a
hand-chosen cutoff.

Thresholds should come from the distribution, never from a constant. "Elite" is
the 90th percentile of this season's actual spread, not `> 0.300`, which drifts
as the run environment moves and has already made ERA cutoffs meaningless twice
in this project's corpus.

### 2. Scanning forty criteria and reporting the extreme reports noise

This is the trap, and it is severe enough to sink the whole idea if ignored.

Take a player with forty metrics, every one of them pure noise. The chance that
at least one lands beyond the 97.5th percentile is `1 - 0.95^40`, about **87%**.
Rank forty criteria and print the top two and you have built a machine that
manufactures a compelling, false observation about every player on the card.

The fix is the machinery already built for `talent.py`: shrink each metric toward
its population mean in proportion to how little evidence supports it, then rank
on the shrunk value. A .400 average over 20 at-bats shrinks to nothing; a .340
over 400 survives. Reliability has to enter the *ranking*, not merely be
mentioned in the sentence afterwards.

This also disposes of the small-sample findings that would otherwise dominate,
because extremes are always found in the smallest samples.

### 3. The interesting findings are contrasts, not values

Single-statistic findings are inert. What reads as insight is tension:

- elite exit velocity next to a bottom-decile launch angle
- a strikeout rate in the 90th percentile next to walk rate in the 15th
- excellent results against fastballs, helpless against anything slower
- strong overall line carried almost entirely by one month

None of these are visible to a per-metric scan; they are pairwise or conditional,
and each has to be written as its own evaluator. A useful rule of thumb: if a
finding could be inferred from the leaderboard the report already prints, it does
not need a sentence.

## Architecture

    digest (exists)  ->  evaluators  ->  findings  ->  scoring  ->  selection  ->  realization

**Finding** is a typed record, not a string:

```python
@dataclass
class Finding:
    subject: int              # player id, or team id for game-level
    kind: str                 # skill | weakness | trend | contrast | situational
    metrics: tuple[str, ...]
    value: float
    reference: float          # league mean, own baseline, or opponent-specific
    percentile: float
    evidence: int             # plate appearances, batters faced, chances
    shrunk_score: float       # significance AFTER reliability shrinkage
    template: str
```

**Evaluators** are pure functions from a digest to zero or more findings. They are
independently testable, which is the practical payoff: "does a player with this
line trigger the pull-heavy finding?" is a unit test, where the equivalent
question about a prompt is not.

**Selection** takes the top *k* under diversity constraints — no two findings of
the same `kind`, no two drawing on the same metric family, at most one trend. A
player who is excellent at everything should yield his two most *distinctive*
traits, not eight restatements of the same fact.

**Realization** fills templates. Needs several phrasings per finding type chosen
deterministically (hash the subject id, so the same player always reads the same
way) or the report acquires a detectable rhythm across fifty-three boxes.

## Mapping the criteria

| Criterion | Source | Notes |
|---|---|---|
| General value | `talent.batter_score` / `pitcher_score` | one number, already opponent-adjusted |
| Trending, and for how long | rolling windows vs own baseline | see below |
| Performance in this series | series logs | exists |
| Value source: hit / field / run | batting runs, OAA→runs, BsR | all on the same run scale, so directly comparable |
| Batted-ball profile | pitch corpus: `launch_speed`, `launch_angle`, `bb_type`, `hc_x/hc_y` | slugger vs contact, pull vs oppo, LD vs FB |
| What a batter looks for | pitch corpus: swing rate and xwOBA by pitch type × zone | needs the corpus; not derivable from leaderboards |
| Athleticism | sprint speed, bat speed, arm | leaderboards already pulled |
| Pitcher repertoire | pitch corpus: mix %, velocity, movement | as-of, unlike the season leaderboard |
| Pitcher strengths | components vs league, by pitch type | |
| Where a pitcher gets beaten | xwOBA by pitch type, zone, count, times through order | `n_thruorder_pitcher` is in the corpus |
| Game: better on paper | Elo | |
| Game: better lately | recent form | descriptive only — measured collinear with Elo, so it must not be presented as adding predictive information |
| Game: key players | finding scores, ranked across both clubs | falls out of the same machinery |

Much of this becomes available only because of the pitch corpus now downloading.
The batter's approach, the pitcher's repertoire and where he gets beaten are all
pitch-level questions, and the season leaderboards cannot answer them as-of a
date — verified: they ignore `start_date` and `end_date` entirely.

## "Trending up — for how long?"

Worth separating out, because the obvious implementation is wrong. Picking a
fixed window and comparing it to season-to-date invites the same multiple-
comparisons problem through the back door: try L5, L10, L15 and L30 and one of
them will look dramatic.

The honest version is change-point detection — find the longest recent window
whose mean differs from the player's baseline by more than sampling noise, and
report *that* window. It answers the question actually asked, and when no window
qualifies it returns nothing, which is the correct answer most of the time.

## Risks

**It will read as generated unless selection is genuinely good.** The failure
mode is not wrong sentences, it is fifty-three correct, interchangeable ones.
This is the whole risk and no amount of template variety fixes it.

**Hand-set thresholds reintroduce arbitrary weights** — the same objection that
sank the hand-weighted composite index. Everything must be distributional.

**Scope.** Comparable to the projection engine: perhaps 25–40 evaluators, each
needing its own reference distribution and test. Not a weekend.

**Loss of range.** A model occasionally makes a connection no evaluator was
written for. That is the real cost, and it is worth paying for output that is
identical every run and checkable line by line — but it is a cost, not a free
win.

## Suggested sequencing

1. Build the `Finding` type, scoring and selection against **five** evaluators only.
2. Run both layers side by side on the same game and compare honestly.
3. Expand evaluator coverage if and only if the five read well.
4. Retire `analysis/agent.py`, and extend `test_no_llm_in_pipeline.py` to the
   whole package.

Step 2 is the decision point. If five well-chosen deterministic findings do not
beat the model's paragraph, more evaluators will not close the gap.
