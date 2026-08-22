# Phase B — both layers on the same pitchers

Run against six San Francisco pitchers from game 824394, using the stored
analysis from an earlier run so the comparison cost no tokens.

## The verdict

**Continue.** The computed layer matches the model on the figures they share and
adds a reference the model omits, but covers less ground. That is an evaluator
problem, not a method problem.

## Side by side

**Hentges** — the clearest case.

> **Model:** four-seamer is his best pitch by a wide margin — a 30.6% whiff rate
> and a .252 xwOBA — and it's the pitch he leans on most at 30.9% usage.

> **Computed:** Hentges's 4-Seam Fastball is the out pitch: a 24.3% whiff rate
> holding hitters to a .252 expected wOBA, against .338 on the pitch
> league-wide, but the Slider is where he gets hurt: .373 expected against, .290
> league.

Same pitch, same .252 to three decimals. The computed line adds the league
baseline the model leaves out, and a weakness it never reaches.

**Brubaker** — changeup .257 in both. **Seymour** — both identify the four-seamer
as the problem, .505 against .548, usage 21% against 19.7%. **Smith** — sweeper
.199 in both. **Wilkinson** — both report that there is nothing to report.

The overlapping figures agree, which is the useful result: two independent paths
to the same numbers.

## What the model still does better

Breadth. It reaches platoon splits, walk-rate trends and percentile rankings
that no evaluator computes yet, and it moves between them naturally. Every one
is computable from data already on disk; none is written.

It also chooses better among equals. Given four negative findings for Seymour it
would likely have led with his one strength; the computed line reports two
weaknesses because that is what the ranking says.

## The design error Phase B found

The first run produced **nothing for all six pitchers**. Every finding was gated
behind the multiple-comparisons floor, and none cleared it.

The floor was wrong for this kind of finding. "His best pitch is the sweeper,
.199 against a .300 league" is *descriptive*: it is true whatever the league
spread and claims nothing about being unusual. "He is trending up" is
*inferential* — it asserts a departure from the population, and scanning forty
criteria for the most extreme one is exactly how that becomes false.

Applying an inferential guard to descriptive findings silenced the report.
`Finding.inferential` now separates them, and the floor applies only to the
second kind. This is the distinction the plan missed entirely, and only running
both layers side by side surfaced it.

## Smaller faults, fixed

- Selection took two findings of the same direction. It now leads with the
  strongest and deliberately reaches for one pointing the other way, because a
  strength next to a weakness reads as a scouting note where two strengths read
  as a list.
- Rates printed as `0.252`. Baseball writes `.252`.
- The same template fired twice in one sentence, reading as a stutter.
- "Smith his 4-Seam Fastball" — templates written to stand alone begin with a
  possessive, which needs turning into the subject's own.

## What remains before this can replace the model

1. **More evaluators.** Six pitchers exposed the gap clearly: platoon splits,
   command trends, contact quality, batted-ball profile. Roughly thirty of the
   forty-four catalogued remain.
2. **Batters.** Only pitchers were compared. On the earlier five-hitter run one
   in five cleared the floor, and with the descriptive/inferential split now in
   place that number should be re-measured rather than assumed.
3. **A verdict on breadth.** The model's advantage is that it always finds
   *something* to say. Whether that is a strength or a weakness is the question
   this project keeps returning to, and it should be settled by reading a full
   card rather than six lines.


---

# Batters built — coverage 49% to 85%

The gap at the last measurement was not quality but absence: pitchers were at
100% and batters at 0%, because no batter evaluator was wired.

`insight/batters.py` adds six, covering four different frames, because a hitter
has no equivalent of an arsenal -- no single small set of named things with an
obvious comparison. That is why the model reaches for percentile rankings when
describing one.

| Subject | n | Covered | |
|---|---|---|---|
| Relievers | 24 | 23 | 96% |
| Batters | 18 | 16 | 89% |
| Bench | 8 | 5 | 62% |
| Starters | 2 | 1 | 50% |
| Matchup | 1 | 0 | 0% |
| **Total** | **53** | **45** | **85%** |

## Reading against the model

> **Model:** Lee's calling card is bat-to-ball skill — a 96th percentile xBA
> paired with a strikeout rate of just 11.1%. What he doesn't do is damage the
> ball: an 11th percentile hard-hit rate.

> **Computed:** Lee misses on 11.2% of his swings, league 22.6%, but squares one
> up 2.5% of the time he makes contact, 7.9% league-wide.

Same hitter, same two facts, arrived at independently. Elsewhere: Eldridge at
92.2 mph against a league 88.1 where the model said "89th percentile exit
velocity"; Bericoto's chase rate and breaking-ball trouble, which the model also
led with.

## A fault the run exposed

One line read "Cox has handled the 4-Seam Fastball at a .053 expected wOBA".
True, and meaningless -- forty pitches.

Shrinkage was already working, but it governs *ranking* while the sentence
prints the raw figure, which is correct for a descriptive finding and
misleading on thin evidence. Selection now applies a reliability gate as well as
the significance floor: two different guards for two different failures. The
floor stops an inferential claim that is really noise; the gate stops a
descriptive one that is technically true and says nothing.

Cox now returns nothing, which is the honest answer for a bench player with
forty pitches on record.

## What is left

- **The matchup page**, still 0%. It is one subject and the projections page
  already computes most of what it needs.
- **The two starters at 50%.** One had too few pitches under the arsenal
  threshold; worth checking whether that threshold is right rather than assuming.
- **Breadth of judgement.** The model still moves between frames more naturally
  and reaches counts, walk rates and percentile context that no evaluator
  computes. Every one is computable; none is written.
