# Deterministic analysis — implementation plan

**Status:** planned, not started. Scheduled after the pitch pull completes and the
talent/lineup blocks are resolved.
**Design rationale:** [deterministic-analysis.md](deterministic-analysis.md)

---

## 0. Goal

Replace `analysis/agent.py` with computed prose. Success is a report where every
sentence is the output of a threshold on a measured quantity, the same game
yields byte-identical output forever, and `test_no_llm_in_pipeline.py` covers the
entire package rather than stopping at `analysis/`.

**Non-goal:** matching a language model's range. It occasionally connects
something no evaluator was written for. That is a real loss, accepted knowingly.

---

## 1. What exists

| Module | Lines | Role after this work |
|---|---|---|
| `metrics/highlights.py` | 237 | **The prototype.** Already produces deterministic game-level items with a `Highlight` dataclass and `build()`. Generalise it; do not start from scratch. |
| `analysis/digest.py` | 456 | Subject selection and per-player structuring. Keep entirely — nothing in it needs a prompt. |
| `analysis/agent.py` | 426 | Deleted at the end. |
| `analysis/store.py` | 147 | Keep; findings still want persisting for the audit trail. |
| `analysis/verify.py` | 143 | Repurpose: it currently checks model output against source figures, which becomes a self-consistency check on findings. |

`highlights.py` also shows the pattern to *avoid*: `MIN_PA_SEASON = 120` and
friends are hard cutoffs, which throw away a real finding at 119 plate
appearances and accept a noisy one at 121. Section 3 replaces them.

---

## 2. Core types

New package `insight/`, sibling to `analysis/`.

```python
# insight/types.py

@dataclass(frozen=True)
class Reference:
    """What a value is being compared against."""
    mean: float
    sd: float
    population: str          # "league_batters_2026", "own_baseline", "vs_RHP"
    n_population: int


@dataclass(frozen=True)
class Finding:
    subject: int                     # player id; team id for game-level
    subject_kind: str                # batter | pitcher | team | game
    code: str                        # stable id, e.g. "bat.power.barrel"
    family: str                      # power | contact | discipline | speed | ...
    kind: str                        # skill | weakness | trend | contrast | matchup

    metrics: tuple[str, ...]
    value: float
    reference: Reference
    evidence: int                    # PA, BF, chances

    z_raw: float
    reliability: float               # 0..1, from the metric's stabilisation point
    z_shrunk: float                  # z_raw * reliability  <- ranked on this

    window: tuple[date, date] | None = None   # for trend findings
    detail: dict = field(default_factory=dict)

    @property
    def significance(self) -> float:
        return abs(self.z_shrunk)


Evaluator = Callable[[Digest, Context], list[Finding]]
```

`Context` carries the opposing starter, park, league reference tables and the
as-of date, so matchup evaluators need no globals.

---

## 3. Significance scoring

The whole design rests on this. Three steps, no hand-set thresholds anywhere.

### 3.1 Reference distributions

Every metric is scored against a *measured* population, rebuilt each season from
the corpus and cached beside the model artifact:

```python
Reference(mean=..., sd=..., population="league_qualified_batters_2026", n=...)
```

Never a constant. `> .300` was a meaningful batting average in 2015 and is not
in 2026, and this project has already watched fixed ERA cutoffs go stale twice.

### 3.2 Reliability, measured not guessed

For each metric, find the stabilisation point `k` — the sample size at which
split-half reliability reaches 0.5 — using the method already applied to
strikeout rate (split-half r = 0.6137 on 1,999 pitcher-seasons, Spearman-Brown
0.7606).

```python
reliability = n / (n + k_metric)
z_shrunk    = z_raw * reliability
```

Approximate expectations, to be measured rather than assumed:

| Metric | Stabilises around |
|---|---|
| Swing / chase rate | 50 PA |
| Strikeout rate | 60 PA |
| Walk rate | 120 PA |
| Exit velocity | 40 batted balls |
| Barrel rate | 150 PA |
| ISO | 160 PA |
| BABIP | 800 PA |
| Fielding OAA | ~1 season |

BABIP at 800 is the point of the exercise: it is the metric most likely to
produce a dramatic-looking finding and the one least likely to mean anything.

### 3.3 Multiple-comparisons floor

With `C` criteria scanned per player, require

```
|z_shrunk| > z_crit,  where  C * 2 * (1 - Phi(z_crit)) < 0.5
```

so fewer than one finding in two players is expected by chance. At `C = 40` that
is `z_crit ≈ 2.50`. **Recompute whenever the evaluator count changes** — adding
evaluators without raising the bar silently increases the false-finding rate,
which is exactly how this fails while looking like it improved.

Without this, 40 pure-noise metrics give an 87% chance that a player receives a
compelling and untrue observation.

---

## 4. Selection

```python
def select(findings, *, k=3, per_family=1) -> list[Finding]:
```

1. Drop anything below `z_crit`.
2. Sort by `significance`.
3. Greedily take, rejecting a candidate that shares `family` with one already
   chosen, or `kind` with two already chosen.
4. Stop at `k`, or earlier — **returning one finding, or none, is a valid
   outcome** and is the correct answer for a replacement-level player having an
   unremarkable season. A model always writes a paragraph; this should not.

The diversity constraint is what prevents an elite player yielding eight
restatements of "he is good."

---

## 5. Realization

```python
def render(finding: Finding, *, variant: int) -> str
```

Each `code` carries 3–5 phrasings. Variant chosen by `hash(subject) % len` so a
given player always reads the same way — deterministic, and it prevents the
detectable rhythm that fifty-three identically-shaped sentences would create.

Templates state the number *and* its reference: not "elite barrel rate" but
"barrels 14.2% of batted balls, seventh-highest among qualified hitters."

---

## 6. Evaluator catalogue

44 evaluators. `[P]` needs the pitch corpus; `[T]` needs the talent model.

### Batters — value and skill (7)

| Code | Trigger | Source |
|---|---|---|
| `bat.value.overall` | talent score percentile | `[T]` |
| `bat.value.source` | which of hit/field/run dominates run contribution | runs on a common scale |
| `bat.power.barrel` | barrel rate vs league | `[P]` |
| `bat.contact.k` | strikeout rate vs league | |
| `bat.discipline.chase` | chase rate vs league | `[P]` |
| `bat.speed.sprint` | sprint speed percentile | leaderboard |
| `bat.field.oaa` | OAA vs positional average | leaderboard |

### Batters — batted-ball profile (5)

| Code | Trigger | Source |
|---|---|---|
| `bat.profile.archetype` | barrel% × K% quadrant → slugger / contact / balanced | `[P]` |
| `bat.profile.spray` | pull vs opposite tendency from `hc_x` | `[P]` |
| `bat.profile.launch` | GB/LD/FB mix vs league | `[P]` |
| `bat.profile.ev` | average and 90th-percentile exit velocity | `[P]` |
| `bat.profile.luck` | xwOBA − wOBA gap, signed | `[P]` |

### Batters — approach (4)

| Code | Trigger | Source |
|---|---|---|
| `bat.approach.pitchtype` | best and worst pitch type by xwOBA | `[P]` |
| `bat.approach.zone` | hottest and coldest zone | `[P]` |
| `bat.approach.count` | performance ahead vs behind | `[P]` |
| `bat.approach.firstpitch` | first-pitch swing rate vs league | `[P]` |

### Batters — trend and split (3)

| Code | Trigger | Source |
|---|---|---|
| `bat.trend.changepoint` | longest window differing from own baseline (§7) | |
| `bat.trend.series` | performance in this series | series logs |
| `bat.split.platoon` | vs LHP/RHP gap, shrunk hard | |

### Batters — matchup (3)

| Code | Trigger | Source |
|---|---|---|
| `bat.matchup.hand` | platoon edge vs tonight's starter | `[T]` |
| `bat.matchup.arsenal` | his xwOBA against that starter's actual mix | `[P]` |
| `bat.matchup.history` | career vs this pitcher — **shrunk to near-nothing**; will almost never clear the floor, which is correct | |

### Pitchers — repertoire (4)

| Code | Trigger | Source |
|---|---|---|
| `pit.arsenal.mix` | pitch usage, as-of | `[P]` |
| `pit.arsenal.best` | best pitch by run value per 100 | `[P]` |
| `pit.arsenal.velo` | velocity percentile **and trend** — a decline is a leading indicator | `[P]` |
| `pit.arsenal.movement` | break vs league for that pitch type | `[P]` |

### Pitchers — skill (4)

| Code | Trigger | Source |
|---|---|---|
| `pit.skill.k` | strikeout and whiff rate | |
| `pit.skill.command` | walk rate, zone rate, edge rate | `[P]` |
| `pit.skill.contact` | barrel and hard-hit rate allowed | `[P]` |
| `pit.skill.prevention` | ERA − FIP gap, signed | |

### Pitchers — weakness (5)

| Code | Trigger | Source |
|---|---|---|
| `pit.weak.pitchtype` | worst offering by xwOBA allowed | `[P]` |
| `pit.weak.zone` | where he gets hit | `[P]` |
| `pit.weak.order` | decline by `n_thruorder_pitcher` | `[P]` |
| `pit.weak.platoon` | vulnerability to opposite hand | `[P]` |
| `pit.weak.count` | performance behind in the count | `[P]` |

### Pitchers — workload and trend (3)

`pit.work.rest`, `pit.work.length` (IP/start trend), `pit.trend.changepoint`.

### Game level (6)

| Code | Trigger |
|---|---|
| `game.paper` | Elo separation |
| `game.form` | recent form — **descriptive only.** Measured collinear with Elo (unique r +0.003 vs SE 0.006), so it must never be phrased as adding predictive information |
| `game.starters` | starter talent differential `[T]` |
| `game.bullpen` | availability and quality |
| `game.key` | highest-significance findings across both clubs |
| `game.park` | park factor when materially non-neutral |

---

## 7. Change-point trends

"Trending up — for how long?" gets its own module because the obvious version is
wrong. Trying L5, L10, L15 and L30 and reporting the most dramatic reintroduces
multiple comparisons through the back door.

```python
def longest_significant_window(series, baseline, *, alpha) -> Window | None:
```

Scan candidate start points backwards; return the **longest** window whose mean
differs from baseline beyond sampling noise, with `alpha` corrected for the
number of windows tested. Returns `None` when nothing qualifies — which will be
most players most nights, and is the honest answer.

---

## 8. Phasing

**Phase A — machinery, 5 evaluators.**
`Finding`, references, reliability, selection, realization. Evaluators:
`bat.value.overall`, `bat.trend.changepoint`, `pit.arsenal.best`,
`pit.weak.order`, `game.starters`. Chosen because each exercises a different
part: talent model, change-point, pitch corpus, corpus-only-derivable, team level.

**Phase B — side-by-side.** Run both layers on the same ten games. Read them
next to each other. **This is the decision point.** If five well-chosen findings
do not beat the model's paragraph, forty will not close the gap — stop here and
keep the LLM.

**Phase C — measure stabilisation points.** Split-half reliability for every
metric in the catalogue. Needed before scoring means anything.

**Phase D — full catalogue**, in the order of the tables above.

**Phase E — retire.** Delete `analysis/agent.py`, drop the SDK dependency,
extend `test_no_llm_in_pipeline.py` to the whole package, remove the LLM
provenance note from the report footer.

---

## 9. Testing

| Test | Guards against |
|---|---|
| Golden findings per evaluator | silent trigger drift |
| Every `code` has ≥3 templates and all slots fill | a crash rendering a rare finding |
| Selection respects diversity constraints | eight ways of saying "he is good" |
| `z_crit` matches the live evaluator count | quietly raising the false-finding rate by adding evaluators |
| Shrinkage: a 20-PA .400 ranks below a 400-PA .340 | small-sample findings dominating |
| Determinism: same bundle → identical text, twice | the property that motivates the whole change |
| No evaluator reads data after `as_of` | leakage into descriptive prose |
| Empty case: a replacement-level player yields 0–1 findings | manufacturing significance |

---

## 10. Open questions

1. **Do findings need to compose?** Two related findings might read better as one
   sentence. Defer — start with independent sentences.
2. **Does the game summary need narrative order?** Probably: better-on-paper,
   then starters, then key players. Fixed order is fine and simpler.
3. **Contrast evaluators are not in the catalogue above.** They are the most
   valuable class and the hardest to enumerate; add after Phase B, informed by
   what the side-by-side shows the model catching that the evaluators miss.
4. **Report footer.** Currently says nothing on the projections page is
   model-generated. On completion that becomes true of the entire report — worth
   saying prominently.
