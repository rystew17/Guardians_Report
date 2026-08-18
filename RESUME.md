# Resume notes

State as of 2026-08-17. Written so a fresh session can pick up without
re-deriving anything.

## What this is

A daily Scouting Report / Game Preview for each Cleveland Guardians game,
aimed at everyone from a casual fan to a front office. Two hard rules:

1. Every number traces to a source endpoint and fetch timestamp.
2. All math is hard-coded in tested pure functions. No LLM computes anything.

Roadmap and gap analysis:
https://claude.ai/code/artifact/1d78bd05-f562-4e9f-9e57-e30273dad345

## Run it

```
cd C:\dev\guards-report
.venv\Scripts\python.exe -m guards_report.cli build --date 2026-08-18 --no-store
```

`--date` accepts `today`, `tomorrow`, `yesterday`, or `YYYY-MM-DD`. Drop
`--no-store` once BigQuery is authenticated. Output lands in `out/`.

**`--no-statcast`** skips the per-player pitch-level fetches. A full run is
~80 requests and about three minutes; without Statcast it is ~29 requests and
about twenty seconds, at the cost of the handedness-split zone maps and spray
charts. Use it when iterating on layout.

Tests: `.venv\Scripts\python.exe -m pytest -q` (82 offline).
Network tests, which validate our math against MLB's published values:
`pytest -m network` (6).

## Current state

Five tabs, ~51 player boxes, 29 source requests, ~737 KB of self-contained HTML.

* **Matchup** (landing) — standings, run differential, Pythagorean record and
  the luck gap, form, bullpen readiness, lineup handedness, both clubs ranked
  1-30 on offense and run prevention, and a "what to watch" block that ranks
  extremes already in the data.
* **CLE / OPP Pitchers** — 13 active arms each, probable starter pinned first,
  then bullpen by rest, then the rest of the rotation last.
* **CLE / OPP Position Players** — 13 each, sorted by plate appearances.

Every player box carries: Savant percentile chips, Season + L5/L15/L30 with
league deltas and a league-average row, pitch arsenal (for hitters, performance
against each pitch type), situational splits, two MLB zone heat maps, a trend
sparkline, and a reserved Phase 2 analysis slot. Position players additionally
carry batted-ball profile, bat tracking, OAA and sprint speed.

## Verified data sources (do not re-litigate)

Checked against live endpoints 2026-08-17.

**Usable:** MLB Stats API (`statsapi.mlb.com`) and Baseball Savant, both free
and unauthenticated.

**Not usable:** FanGraphs returns a Cloudflare interstitial (HTTP 403).
Baseball-Reference and Stathead have no API, prohibit automated access, and
enforce 10-20 requests/minute with day-long bans. `pybaseball` has disabled
both. A Stathead subscription grants manual CSV export, not API rights.

Key endpoint facts:

* `stats(...)` hydrate accepts a type list, a sitCodes list, and a personIds
  list **simultaneously** — this is what keeps a 52-player report at ~29
  requests instead of ~200. Game logs use a smaller batch (~130 KB/player).
* `statsapi/situationCodes` returns **602** split codes. We use ten per role.
  Groups: Count (19), Runners (15), Pitch Type (15), Inning (15), Position
  (14), Pitch Count (13), Month (12), At-Bat (10), Order (9).
* `hotColdZones` gives MLB's 13-cell zone grid **with its own colours**, which
  we use directly so maps match Savant.
* Savant leaderboards all accept `csv=true`. Id column is inconsistent:
  `player_id` on most, `id` on batted-ball and bat-tracking, `entity_id` on
  poptime.
* Standings carries `xWinLoss` — MLB's own Pythagorean record.

## Design decisions worth not re-deriving

* **Innings notation.** `"5.1"` is five and one third. Everything works in
  whole outs; game-log aggregation sums the API's `outs` field directly.
* **Undefined rates return `None`, never 0.0.** The renderer's `_missing()`
  covers None, Jinja `Undefined`, and blank CSV cells alike.
* **Sum counting stats over a window, then compute rates.** Never average
  rates.
* **Doubleheaders:** `gamePk` is *not* chronological within a date (observed
  2026-07-28, where the higher id was game one). Rows tie-break on payload
  order, which is authoritative.
* **OPS:** MLB publishes the sum of *rounded* OBP and SLG. `published_ops()`
  matches their convention for display; `ops()` keeps full precision.
* **League averages** come from summed team totals, never averaged across
  players.
* **FIP constant is derived per season** from league totals, validated to a
  2.5-3.7 band. 2026 derives to ~3.074 against a league ERA of ~4.17.
  League FIP equals league ERA by construction — a useful self-check.
* **Two-way players deliberately appear on both pages.**
* **Light theme is pinned** with `color-scheme: light`; do not add a dark mode.
* **Boxes are one per row, full width.** Two columns was too narrow.

## Outstanding

1. **BigQuery is not authenticated.** The ADC flow ran, opened a browser, but
   never wrote `application_default_credentials.json`. Everything works with
   `--no-store`. To finish:
   ```
   "C:\Users\ryste\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd" auth application-default login
   ```
   The terminal process must stay alive until the browser redirect completes.
   Then `guards-report setup-bq`, and add the $5 budget alert.
   GCP project is `ham-and-frank-67`; `.env` is already written and gitignored.

2. **Stage 6 of the roadmap**, not yet started: home plate umpire zone
   tendency (we fetch the umpire from the live feed and discard it), park
   factors, and transactions / injured-list movement.

3. **Phase 2** (LLM read-out into the reserved slots) and **Phase 3** (own
   projections) remain untouched by design.

4. Batter-vs-pitcher history (`stats=vsPlayer`) is built but not rendered.
   If restored, show raw counts only — rendering "3 for 8" as .375 implies a
   signal eleven plate appearances cannot carry.
