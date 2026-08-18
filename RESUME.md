# Resume notes

State of the build as of 2026-08-17. Written so a fresh session can pick up
without re-deriving anything.

## What this project is

A daily Scouting Report / Game Preview for each Cleveland Guardians game,
covering both starting pitchers, available relievers, and batters, with season
and recent-form numbers. Two hard rules:

1. Every number is traceable to a source endpoint and fetch timestamp.
2. All math is hard-coded in tested pure functions. No LLM computes anything.

The full plan lives at:
`C:\Users\ryste\.claude\plans\i-am-enlisting-your-silly-yeti.md`

## Where we stopped, and why

Blocked on installing Python. The python.org MSI fails with error 2203 /
`ERROR_PATH_NOT_FOUND` because Windows has a **pending update reboot**
(`RebootPending`, `RebootRequired`, and `PendingFileRenameOperations` are all
set). Windows Installer cannot complete a multi-package transaction in that
state, so `core.msi` never lands in the Package Cache. This is unrelated to
Python or to the sandbox -- it failed identically with the sandbox disabled.

**Next step: reboot Windows, then re-run the installer.** It is already
downloaded, no need to fetch it again:

```
C:\Users\ryste\AppData\Local\Temp\claude\installers\python-3.13.15-amd64.exe
```

Install per-user, quiet, added to PATH:

```powershell
Start-Process -FilePath "C:\Users\ryste\AppData\Local\Temp\claude\installers\python-3.13.15-amd64.exe" -ArgumentList "/quiet","InstallAllUsers=0","PrependPath=1","Include_test=0" -Wait -PassThru
```

Note: Python 3.12 is now security-only and has no Windows installer past
3.12.10, which is why this is 3.13.15.

## Verified research findings (do not re-litigate)

Checked against live endpoints on 2026-08-17.

**Usable:**
- **MLB Stats API** (`statsapi.mlb.com`), free and unauthenticated. Confirmed
  working: `schedule` with `hydrate=probablePitcher,lineups,venue`;
  `teams/114/roster/active`; `people?personIds=a,b,c` (batch lookup);
  `people/{id}/stats` with `stats=gameLog`, `lastXGames&limit=N`,
  `byDateRange`, `statSplits&sitCodes=vl,vr,h,a`, `vsPlayer`, and
  `sabermetrics` (returns wOBA, wRC+, WAR, wRAA, RAR directly);
  `v1.1/game/{pk}/feed/live` for weather and umpires; `v1/teams/stats` for the
  30 team season lines.
- **Baseball Savant**, free and unauthenticated. Confirmed working:
  `leaderboard/expected_statistics?csv=true`,
  `leaderboard/pitch-arsenal-stats?csv=true`,
  `leaderboard/percentile-rankings`, `player-services/statcast-pitches-breakdown`.

**Not usable programmatically:**
- **FanGraphs** returns a Cloudflare interstitial (HTTP 403) even with a
  browser User-Agent. `pybaseball` dropped FanGraphs support entirely.
- **Baseball-Reference / Stathead** have no API. Terms of Use prohibit
  automated access, enforced at 10 req/min (Stathead) and 20 req/min (BRef)
  with day-long bans. `pybaseball` has disabled its BRef modules. A Stathead
  subscription grants manual CSV/Excel export, not API rights.

Decision: build on MLB Stats API + Savant, and hard-code the derived metrics
(FIP, xFIP, SIERA, K%, BB%, Game Score). Computing these from audited raw
inputs is *better* for verifiability than scraping someone else's derived
number.

## Useful constants already confirmed against live data

- Cleveland Guardians `teamId` = **114**; Progressive Field `venueId` = 5.
- 2026 league pitching totals (summed across all 30 clubs, from
  `v1/teams/stats`, as of 2026-08-17): ER 15381, outs 99712, HR 4326,
  BB 12642, HBP 1616, K 31341.
- Those give league ERA **4.1649** and a derived **FIP constant of 3.0718**,
  which is right in the historically normal band. `league_constants.py`
  validates against a 2.5-3.7 range and this passes comfortably.

## What is built

```
src/guards_report/
  config.py                    settings, team ids, table names, rate limits
  sources/http.py              THE choke point for all outbound requests:
                               rate limiting, retry/backoff, content-addressed
                               gzip archive + sha256 provenance
  sources/mlb_statsapi.py      typed client for every endpoint listed above
  metrics/formulas.py          all hard-coded math, pure functions
  metrics/league_constants.py  derives the FIP constant from league totals
  metrics/windows.py           L5/L15/L30 aggregation + bullpen availability
tests/test_formulas.py         formula tests
```

Design decisions worth not re-deriving:

- **Innings-pitched notation.** `"5.1"` means five and one third, not 5.1.
  Everything works in whole outs internally (`ip_to_outs`). Game-log
  aggregation sums the API's `outs` field directly and never parses the
  innings string.
- **Undefined rates return `None`, never 0.0.** A 0.000 ERA and an undefined
  ERA are different claims and the report must not conflate them.
- **Sum counting stats over a window, then compute rates.** Averaging per-game
  rates is a different and wrong number.
- **Windows differ by role.** Hitters get L5/L15/L30 games. A starter's "last
  30 games" would span two seasons, so starters get day-based windows
  (15/30/60 days). Relievers get a mix. See `windows.py`. **Worth confirming
  with the user** -- the original ask said L5/L15/L30 games for everyone.
- **`sabermetrics` values are taken from the source, not derived.** wOBA and
  wRC+ need season-specific linear weights and park factors that MLB computes
  internally; reconstructing them would mean guessing at inputs.

## Storage design (the cost question, already settled)

Governing principle: **store the atom, derive the rest.** Game logs are
immutable once a game is final and everything else (season, L5, L15, L30) is a
deterministic aggregation of them. So game logs persist; windows are
zero-storage SQL views.

Sized from measured payloads: **~40 MB per full season, league-wide**, against
BigQuery's 10 GiB free storage tier. Cut from the original scope:
league-wide pitch-level Statcast (~500 MB/season, replaced by Savant's
pre-aggregated leaderboards) and raw JSON blobs in BQ (~2 GB/season, replaced
by gzipped local files plus metadata rows).

Guardrails: `maximum_bytes_billed` on every query, `require_partition_filter`
on fact tables, no `SELECT *`, batch loads only (never the streaming API,
which is the path that actually costs money), and a $5/month budget alert.

## Still to build

- `sources/savant.py` -- CSV/JSON client for the Savant leaderboards
- `ingest/` -- schedule, rosters, game logs (incremental), splits
- `store/` -- BQ schemas with partitioning, client with byte caps, window views
- `report/` -- ReportBundle assembly, Jinja2 HTML renderer with provenance and
  a Sources & Audit appendix
- `cli.py` -- `build | backfill | cost`
- `tests/test_source_agreement.py` -- asserts our formulas reproduce the
  source API's own published AVG/OBP/SLG/ERA/WHIP from its own counting stats.
  This is the strongest verification available and needs network.

## Open questions for the user

1. **GCP project ID** is still needed for `.env` (`GCP_PROJECT`). Run
   `gcloud projects list` after installing the SDK.
2. **Pitcher recent-form windows**: confirm day-based windows for starters
   instead of literal L5/L15/L30 games.
