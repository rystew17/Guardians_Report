# Guardians Report

A daily scouting report and game preview for Cleveland Guardians games, built
from the pitch corpus rather than from a summary of it.

Two rules the whole project is built around:

* **Every number is verifiable.** Anything the report shows that is not verbatim
  from a source API is computed in `metrics/formulas.py` or `insight/profile.py`,
  from raw counting stats, by a pure function with its formula in the docstring.
  Any figure on the page can be re-derived by hand.
* **No language model does arithmetic.** The written analysis is computed, not
  generated: the same game produces the same words every time, and each sentence
  traces to a number rather than needing to be checked against one.

## What it produces

A single HTML page per game covering both clubs' hitters and pitchers, the
series context, and a projections page carrying the fitted models' own held-out
accuracy beside their predictions.

There is also a **Not Gambling Advice** page comparing posted sportsbook prices
against the model's own probabilities. It prices the moneyline, run line,
totals, first five innings, starter strikeouts, batter hits and batter home
runs; it stakes only markets that have been measured against outcomes, and it
says plainly when it has nothing to bet, which is most nights.

## Running it

```bash
python -m guards_report.cli build --date today
python -m guards_report.cli serve          # the app, at :8765
python -m guards_report.cli publish <name> # copy a built report to the bucket
```

Configuration comes from `.env`; see `.env.example` for the keys. Nothing
environment-specific is committed.

## The models

| model | what it answers | held out on |
|---|---|---|
| Game outcome | who wins | seasons the fit never saw |
| Score | how many runs, per side | same |
| First five | the F5 result and total | same |
| Props | hits, home runs, strikeouts per player | same |

Every market the betting page prices has been checked against realized
frequencies, walking forward season by season and refitting on prior seasons
only. That record is what produces a standard error, and without one nothing is
staked -- an unmeasured model is shown and refused rather than sized.

## Layout

```
src/guards_report/
  ingest/       pulling and assembling a game
  metrics/      pure statistical functions
  insight/      player profiles and the written analysis
  projections/  the fitted models, training and calibration
  betting/      prices, edge, staking and the closing-line record
  odds/         sportsbook prices in, stored and grouped into markets
  report/       rendering
  app/          the local web app
scripts/        refit, calibrate, sync
tests/          741 tests
```

## Deploying

`Dockerfile` and `scripts/deploy.sh` put the app on Cloud Run with the data
bucket mounted, so it can be run from a phone. See `scripts/deploy.sh` for what
it sets and why.
