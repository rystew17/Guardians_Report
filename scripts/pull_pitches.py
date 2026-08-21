"""Fetch the pitch-level corpus. Resumable: cached team-seasons are skipped.

One download serves both grains. Savant returns the full pitch stream for a
team-season regardless of what is asked for, so keeping every pitch costs disk
(about 0.26 GB) and no additional time, while discarding it would mean paying
for the same bytes again later.
"""
from datetime import date
from pathlib import Path
import time
from guards_report.projections import corpus, pitches

root = Path("data")
CURRENT = date.today().year
# Must include the current season: `season_teams` reads the club list from
# this frame, and a range that stops short returns no teams and fetches
# nothing, silently.
games = corpus.build(range(corpus.FIRST_SEASON, CURRENT + 1), cache_dir=root / "corpus")
games = games.query("game_type=='R'")

start = time.time()
grand = 0

for season in range(pitches.FIRST_SEASON, CURRENT + 1):
    teams = pitches.season_teams(games, season)
    if season < CURRENT:
        # A finished season never changes: fetch once, read forever.
        frame = pitches.build([season], teams, cache_dir=root / "pitches")
        n = len(frame[frame.season == season])
    else:
        # The season in progress is append-only, and it is the one the talent
        # estimates actually serve from -- without it `update_as_of` has nothing
        # to carry forward and the report falls back to last year's players.
        added = pitches.refresh_current_season(
            season, teams, cache_dir=root / "pitches"
        )
        n = sum(added.values())
    grand += n
    print(f"{season}: {n:>9,} pitches  [{time.time()-start:6.0f}s]", flush=True)

print(f"\nDONE  {grand:,} pitches in {(time.time()-start)/60:.1f} min")
