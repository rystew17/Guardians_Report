"""Fetch the pitch-level corpus. Resumable: cached team-seasons are skipped.

One download serves both grains. Savant returns the full pitch stream for a
team-season regardless of what is asked for, so keeping every pitch costs disk
(about 0.26 GB) and no additional time, while discarding it would mean paying
for the same bytes again later.
"""
from pathlib import Path
import time
from guards_report.projections import corpus, pitches

root = Path("data")
games = corpus.build(range(corpus.FIRST_SEASON, 2026), cache_dir=root / "corpus")
games = games.query("game_type=='R'")

start = time.time()
grand = 0
for season in range(pitches.FIRST_SEASON, 2026):
    teams = pitches.season_teams(games, season)
    frame = pitches.build([season], teams, cache_dir=root / "pitches")
    n = len(frame[frame.season == season])
    grand += n
    print(f"{season}: {n:>9,} pitches  [{time.time()-start:6.0f}s]", flush=True)

print(f"\nDONE  {grand:,} pitches in {(time.time()-start)/60:.1f} min")
