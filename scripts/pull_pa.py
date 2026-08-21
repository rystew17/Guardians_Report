"""Fetch the plate-appearance corpus. Resumable: cached chunks are skipped."""
from pathlib import Path
import sys, time
from guards_report.projections import corpus, pa

root = Path("data")
games = corpus.build(range(corpus.FIRST_SEASON, 2026), cache_dir=root / "corpus")
games = games.query("game_type=='R'")
seasons = list(range(pa.FIRST_SEASON, 2026))

start = time.time()
total = 0
for season in seasons:
    teams = pa.season_teams(games, season)
    frame = pa.build([season], teams, cache_dir=root / "pa")
    n = len(frame[frame.season == season])
    total += n
    print(f"{season}: {n:>7,} PA   [{time.time()-start:6.0f}s elapsed]", flush=True)

print(f"\nDONE  {total:,} plate appearances in {(time.time()-start)/60:.1f} min")
