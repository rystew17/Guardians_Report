"""Progress of the pitch-corpus pull. Safe to run any time, including mid-pull."""
from pathlib import Path
import time

ROOT = Path("data/pitches")
TARGET = 360          # 30 clubs x 2015-2026
LOG = Path(
    "C:/Users/ryste/AppData/Local/Temp/claude/"
    "C--Users-ryste-OneDrive-Desktop-ClaudeWorkspace/"
    "a31d0354-a9e2-494a-be00-883edf2dfe2b/tasks/b0n3szok5.output"
)

files = sorted(ROOT.glob("*.parquet"))
done = len(files)
size = sum(f.stat().st_size for f in files)

if not files:
    print("no chunks yet")
    raise SystemExit

started = min(f.stat().st_mtime for f in files)
newest = max(f.stat().st_mtime for f in files)
elapsed = newest - started
rate = done / elapsed if elapsed > 0 else 0
remaining = (TARGET - done) / rate if rate else 0

bar = int(done / TARGET * 40)
print(f"[{'#' * bar}{'.' * (40 - bar)}] {done}/{TARGET}  {done/TARGET:.0%}")
print(f"  on disk      {size/1e6:,.0f} MB   (projected {size/max(done,1)*TARGET/1e9:.2f} GB)")
print(f"  rate         {rate*60:.1f} chunks/min")
print(f"  elapsed      {elapsed/60:.0f} min")
print(f"  est. remain  {remaining/60:.0f} min")
print(f"  stale?       last chunk written {(time.time()-newest):.0f}s ago")

seasons = {}
for f in files:
    seasons[f.name[:4]] = seasons.get(f.name[:4], 0) + 1
line = "  ".join(f"{s}:{n}/30" for s, n in sorted(seasons.items()))
print(f"  by season    {line}")

if LOG.exists():
    tail = [l for l in LOG.read_text(errors="replace").splitlines() if l.strip()][-2:]
    for l in tail:
        print(f"  log          {l.strip()}")
