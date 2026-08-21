"""Mirror the corpora to object storage. Safe to re-run: unchanged files skip."""
import time
from pathlib import Path
from guards_report.config import load_settings
from guards_report.publish import datasets

s = load_settings()
t0 = time.time()
result = datasets.sync(
    Path("data"), bucket_name=s.gcs_bucket, project=getattr(s, "gcp_project", "")
)
print(f"\n{result.line()}   [{(time.time()-t0)/60:.1f} min]")
for failure in result.failures:
    print(f"  FAILED {failure}")
raise SystemExit(1 if result.failures else 0)
