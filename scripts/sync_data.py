"""Mirror the corpora to object storage, then confirm the bucket matches disk.

Re-runnable: unchanged files are skipped after a metadata lookup, so a repeat
sync costs seconds rather than a re-upload.
"""
import time
from pathlib import Path
from guards_report.config import load_settings
from guards_report.publish import datasets

s = load_settings()
project = getattr(s, "gcp_project", "")
t0 = time.time()

result = datasets.sync(Path("data"), bucket_name=s.gcs_bucket, project=project)
print(f"\nsync   {result.line()}   [{(time.time()-t0)/60:.1f} min]")
for failure in result.failures:
    print(f"  FAILED {failure}")

check = datasets.verify(Path("data"), bucket_name=s.gcs_bucket, project=project)
print(f"verify {check.line()}")
for name in check.missing[:5]:
    print(f"  MISSING {name}")
for name in check.mismatched[:5]:
    print(f"  MISMATCH {name}")

print("\nBACKUP COMPLETE AND VERIFIED" if check.ok and not result.failures
      else "\nBACKUP INCOMPLETE -- see above")
raise SystemExit(0 if check.ok and not result.failures else 1)
