"""Refit the models on a schedule, and refuse a refit that made things worse.

Two scripts fit this project's models and they age at completely different
rates, which is the whole reason this wrapper exists rather than a task that
runs both.

`train_props` fits on the season in progress. Its `corpus_through` was two days
behind after two days, so it is genuinely perishable and cheap to redo -- forty
seconds. That is the one worth running often.

`train` fits the game-outcome coefficients on *completed* seasons only, so
running it in August reproduces the same coefficients it produced in July. What
does move is the talent prior, which absorbs the current season's plate
appearances -- and it moves slowly: four plate appearances and a fifth decimal
place over two days. Three and a half minutes to refresh that is not a weekly
job, so it runs on a much longer clock or when asked.

The guard matters more than the schedule. A refit that produces a worse model
should not silently become the model, and the training run already measures
itself on held-out seasons. So the new artifact's pooled log loss is compared
against the one it would replace, and a material regression puts the old file
back. Lower is better for log loss, which is the sort of thing that reads
correctly in either direction and has to be pinned by a test.

Exit codes are for the scheduler: 0 refit and kept, 1 something failed, 2 refit
ran and was rejected. A silent failure is the thing a scheduled job must never
do, so all three say so in the log.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "data" / "models"
LOG = ROOT / "data" / "refit.log"

# A refit is rejected when it loses more than this much log loss against the
# artifact it would replace. Not zero: the training set grows every day, so the
# figure moves slightly in both directions on its own, and rejecting every
# wobble would mean never accepting a refit at all. The pooled figure has a
# standard deviation of 0.003 across seasons, so this is well inside noise.
REGRESSION_TOLERANCE = 0.004

# How old the game-outcome artifact has to be before `train` is worth the three
# and a half minutes. Its coefficients do not move mid-season at all; only the
# talent prior drifts, and it drifts slowly.
OUTCOME_REFIT_DAYS = 30


def _say(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp}  {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _pooled_log_loss(path: Path) -> float | None:
    """The held-out figure the training run recorded, if there is one."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    metrics = payload.get("metrics") or {}
    for block in (metrics.get("win_model"), metrics):
        if isinstance(block, dict) and "log_loss" in block:
            try:
                return float(block["log_loss"])
            except (TypeError, ValueError):
                continue
    return None


def _age_days(path: Path) -> float:
    """Days since the artifact was fitted, from its own stamp.

    Not file mtime: a restore from cloud storage resets that on every artifact
    at once, which would make the whole set look freshly trained.
    """
    if not path.is_file():
        return float("inf")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        stamp = payload.get("fitted_at")
        if not stamp:
            return float("inf")
        fitted = datetime.fromisoformat(stamp)
        if fitted.tzinfo is None:
            fitted = fitted.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - fitted).total_seconds() / 86400
    except (OSError, ValueError):
        return float("inf")


def _run(module: str) -> tuple[int, str]:
    started = time.time()
    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    _say(f"  {module}: exit {result.returncode} in {time.time() - started:.0f}s")
    if result.returncode != 0:
        for line in (result.stderr or "").splitlines()[-8:]:
            _say(f"    {line[:160]}")
    return result.returncode, result.stdout or ""


def refit_props() -> bool:
    """The perishable half. Nothing to validate: it has no held-out metric."""
    _say("refitting props and first-five (fits on the current season)")
    code, _ = _run("guards_report.projections.train_props")
    return code == 0


def refit_outcome(*, force: bool = False) -> str:
    """The slow half, with a guard. Returns kept | skipped | rejected | failed."""
    artifact = MODELS / "game_outcome.json"
    age = _age_days(artifact)
    if not force and age < OUTCOME_REFIT_DAYS:
        _say(f"game-outcome model is {age:.0f} days old; "
             f"coefficients do not move mid-season, skipping")
        return "skipped"

    before = _pooled_log_loss(artifact)
    backup = artifact.with_suffix(".json.previous")
    if artifact.is_file():
        shutil.copy2(artifact, backup)

    _say(f"refitting the game-outcome model (was {age:.0f} days old)")
    code, _ = _run("guards_report.projections.train")
    if code != 0:
        if backup.is_file():
            shutil.copy2(backup, artifact)
            _say("  training failed; previous model restored")
        return "failed"

    after = _pooled_log_loss(artifact)
    if before is None or after is None:
        _say(f"  no held-out figure to compare (before={before}, after={after}); keeping")
        return "kept"

    # Lower log loss is better, so a positive delta is a regression.
    delta = after - before
    if delta > REGRESSION_TOLERANCE:
        if backup.is_file():
            shutil.copy2(backup, artifact)
        _say(f"  REJECTED: log loss {before:.5f} -> {after:.5f} "
             f"({delta:+.5f}, tolerance {REGRESSION_TOLERANCE}); previous model restored")
        return "rejected"

    _say(f"  kept: log loss {before:.5f} -> {after:.5f} ({delta:+.5f})")
    return "kept"


def sync() -> bool:
    _say("syncing artifacts to cloud storage")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "sync_data.py")],
        cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    for line in (result.stdout or "").splitlines()[-2:]:
        _say(f"  {line[:160]}")
    return result.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-outcome", action="store_true",
                        help="refit the game-outcome model regardless of age")
    parser.add_argument("--no-sync", action="store_true",
                        help="skip the upload to cloud storage")
    args = parser.parse_args(argv)

    _say("=" * 60)
    _say("refit starting")

    ok = refit_props()
    outcome = refit_outcome(force=args.force_outcome)

    if not args.no_sync and (ok or outcome == "kept"):
        sync()

    if not ok or outcome == "failed":
        _say("refit FAILED")
        return 1
    if outcome == "rejected":
        _say("refit ran and was rejected; the previous model is still in place")
        return 2
    _say("refit complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
