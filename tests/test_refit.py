"""The scheduled refit, and the guard that can refuse one.

A scheduled job has one failure mode worse than crashing: succeeding quietly
while doing the wrong thing. So the two things tested here are that a refit
which made the model worse is actually rejected, and that a failure is reported
rather than swallowed.

The rejection guard reads correctly in both directions, which is why it is worth
pinning. Log loss is lower-is-better, so a *rise* is the regression -- inverted,
the guard would keep every bad refit and throw away every good one, and the log
line would still say something reassuring.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "refit.py"


def _module():
    spec = importlib.util.spec_from_file_location("refit_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["refit_script"] = module
    spec.loader.exec_module(module)
    return module


refit = _module()


def _artifact(path: Path, *, log_loss: float, fitted_at: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "fitted_at": fitted_at,
        "metrics": {"win_model": {"log_loss": log_loss}},
    }), encoding="utf-8")


# ---------------------------------------------------------------------------
# Reading the artifact
# ---------------------------------------------------------------------------

def test_the_held_out_figure_is_read_from_the_artifact(tmp_path):
    path = tmp_path / "game_outcome.json"
    _artifact(path, log_loss=0.67668, fitted_at="2026-08-01T00:00:00+00:00")
    assert refit._pooled_log_loss(path) == pytest.approx(0.67668)


def test_a_missing_or_unreadable_artifact_yields_no_figure(tmp_path):
    """None means "cannot compare", which the caller treats as keep rather than
    as a regression. Returning 0.0 would reject every refit forever."""
    assert refit._pooled_log_loss(tmp_path / "absent.json") is None
    broken = tmp_path / "broken.json"
    broken.write_text("{ not json", encoding="utf-8")
    assert refit._pooled_log_loss(broken) is None


def test_age_comes_from_the_artifacts_own_stamp(tmp_path):
    """Not file mtime: a restore from cloud storage resets that on every
    artifact at once, making the whole set look freshly trained."""
    path = tmp_path / "game_outcome.json"
    stamp = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    _artifact(path, log_loss=0.677, fitted_at=stamp)
    assert refit._age_days(path) == pytest.approx(40, abs=0.1)


def test_an_artifact_with_no_stamp_reads_as_infinitely_old(tmp_path):
    """So it refits rather than being skipped forever."""
    path = tmp_path / "game_outcome.json"
    path.write_text(json.dumps({"metrics": {}}), encoding="utf-8")
    assert refit._age_days(path) == float("inf")
    assert refit._age_days(tmp_path / "absent.json") == float("inf")


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------

def _run_outcome(tmp_path, monkeypatch, *, before, after, exit_code=0):
    """Drive `refit_outcome` with a fake training run of a chosen outcome."""
    models = tmp_path / "models"
    artifact = models / "game_outcome.json"
    _artifact(artifact, log_loss=before, fitted_at="2020-01-01T00:00:00+00:00")

    monkeypatch.setattr(refit, "MODELS", models)
    monkeypatch.setattr(refit, "LOG", tmp_path / "refit.log")

    def fake_run(module):
        if exit_code == 0:
            _artifact(artifact, log_loss=after,
                      fitted_at="2026-08-23T00:00:00+00:00")
        return exit_code, ""

    monkeypatch.setattr(refit, "_run", fake_run)
    verdict = refit.refit_outcome(force=True)
    return verdict, refit._pooled_log_loss(artifact)


def test_a_refit_that_improved_the_model_is_kept(tmp_path, monkeypatch):
    verdict, kept = _run_outcome(
        tmp_path, monkeypatch, before=0.6800, after=0.6750)
    assert verdict == "kept"
    assert kept == pytest.approx(0.6750)


def test_a_refit_that_made_the_model_worse_is_rejected(tmp_path, monkeypatch):
    """Lower log loss is better, so a rise is the regression.

    Inverted, this guard would keep every bad refit and discard every good one
    while logging something reassuring either way.
    """
    verdict, kept = _run_outcome(
        tmp_path, monkeypatch, before=0.6750, after=0.7100)
    assert verdict == "rejected"
    assert kept == pytest.approx(0.6750), "the previous model must be restored"


def test_a_wobble_inside_the_tolerance_is_accepted(tmp_path, monkeypatch):
    """The training set grows daily, so the figure moves on its own.

    Pooled log loss has a standard deviation of 0.003 across seasons; rejecting
    every movement would mean never accepting a refit at all.
    """
    assert refit.REGRESSION_TOLERANCE > 0
    verdict, _ = _run_outcome(
        tmp_path, monkeypatch,
        before=0.6750, after=0.6750 + refit.REGRESSION_TOLERANCE / 2)
    assert verdict == "kept"


def test_the_tolerance_is_smaller_than_a_difference_worth_noticing():
    """Wide enough to absorb noise, narrow enough to catch a real regression.

    The measured spread across folds is 0.003 and the model's whole edge over
    the baseline is about 0.015, so a tolerance near the latter would wave
    through a refit that gave the edge away.
    """
    assert 0.001 <= refit.REGRESSION_TOLERANCE <= 0.008


def test_a_failed_training_run_restores_the_previous_model(tmp_path, monkeypatch):
    """A crashed refit must cost nothing. The artifact is written atomically,
    and this is the belt to that brace."""
    verdict, kept = _run_outcome(
        tmp_path, monkeypatch, before=0.6750, after=0.0, exit_code=1)
    assert verdict == "failed"
    assert kept == pytest.approx(0.6750)


def test_a_recent_model_is_skipped_rather_than_refitted(tmp_path, monkeypatch):
    """Three and a half minutes to reproduce coefficients that cannot move.

    They are fitted on completed seasons, so an August refit returns July's
    numbers. Only the talent prior drifts, and slowly.
    """
    models = tmp_path / "models"
    stamp = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    _artifact(models / "game_outcome.json", log_loss=0.6750, fitted_at=stamp)
    monkeypatch.setattr(refit, "MODELS", models)
    monkeypatch.setattr(refit, "LOG", tmp_path / "refit.log")

    called: list[str] = []

    def spy(module):
        called.append(module)
        return 0, ""

    monkeypatch.setattr(refit, "_run", spy)
    assert refit.refit_outcome(force=False) == "skipped"
    assert called == [], "the expensive run must not have happened"


# ---------------------------------------------------------------------------
# Exit codes — what the scheduler reads
# ---------------------------------------------------------------------------

def test_the_exit_codes_distinguish_the_three_outcomes(tmp_path, monkeypatch):
    """A scheduled job that returns 0 whatever happened is a job nobody checks."""
    monkeypatch.setattr(refit, "LOG", tmp_path / "refit.log")
    monkeypatch.setattr(refit, "sync", lambda: True)

    monkeypatch.setattr(refit, "refit_props", lambda: True)
    monkeypatch.setattr(refit, "refit_outcome", lambda *, force: "kept")
    assert refit.main(["--no-sync"]) == 0

    monkeypatch.setattr(refit, "refit_outcome", lambda *, force: "rejected")
    assert refit.main(["--no-sync"]) == 2

    monkeypatch.setattr(refit, "refit_props", lambda: False)
    assert refit.main(["--no-sync"]) == 1


def test_every_run_leaves_a_line_in_the_log(tmp_path, monkeypatch):
    """The whole point of an unattended job: something to read afterwards."""
    log = tmp_path / "refit.log"
    monkeypatch.setattr(refit, "LOG", log)
    monkeypatch.setattr(refit, "refit_props", lambda: True)
    monkeypatch.setattr(refit, "refit_outcome", lambda *, force: "skipped")
    refit.main(["--no-sync"])
    assert log.is_file()
    text = log.read_text(encoding="utf-8")
    assert "refit starting" in text and "refit complete" in text
