"""The app's job machinery, which is what a failed build reports itself through.

Three ways a run could end without ever saying so, all of them seen in
production as "the first build of the day fails and the second succeeds":

  a crash in the relay      left the job at "running" for the life of the
                            instance, so the browser sat on keepalives and
                            every later build was turned away with "a report is
                            already building" by a job that had died
  a collected task          `asyncio.create_task` keeps only a weak reference,
                            so a build stored nowhere could be garbage-collected
                            mid-run and end with no exit code at all
  a very long line          asyncio's stream limit is 64 KiB by default, and a
                            traceback past it raises inside the relay instead of
                            being delivered

Each one ends with the job silently stuck rather than failed, which is the
single worst outcome available: the thing that explains the failure is the
thing that got lost.
"""

from __future__ import annotations

import asyncio
import sys

from guards_report.app import main


def _job() -> main.Job:
    return main.Job(id="test01", game_date="2026-09-24")


def test_a_crash_inside_a_job_fails_it_rather_than_leaving_it_running():
    job = _job()

    async def boom():
        raise RuntimeError("relay died")

    asyncio.run(main._guarded(job, boom()))

    assert job.status == "failed"
    assert "RuntimeError" in (job.error or "")
    assert any(line.startswith("__FAILED__") for line in job.lines)


def test_a_cancelled_job_says_the_instance_went_away():
    job = _job()

    async def cancelled():
        raise asyncio.CancelledError

    try:
        asyncio.run(main._guarded(job, cancelled()))
    except asyncio.CancelledError:
        pass

    assert job.status == "failed"
    assert "reclaimed" in (job.error or "")


def test_a_running_job_is_referenced_so_it_cannot_be_collected():
    job = _job()
    started = asyncio.Event()

    async def slow():
        started.set()
        await asyncio.sleep(0.05)

    async def drive():
        main._spawn(job, slow())
        await started.wait()
        # While it runs it is reachable from the module, which is what keeps
        # the event loop's weak reference from being the only one.
        assert any(not task.done() for task in main._TASKS)
        await asyncio.sleep(0.2)
        assert not [task for task in main._TASKS if not task.done()]

    asyncio.run(drive())


def test_a_line_past_the_default_stream_limit_still_reaches_the_job(tmp_path,
                                                                    monkeypatch):
    job = _job()
    monkeypatch.setattr(main, "_settings", lambda: _Settings(tmp_path))

    long_line = "x" * 200_000
    # Built inside the child, because Windows caps a command line well below
    # the length this test is about.
    code = "print('x' * 200000)"
    returncode = asyncio.run(
        main._relay(job, [sys.executable, "-c", code], cwd=str(tmp_path)))

    assert returncode == 0
    assert long_line in job.lines


def test_a_finished_job_leaves_its_log_on_disk(tmp_path, monkeypatch):
    job = _job()
    job.publish("one line")
    job.status = "done"
    monkeypatch.setattr(main, "_settings", lambda: _Settings(tmp_path))

    main._archive(job)

    written = list((tmp_path / "logs").glob("*.log"))
    assert len(written) == 1
    assert "one line" in written[0].read_text(encoding="utf-8")
    assert "done" in written[0].name


class _Settings:
    def __init__(self, root):
        self.output_dir = root
