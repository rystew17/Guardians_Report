"""Local web app for generating and downloading reports.

Runs on localhost only. A full build takes a few minutes, so generation happens
in a background job and progress streams to the browser over server-sent
events rather than leaving a request hanging.

The job runs the CLI as a subprocess and relays its stderr as progress. That
keeps the app decoupled from the pipeline's internals: the CLI already prints
meaningful progress, and a crash in a build cannot take the server down with
it.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from guards_report.config import load_settings
from guards_report.metrics import clocks
from guards_report.publish import gcs

app = FastAPI(title="Guardians Report")


@dataclass
class Job:
    """One report build, and everything the browser needs to follow it."""

    id: str
    game_date: str
    status: str = "running"       # running | done | failed
    lines: list[str] = field(default_factory=list)
    output_path: str | None = None
    error: str | None = None
    started_at: datetime = field(default_factory=datetime.now)
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)


JOBS: dict[str, Job] = {}


def _root() -> Path:
    """The project root, from this file's location."""
    return Path(__file__).resolve().parents[3]


def _settings():
    return load_settings()


async def _run_build(job: Job, *, statcast: bool, analysis: bool) -> None:
    """Run the CLI and relay its output to the job's event queue.

    The analysis is computed, not generated. This used to pass
    `--with-analysis --model`, which is the flag that routes to a language
    model -- so the app kept calling one for months after the CLI default
    stopped. Written analysis is now the CLI's default and the checkbox only
    turns it off.
    """
    cmd = [
        sys.executable, "-m", "guards_report.cli", "build",
        "--date", job.game_date, "--no-store",
    ]
    if not statcast:
        cmd.append("--no-statcast")
    if not analysis:
        cmd.append("--no-analysis")

    # Windows defaults child stdio to the ANSI code page, so the separators and
    # accented names the CLI prints would arrive as mojibake once decoded as
    # UTF-8. Pinning the child's encoding is the fix; decoding leniently only
    # hides it.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(Path(__file__).resolve().parents[3]),
        env=env,
    )

    assert process.stdout is not None
    async for raw in process.stdout:
        line = raw.decode("utf-8", errors="replace").rstrip()
        if not line:
            continue
        job.lines.append(line)
        # The CLI prints the finished path on stdout as its last line.
        if line.endswith(".html"):
            job.output_path = line.strip()
        await job.queue.put(line)

    await process.wait()
    if process.returncode == 0 and job.output_path:
        job.status = "done"
        await job.queue.put(f"__DONE__ {Path(job.output_path).name}")
    else:
        job.status = "failed"
        job.error = f"build exited with code {process.returncode}"
        await job.queue.put(f"__FAILED__ {job.error}")


@app.post("/api/generate")
async def generate(request: Request) -> JSONResponse:
    body = await request.json()
    game_date = (body.get("date") or date.today().isoformat()).strip()

    job = Job(id=uuid.uuid4().hex[:12], game_date=game_date)
    JOBS[job.id] = job

    asyncio.create_task(
        _run_build(
            job,
            statcast=bool(body.get("statcast", True)),
            analysis=bool(body.get("analysis", True)),
        )
    )
    return JSONResponse({"job_id": job.id})


async def _run_refit(job: Job, *, force_outcome: bool) -> None:
    """Run the refit script and relay its output, exactly as a build does.

    Same job machinery on purpose. A refit is the other long-running thing this
    app starts, and giving it its own progress channel would mean two ways to
    watch a subprocess that behave subtly differently.

    The script distinguishes its outcomes by exit code -- 0 kept, 1 failed,
    2 refit ran and was rejected -- and 2 is not a failure. A refit that was
    measured and turned down is the guard working, and reporting it as an error
    would train the reader to ignore the one message worth reading.
    """
    cmd = [sys.executable, str(_root() / "scripts" / "refit.py")]
    if force_outcome:
        cmd.append("--force-outcome")

    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(_root()),
        env=env,
    )

    assert process.stdout is not None
    async for raw in process.stdout:
        line = raw.decode("utf-8", errors="replace").rstrip()
        if not line:
            continue
        job.lines.append(line)
        await job.queue.put(line)

    await process.wait()
    if process.returncode == 0:
        job.status = "done"
        await job.queue.put("__DONE__ refit complete")
    elif process.returncode == 2:
        job.status = "done"
        await job.queue.put(
            "__DONE__ refit rejected — the previous model is still in place")
    else:
        job.status = "failed"
        job.error = f"refit exited with code {process.returncode}"
        await job.queue.put(f"__FAILED__ {job.error}")


@app.get("/api/model-status")
async def model_status() -> JSONResponse:
    """How old each fitted model is, so the page can say whether a refit is due.

    Read from each artifact's own `fitted_at` rather than file mtime, which a
    restore from cloud storage resets on every artifact at once -- the whole set
    would look freshly trained the day after a disaster.
    """
    from guards_report.projections import freshness

    models = _root() / "data" / "models"
    try:
        ages = freshness.fitted_ages(models)
    except Exception:  # noqa: BLE001 -- a status panel must not break the page
        ages = {}

    oldest = max(ages.values()) if ages else None
    return JSONResponse({
        "ages": ages,
        "oldest": oldest,
        "stale_after": freshness.REFIT_AFTER_DAYS,
        "due": bool(oldest is not None and oldest > freshness.REFIT_AFTER_DAYS),
    })


@app.post("/api/refit")
async def refit(request: Request) -> JSONResponse:
    """Start a refit and hand back a job id to follow it with."""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 -- an empty body is a plain refit
        body = {}

    running = [j for j in JOBS.values()
               if j.status == "running" and j.game_date == "refit"]
    if running:
        # Two refits at once would race on the same artifacts, and the second
        # would overwrite whatever the first had just validated.
        return JSONResponse(
            {"ok": False, "error": "a refit is already running"},
            status_code=409)

    job = Job(id=uuid.uuid4().hex[:12], game_date="refit")
    JOBS[job.id] = job
    asyncio.create_task(
        _run_refit(job, force_outcome=bool(body.get("force_outcome", False))))
    return JSONResponse({"job_id": job.id})


@app.get("/api/events/{job_id}")
async def events(job_id: str) -> StreamingResponse:
    """Stream a job's progress lines as server-sent events."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")

    async def stream():
        # Replay anything already emitted, so a late or reconnecting browser
        # sees the whole run rather than joining midway.
        for line in list(job.lines):
            yield f"data: {line}\n\n"
        if job.status != "running":
            marker = "__DONE__" if job.status == "done" else "__FAILED__"
            name = Path(job.output_path).name if job.output_path else (job.error or "")
            yield f"data: {marker} {name}\n\n"
            return

        while True:
            try:
                line = await asyncio.wait_for(job.queue.get(), timeout=30)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"   # keeps proxies and browsers from closing
                continue
            yield f"data: {line}\n\n"
            if line.startswith(("__DONE__", "__FAILED__")):
                return

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/reports")
async def reports() -> JSONResponse:
    out = _settings().output_dir
    if not out.exists():
        return JSONResponse([])

    items = []
    for path in sorted(out.glob("*.html"), reverse=True):
        stat = path.stat()
        items.append({
            "name": path.name,
            "size_kb": round(stat.st_size / 1024),
            "modified": clocks.stamp(datetime.fromtimestamp(stat.st_mtime).astimezone()),
        })
    return JSONResponse(items)


def _safe_report(name: str) -> Path:
    """Resolve a report name to a path inside the output directory.

    The name arrives from the URL, so it is untrusted: resolve it and confirm
    it is still inside the output directory before serving anything.
    """
    out = _settings().output_dir.resolve()
    path = (out / name).resolve()
    if not path.is_relative_to(out) or not path.is_file():
        raise HTTPException(status_code=404, detail="report not found")
    return path


@app.get("/report/{name}")
async def view_report(name: str) -> FileResponse:
    return FileResponse(_safe_report(name), media_type="text/html")


@app.get("/download/{name}")
async def download_report(name: str) -> FileResponse:
    path = _safe_report(name)
    return FileResponse(path, media_type="text/html", filename=path.name)


@app.post("/api/publish")
async def publish(request: Request) -> JSONResponse:
    """Copy one built report to Cloud Storage and return its link.

    Runs in a worker thread: the upload is blocking network I/O and would
    otherwise stall every other request the app is serving.
    """
    body = await request.json()
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="no report named")

    path = _safe_report(name)
    settings = _settings()

    try:
        result = await asyncio.to_thread(
            gcs.publish,
            path,
            bucket_name=settings.gcs_bucket,
            project=settings.gcp_project,
        )
    except gcs.PublishError as exc:
        # A publishing problem is a configuration or credentials problem, and
        # the message says which. It is shown to the user, not raised as a 500.
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=200)

    return JSONResponse({
        "ok": True,
        "url": result.url,
        "bucket": result.bucket,
        "kb": round(result.bytes_uploaded / 1024),
        "published": clocks.stamp(result.published_at),
    })


@app.get("/api/publish-config")
async def publish_config() -> JSONResponse:
    """Whether publishing is set up, so the UI can explain itself up front."""
    settings = _settings()
    return JSONResponse({
        "enabled": bool(settings.gcs_bucket),
        "bucket": settings.gcs_bucket,
        "project": settings.gcp_project,
    })


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((Path(__file__).parent / "index.html").read_text(encoding="utf-8"))


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    print(f"Guardians Report running at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
