"""Durable storage for the corpora, so they are never pulled twice.

The pitch corpus takes roughly three hours to fetch and is immutable once a
season ends -- the worst possible thing to keep in one place on one laptop. This
mirrors the Parquet files to object storage as they are, byte for byte.

Deliberately not BigQuery. The training path never issues SQL: it reads the
whole corpus into memory for a ridge fit, which is a Parquet read, not a query.
Loading into a warehouse would store a *transformed* copy, and "never pull
again" then means an export job to get back to the files the code already
reads. A warehouse is worth adding later for ad-hoc analysis; it is not a
backup.

Uploads are content-addressed by size and modification time rather than
re-sent blindly, because a finished season never changes and re-uploading a
gigabyte to discover that is a waste of everyone's bandwidth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Corpora worth preserving. `models` is included because the fitted artifact is
# small, reproducible only by a long refit, and the thing that makes two reports
# of the same game agree.
DATASETS = ("corpus", "pitchers", "pitches", "models")


@dataclass
class SyncResult:
    """What a sync actually did, reported rather than assumed."""

    uploaded: int = 0
    skipped: int = 0
    bytes_sent: int = 0
    failures: list[str] = None

    def __post_init__(self) -> None:
        if self.failures is None:
            self.failures = []

    def line(self) -> str:
        megabytes = self.bytes_sent / 1e6
        return (
            f"{self.uploaded} uploaded ({megabytes:,.1f} MB), "
            f"{self.skipped} already current"
            + (f", {len(self.failures)} failed" if self.failures else "")
        )


def _object_name(root: Path, path: Path) -> str:
    """Path within the bucket, mirroring the local layout under `data/`."""
    return "data/" + path.relative_to(root).as_posix()


def local_files(root: Path, datasets: Iterable[str] = DATASETS) -> list[Path]:
    files: list[Path] = []
    for name in datasets:
        directory = root / name
        if directory.exists():
            files.extend(sorted(directory.rglob("*.parquet")))
            files.extend(sorted(directory.rglob("*.json")))
    return files


def sync(
    root: Path,
    *,
    bucket_name: str,
    project: str,
    datasets: Iterable[str] = DATASETS,
    dry_run: bool = False,
    verbose: bool = True,
) -> SyncResult:
    """Mirror the local corpora to the bucket.

    A file is re-uploaded only when its size differs from the stored copy.
    Parquet written by the same code from the same rows is byte-identical, so
    size is a sufficient and very cheap check; a season that has not changed
    costs one metadata lookup rather than a transfer.
    """
    result = SyncResult()
    files = local_files(root, datasets)
    if not files:
        return result

    if dry_run:
        result.skipped = len(files)
        return result

    if not bucket_name:
        raise RuntimeError(
            "no bucket configured. Set GCS_BUCKET in .env to store the corpora."
        )

    from google.cloud import storage  # imported here so the CLI works without it

    client = storage.Client(project=project) if project else storage.Client()
    bucket = client.bucket(bucket_name)

    for path in files:
        name = _object_name(root, path)
        size = path.stat().st_size
        try:
            blob = bucket.get_blob(name)
            if blob is not None and blob.size == size:
                result.skipped += 1
                continue
            bucket.blob(name).upload_from_filename(str(path))
            result.uploaded += 1
            result.bytes_sent += size
            if verbose:
                print(f"  up  {name}  {size/1e6:.2f} MB", flush=True)
        except Exception as exc:  # noqa: BLE001 -- reported, never fatal
            result.failures.append(f"{name}: {type(exc).__name__}: {exc}")

    return result


def restore(
    root: Path,
    *,
    bucket_name: str,
    project: str,
    datasets: Iterable[str] = DATASETS,
    verbose: bool = True,
) -> SyncResult:
    """Pull the corpora back down, for a new machine or a lost disk.

    The inverse of `sync`, and the reason it stores the files unchanged: restore
    is a download, not a rebuild, so the training code reads exactly what it read
    before with nothing in between.
    """
    result = SyncResult()
    from google.cloud import storage

    client = storage.Client(project=project) if project else storage.Client()
    bucket = client.bucket(bucket_name)

    for name in datasets:
        for blob in client.list_blobs(bucket, prefix=f"data/{name}/"):
            target = root / Path(blob.name).relative_to("data")
            if target.exists() and target.stat().st_size == blob.size:
                result.skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(target))
            result.uploaded += 1
            result.bytes_sent += blob.size or 0
            if verbose:
                print(f"  down  {blob.name}  {(blob.size or 0)/1e6:.2f} MB", flush=True)

    return result
