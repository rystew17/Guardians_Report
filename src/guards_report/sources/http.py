"""Shared HTTP layer: rate limiting, retries, and payload archiving.

Every outbound request in this project goes through `fetch`. That is what makes
the provenance guarantee hold: there is exactly one place where bytes enter the
system, so there is exactly one place that has to record where they came from.

Each call returns a FetchResult carrying the response body plus the metadata
needed to prove where it came from and to reproduce it later. The raw bytes are
written to a gzipped file on disk; BigQuery stores only the metadata and the
sha256, which keeps the warehouse tiny while preserving full auditability.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

from guards_report.config import (
    MAX_RETRIES,
    REQUEST_TIMEOUT_SECONDS,
    REQUESTS_PER_SECOND,
    USER_AGENT,
)


@dataclass(frozen=True)
class FetchResult:
    """One HTTP response plus everything needed to audit it."""

    url: str
    source: str
    fetched_at: datetime
    http_status: int
    content: bytes
    sha256: str
    archive_path: Path
    elapsed_seconds: float
    from_cache: bool = False

    @property
    def byte_count(self) -> int:
        return len(self.content)

    def json(self) -> Any:
        return json.loads(self.content)

    def text(self) -> str:
        return self.content.decode("utf-8-sig")

    def provenance(self) -> dict[str, Any]:
        """The row shape written to BigQuery's raw_api_call table."""
        return {
            "url": self.url,
            "source": self.source,
            "fetched_at": self.fetched_at.isoformat(),
            "http_status": self.http_status,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "archive_path": str(self.archive_path),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


class _RateLimiter:
    """Process-wide minimum spacing between requests.

    Deliberately conservative. These are free, unauthenticated endpoints run by
    someone else, and the project stops working if we get blocked.
    """

    def __init__(self, per_second: float) -> None:
        self._min_interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()


_limiter = _RateLimiter(REQUESTS_PER_SECOND)
_session: requests.Session | None = None
_session_lock = threading.Lock()


def _get_session() -> requests.Session:
    global _session
    with _session_lock:
        if _session is None:
            _session = requests.Session()
            _session.headers.update(
                {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
            )
        return _session


@dataclass
class Archiver:
    """Writes raw response bodies to gzipped files, content-addressed.

    Content addressing means an unchanged payload fetched twice occupies one
    file, and a payload can never be silently altered after the fact without
    the hash changing.
    """

    root: Path
    written: list[FetchResult] = field(default_factory=list)

    def path_for(self, source: str, digest: str) -> Path:
        # Shard by the first two hex characters so no directory grows without
        # bound over a long season.
        return self.root / source / digest[:2] / f"{digest}.gz"

    def store(self, source: str, content: bytes) -> tuple[str, Path]:
        digest = hashlib.sha256(content).hexdigest()
        path = self.path_for(source, digest)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temp name then rename, so an interrupted run can never
            # leave a truncated file sitting at a valid content hash.
            tmp = path.with_suffix(".gz.partial")
            with gzip.open(tmp, "wb") as handle:
                handle.write(content)
            tmp.replace(path)
        return digest, path


class RetryableStatus(Exception):
    """Raised internally to trigger a retry on a transient HTTP status."""


def fetch(
    url: str,
    *,
    source: str,
    archiver: Archiver,
    params: dict[str, Any] | None = None,
) -> FetchResult:
    """Fetch a URL, archive the body, and return it with provenance attached.

    Retries on connection errors and on 429/5xx with exponential backoff. A 4xx
    other than 429 is not retried -- that is a bug in our request, and retrying
    a malformed request just annoys the server.
    """
    full_url = f"{url}?{urlencode(params)}" if params else url
    session = _get_session()

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        if attempt:
            # 1s, 2s, 4s, ... plus the rate limiter's own spacing.
            time.sleep(2.0 ** (attempt - 1))

        _limiter.wait()
        started = time.monotonic()
        try:
            response = session.get(full_url, timeout=REQUEST_TIMEOUT_SECONDS)
            elapsed = time.monotonic() - started

            if response.status_code == 429 or response.status_code >= 500:
                last_error = RetryableStatus(
                    f"HTTP {response.status_code} from {full_url}"
                )
                continue

            response.raise_for_status()

            digest, path = archiver.store(source, response.content)
            result = FetchResult(
                url=full_url,
                source=source,
                fetched_at=datetime.now(timezone.utc),
                http_status=response.status_code,
                content=response.content,
                sha256=digest,
                archive_path=path,
                elapsed_seconds=elapsed,
            )
            archiver.written.append(result)
            return result

        except requests.RequestException as exc:
            last_error = exc

    raise RuntimeError(
        f"giving up on {full_url} after {MAX_RETRIES} attempts"
    ) from last_error
