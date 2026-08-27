"""Publish a finished report to Cloud Storage and hand back a link.

A report is a single self-contained HTML file, which makes sharing it a copy
rather than a deployment: upload the file, make it readable, return its URL.
Nothing about the report changes when it is published.

Publishing is entirely optional. The report is complete on disk, and every
failure here -- no bucket configured, no credentials, no network -- is reported
as a message rather than raised into the build. Losing a link is a nuisance;
losing a report that took minutes to build is not acceptable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# A report is *republished* to the same URL whenever it is rebuilt -- a lineup
# is posted, a bug is fixed -- so it is not immutable and must not be cached as
# though it were. An hour of caching meant a rebuilt report kept showing the old
# one, with nothing to suggest the link was stale.
#
# "no-cache" still allows the browser to store it; it just has to revalidate
# first. Combined with the ETag that Cloud Storage sets, an unchanged report
# comes back as a 304 with no body, so repeat views stay fast.
CACHE_CONTROL = "no-cache"


class PublishError(RuntimeError):
    """Publishing failed. Carries a message meant to be shown to a person."""


@dataclass
class Published:
    """Where a report ended up, and how to reach it."""

    url: str
    bucket: str
    object_name: str
    bytes_uploaded: int
    published_at: datetime

    @property
    def filename(self) -> str:
        return self.object_name.rsplit("/", 1)[-1]


def _client(project: str):
    try:
        from google.cloud import storage
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise PublishError(
            "google-cloud-storage is not installed; run "
            "'pip install google-cloud-storage'"
        ) from exc

    try:
        return storage.Client(project=project)
    except Exception as exc:
        raise PublishError(
            "could not authenticate to Google Cloud. Run "
            "'gcloud auth application-default login' once, then try again "
            f"({type(exc).__name__})"
        ) from exc


def object_name_for(path: Path) -> str:
    """The stable object path for a report file.

    Keyed on the report's own filename, which already carries the date and the
    matchup. Re-publishing the same day overwrites in place, so the link a
    person was given keeps working and shows the newest build.
    """
    return f"reports/{path.name}"


def publish(path: Path, *, bucket_name: str, project: str) -> Published:
    """Upload one report and return its public URL."""
    if not bucket_name:
        raise PublishError(
            "no bucket configured. Set GCS_BUCKET in .env to publish reports."
        )
    if not path.is_file():
        raise PublishError(f"report not found: {path}")

    client = _client(project)

    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name_for(path))
        blob.cache_control = CACHE_CONTROL
        # Charset matters: the report carries accented player names, and a
        # browser told only "text/html" may decode them as Latin-1.
        blob.upload_from_filename(path, content_type="text/html; charset=utf-8")
    except Exception as exc:
        raise PublishError(f"upload failed ({type(exc).__name__}: {exc})") from exc

    _make_readable(blob)

    # The preview card travels with the report. `og:image` points at it, and an
    # unfurler given an image URL that 404s renders a broken thumbnail -- worse
    # than the text card it was meant to improve on. Publishing the pair
    # together is what keeps that from happening.
    share = path.with_name(path.stem + "-share.html")
    if share.is_file():
        try:
            page = bucket.blob(object_name_for(share))
            page.cache_control = CACHE_CONTROL
            page.upload_from_filename(
                share, content_type="text/html; charset=utf-8")
            _make_readable(page)
        except Exception:  # noqa: BLE001
            pass

    card = path.with_suffix(".png")
    if card.is_file():
        try:
            image = bucket.blob(object_name_for(card))
            image.cache_control = CACHE_CONTROL
            image.upload_from_filename(card, content_type="image/png")
            _make_readable(image)
        except Exception:  # noqa: BLE001 -- a missing picture is not a failed publish
            pass

    return Published(
        url=f"https://storage.googleapis.com/{bucket_name}/{blob.name}",
        bucket=bucket_name,
        object_name=blob.name,
        bytes_uploaded=path.stat().st_size,
        published_at=datetime.now(timezone.utc),
    )


def _make_readable(blob) -> None:
    """Grant public read on the object, tolerating uniform bucket access.

    A bucket with uniform bucket-level access refuses per-object ACLs outright.
    That is not an error: it means access is managed on the bucket, and the
    upload has already succeeded. Treating the refusal as fatal would fail a
    publish that actually worked.
    """
    try:
        blob.make_public()
    except Exception:
        return


def ensure_bucket_public(bucket_name: str, *, project: str) -> str:
    """Grant public read on the whole bucket, for first-time setup.

    Separate from `publish` because it changes who can see the bucket, which
    is a decision rather than a side effect of building a report.
    """
    client = _client(project)
    try:
        bucket = client.bucket(bucket_name)
        policy = bucket.get_iam_policy(requested_policy_version=3)
        policy.bindings.append(
            {"role": "roles/storage.objectViewer", "members": {"allUsers"}}
        )
        bucket.set_iam_policy(policy)
    except Exception as exc:
        raise PublishError(
            f"could not make {bucket_name} public ({type(exc).__name__}: {exc})"
        ) from exc
    return f"https://storage.googleapis.com/{bucket_name}/"
