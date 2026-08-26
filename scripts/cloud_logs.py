"""Read the Cloud Run service's logs using application-default credentials.

The gcloud CLI is a separate credential store from ADC, and on this machine it
is empty -- `configurations/config_default` is a zero-byte file and
`credentials.db` has no account in it, so `gcloud auth list` reports nothing
while the Python clients authenticate perfectly well. Rather than repair the
CLI, this goes straight at the Logging REST API with the credentials that
already work.

    python scripts/cloud_logs.py                  # last 30 minutes
    python scripts/cloud_logs.py --minutes 180
    python scripts/cloud_logs.py --grep Traceback

Note what this can and cannot show. A report build runs as a subprocess whose
output is piped back to the browser over SSE, so those lines never reach the
container's stdout and never reach here. What does reach here is anything the
container itself says: startup failures, unhandled exceptions in the app, and
the reason an instance died -- an out-of-memory kill among them, which is
invisible from the browser because it takes the event stream down with it.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

ENDPOINT = "https://logging.googleapis.com/v2/entries:list"


def _token_and_project() -> tuple[str, str]:
    import google.auth
    import google.auth.transport.requests as tr

    creds, project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/logging.read"])
    creds.refresh(tr.Request())
    # ADC from `gcloud auth application-default login` carries no project of
    # its own; the quota project it was granted against is the one we want.
    project = project or getattr(creds, "quota_project_id", None)
    if not project:
        raise SystemExit(
            "No project on the credentials. Pass --project explicitly.")
    return creds.token, project


def fetch(minutes: int, service: str, project: str | None,
          severity: str | None, limit: int) -> list[dict]:
    token, found = _token_and_project()
    project = project or found
    since = (datetime.now(timezone.utc)
             - timedelta(minutes=minutes)).isoformat().replace("+00:00", "Z")

    terms = [
        'resource.type="cloud_run_revision"',
        f'resource.labels.service_name="{service}"',
        f'timestamp>="{since}"',
    ]
    if severity:
        terms.append(f'severity>={severity}')

    body = json.dumps({
        "resourceNames": [f"projects/{project}"],
        "filter": " AND ".join(terms),
        "orderBy": "timestamp desc",
        "pageSize": limit,
    }).encode()

    request = urllib.request.Request(
        ENDPOINT, data=body, method="POST",
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json",
                 "x-goog-user-project": project})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:600]
        raise SystemExit(f"Logging API {exc.code}: {detail}") from exc

    # Oldest first, which is the order they happened in and the order anybody
    # reading a failure wants them.
    return list(reversed(payload.get("entries", [])))


def line_of(entry: dict) -> str:
    stamp = (entry.get("timestamp") or "")[11:19]
    severity = (entry.get("severity") or "DEFAULT")[:7]
    text = entry.get("textPayload")
    if text is None:
        payload = entry.get("jsonPayload") or entry.get("protoPayload") or {}
        text = payload.get("message") or json.dumps(payload)[:400]
    return f"{stamp}  {severity:<7}  {str(text).rstrip()}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--service", default="guards-report")
    parser.add_argument("--project", default=None)
    parser.add_argument("--severity", default=None,
                        help="e.g. WARNING, ERROR")
    parser.add_argument("--grep", default=None)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    entries = fetch(args.minutes, args.service, args.project,
                    args.severity, args.limit)
    if not entries:
        print(f"(nothing in the last {args.minutes} minutes)")
        return 0

    shown = 0
    for entry in entries:
        line = line_of(entry)
        if args.grep and args.grep.lower() not in line.lower():
            continue
        print(line)
        shown += 1
    if args.grep and not shown:
        print(f"({len(entries)} entries, none matching {args.grep!r})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
