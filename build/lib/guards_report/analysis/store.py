"""Persist written analyses so they are generated once, not once per render.

Notes are keyed by subject and by a fingerprint of the figures behind them, so
a stored note is reused only while the numbers it describes are unchanged. That
makes re-rendering free and makes a repeat run of an unchanged day free, which
matters because the Agent SDK credit is a fixed monthly allowance rather than a
metered balance.

Local JSON is the source of truth for the cache; BigQuery gets the same rows
for history and auditing. The report must be producible with no cloud
dependency, so a BigQuery failure never blocks a run.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from guards_report.analysis.agent import Analysis
from guards_report.analysis.verify import Verification


def _to_row(analysis: Analysis) -> dict[str, Any]:
    return {
        "subject_id": analysis.subject_id,
        "kind": analysis.kind,
        "label": analysis.label,
        "fingerprint": analysis.fingerprint,
        "text": analysis.text,
        "generated_at": analysis.generated_at.isoformat(),
        "model": analysis.model,
        "cost_usd": analysis.cost_usd,
        "figures_checked": analysis.verification.checked,
        "figures_unverified": analysis.verification.unverified,
        # Tokens are the meter that actually applies on a subscription, so they
        # are persisted alongside the note rather than discarded with the run.
        "input_tokens": (analysis.usage or {}).get("input_tokens"),
        "output_tokens": (analysis.usage or {}).get("output_tokens"),
        "cache_read_tokens": (analysis.usage or {}).get("cache_read_tokens"),
        "cache_write_tokens": (analysis.usage or {}).get("cache_write_tokens"),
        "batched": (analysis.usage or {}).get("batched"),
    }


def _from_row(row: dict[str, Any]) -> Analysis:
    verification = Verification(
        checked=int(row.get("figures_checked") or 0),
        unverified=list(row.get("figures_unverified") or []),
    )
    return Analysis(
        subject_id=row["subject_id"],
        kind=row["kind"],
        label=row["label"],
        fingerprint=row["fingerprint"],
        text=row["text"],
        verification=verification,
        generated_at=datetime.fromisoformat(row["generated_at"]),
        model=row.get("model", "unknown"),
        cost_usd=row.get("cost_usd"),
        usage={
            k: row.get(k) for k in (
                "input_tokens", "output_tokens",
                "cache_read_tokens", "cache_write_tokens", "batched",
            )
        },
    )


class AnalysisStore:
    """Cache of generated notes, one file per game date."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, game_pk: int) -> Path:
        return self.root / f"game-{game_pk}.json"

    def load(self, game_pk: int) -> dict[str, Analysis]:
        """Stored notes keyed as `subject_id:fingerprint`.

        A corrupt or partially written cache file is treated as an empty cache
        rather than an error -- regenerating costs a little credit, while
        failing the run costs the whole report.
        """
        path = self._path(game_pk)
        if not path.exists():
            return {}
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

        cache: dict[str, Analysis] = {}
        for row in rows:
            try:
                analysis = _from_row(row)
            except (KeyError, ValueError):
                continue
            cache[f"{analysis.subject_id}:{analysis.fingerprint}"] = analysis
        return cache

    def save(self, game_pk: int, analyses: list[Analysis]) -> Path:
        """Write the current set of notes, merged over anything already stored."""
        existing = self.load(game_pk)
        for analysis in analyses:
            existing[f"{analysis.subject_id}:{analysis.fingerprint}"] = analysis

        path = self._path(game_pk)
        rows = [_to_row(a) for a in existing.values()]
        # Write then rename, so an interrupted run cannot leave a truncated
        # file that would read back as an empty cache.
        tmp = path.with_suffix(".json.partial")
        tmp.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        tmp.replace(path)
        return path


def summarize(analyses: list[Analysis]) -> dict[str, Any]:
    """Run-level totals for reporting and cost telemetry."""
    generated = [a for a in analyses if not a.from_cache]
    costs = [a.cost_usd for a in generated if a.cost_usd is not None]
    unverified = [a for a in analyses if not a.verification.ok]

    # Everything the call actually moved, cache included -- reading only
    # input+output reported about half the real volume.
    tokens = sum(
        (a.usage or {}).get(field) or 0
        for a in generated
        for field in (
            "input_tokens", "output_tokens",
            "cache_read_tokens", "cache_write_tokens",
        )
    )

    return {
        "total": len(analyses),
        "tokens": tokens or None,
        "generated": len(generated),
        "from_cache": len(analyses) - len(generated),
        "cost_usd": round(sum(costs), 4) if costs else None,
        "with_unverified_figures": len(unverified),
        "unverified_subjects": [a.label for a in unverified],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
