"""Command line entry point.

    guards-report build [--date today|YYYY-MM-DD] [--no-store]
    guards-report serve [--port 8765]
    guards-report setup-bq
    guards-report cost
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from guards_report.config import CLEVELAND_GUARDIANS_TEAM_ID, load_settings


def _parse_date(text: str) -> date:
    if text == "today":
        return date.today()
    if text == "tomorrow":
        return date.today() + timedelta(days=1)
    if text == "yesterday":
        return date.today() - timedelta(days=1)
    return date.fromisoformat(text)


def attach_computed_analysis(bundle, settings, *, on) -> dict:
    """Fill the analysis slots from computed findings rather than a model.

    Produces the same shape the template already reads, so nothing downstream
    changes: one short piece of prose per subject, keyed by the same subject id.

    The verification badge is different in kind. A model's figures had to be
    checked back against the source because it might have invented one; these
    figures *are* the source, arithmetic applied to the pitch corpus, so there is
    nothing to verify after the fact. The badge says computed instead.
    """
    from dataclasses import dataclass, field

    from guards_report.insight import card as insight_card

    @dataclass
    class _Verification:
        ok: bool = True
        checked: int = 0
        unverified: list = field(default_factory=list)
        summary: str = ""

    @dataclass
    class _Computed:
        subject_id: str
        text: str
        findings: int = 0
        verification: _Verification = field(default_factory=_Verification)
        # The three-part note, when the player qualified for one. The template
        # leads with these and keeps `text` beneath as the supporting detail --
        # the evaluator sentence says what is unusual about him, which is worth
        # having once the reader knows who he is.
        parts: list = field(default_factory=list)
        tier: str = ""
        labels: list = field(default_factory=list)

    try:
        analysis = insight_card.analyse(
            bundle, pitch_dir=settings.raw_archive_dir.parent / "pitches", on=on
        )
    except Exception as exc:  # noqa: BLE001 -- computed prose is additive
        print(f"  warning: computed analysis skipped ({exc})", file=sys.stderr)
        return {}

    entries = {}
    subjects = set(analysis.subjects) | set(analysis.dossiers)
    for player_id in subjects:
        text = analysis.subjects.get(player_id, "")
        found = analysis.findings.get(player_id, 0)
        note = analysis.dossiers.get(player_id)
        verification = _Verification(
            ok=True, checked=found,
            summary=f"{found} criteria computed from the pitch corpus; "
                    "no figure is model-generated",
        )
        for prefix in ("pitcher", "batter"):
            entries[f"{prefix}-{player_id}"] = _Computed(
                subject_id=f"{prefix}-{player_id}", text=text,
                findings=found, verification=verification,
                parts=list(note.parts) if note else [],
                tier=note.tier if note else "",
                labels=list(note.labels) if note else [],
            )

    if analysis.matchup:
        entries[f"game-{bundle.game_pk}"] = _Computed(
            subject_id=f"game-{bundle.game_pk}", text=analysis.matchup,
            verification=_Verification(
                ok=True, checked=6,
                summary="assembled from the fitted projections",
            ),
        )

    bundle.analyses = entries
    bundle.analysis_summary = {
        "source": "computed",
        "covered": analysis.covered,
        "attempted": analysis.attempted,
        "coverage": analysis.coverage,
        "seconds": round(analysis.seconds, 1),
        "warnings": len(analysis.warnings),
        "profiled": len(analysis.dossiers),
    }
    for warning in analysis.warnings[:3]:
        print(f"  warning: {warning}", file=sys.stderr)
    print(
        f"  computed analysis: {analysis.covered}/{analysis.attempted} subjects "
        f"({analysis.coverage:.0%}), {len(analysis.dossiers)} profiled, "
        f"in {analysis.seconds:.0f}s",
        file=sys.stderr,
    )
    return bundle.analysis_summary


def attach_analysis(bundle, settings, *, model: str) -> dict:
    """Generate written notes and attach them to a finished bundle.

    Deliberately fault-tolerant: the report is complete without prose, so an
    unauthenticated CLI or a failed call degrades to an un-annotated report
    with a clear message rather than losing the run.
    """
    from guards_report.analysis.agent import (
        AgentUnavailable,
        PartialResult,
        generate_sync,
    )
    from guards_report.analysis.digest import select_subjects
    from guards_report.analysis.store import AnalysisStore, summarize

    digests = select_subjects(bundle)
    store = AnalysisStore(settings.raw_archive_dir.parent / "analysis")
    cache = store.load(bundle.game_pk)

    print(f"  analysing {len(digests)} subjects ...", file=sys.stderr)

    def progress(done: int, total: int, label: str) -> None:
        print(f"    [{done}/{total}] {label}", file=sys.stderr)

    partial = False
    try:
        analyses = generate_sync(
            digests, model=model, cache=cache, on_progress=progress
        )
    except PartialResult as exc:
        # The subscription's usage window ran out mid-run. Everything already
        # generated is kept and saved, so a later run resumes from the cache
        # instead of paying for these notes twice.
        analyses, partial = exc.analyses, True
        print(
            f"  stopped: {exc.reason}. Keeping {len(analyses)} of "
            f"{len(digests)} notes; rerun later to finish the rest.",
            file=sys.stderr,
        )
    except AgentUnavailable as exc:
        print(
            f"  warning: analysis skipped ({exc})\n"
            f"  the report is complete without it; run "
            f'"claude login" to enable written analysis',
            file=sys.stderr,
        )
        return {}

    store.save(bundle.game_pk, analyses)
    bundle.analyses = {a.subject_id: a for a in analyses}
    bundle.analysis_summary = summarize(analyses)

    s = bundle.analysis_summary
    # Tokens, not dollars: with subscription auth nothing here is billed to a
    # card, and reporting a dollar figure invites exactly the wrong decision.
    # The equivalent is kept in parentheses purely for scale.
    detail = f"  analysis: {s['generated']} generated, {s['from_cache']} cached"
    if s.get("tokens"):
        detail += f" · {s['tokens']:,} tokens"
    if s.get("cost_usd"):
        detail += f" (~${s['cost_usd']} at API rates; billed to your subscription)"
    if partial:
        detail += " · INCOMPLETE"
    print(detail, file=sys.stderr)
    if s["with_unverified_figures"]:
        print(
            f"  warning: {s['with_unverified_figures']} notes cite figures not "
            f"found in their source data: {', '.join(s['unverified_subjects'][:3])}",
            file=sys.stderr,
        )
    return s


def cmd_build(args: argparse.Namespace) -> int:
    from guards_report.ingest.preview import build_preview
    from guards_report.report.render_html import render

    settings = load_settings()
    on = _parse_date(args.date)

    print(f"Building preview for {on.isoformat()} ...", file=sys.stderr)
    try:
        bundle = build_preview(settings, on=on, team_id=args.team, include_statcast=not args.no_statcast)
    except LookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Written analysis is a separate, optional stage that runs only after the
    # numbers are final. A failure here never costs the report.
    if args.with_analysis:
        attach_analysis(bundle, settings, model=args.model)
    elif not args.no_analysis:
        # Computed by default. The findings are arithmetic on the pitch corpus,
        # so this costs no tokens, produces the same words for the same game
        # every time, and every figure traces to a computation rather than
        # needing to be checked back against one.
        attach_computed_analysis(bundle, settings, on=on)

    path = render(bundle, output_dir=settings.output_dir)

    print(
        f"\n{bundle.away.name} at {bundle.home.name}"
        f"\n  {bundle.game_date} · {bundle.venue_name} · {bundle.status}",
        file=sys.stderr,
    )
    for team in (bundle.away, bundle.home):
        starter = next(
            (p.name for p in team.pitchers if p.is_probable_starter), "TBD"
        )
        with_zones = sum(1 for p in team.batters if p.zone_grids)
        print(
            f"  {team.abbreviation}: {starter}"
            f" · {len(team.pitchers)} pitchers, {len(team.batters)} batters"
            f" ({with_zones} with zone data)",
            file=sys.stderr,
        )
    print(
        f"  {len(bundle.provenance)} source requests"
        f" · FIP constant {bundle.league.fip_constant:.4f}"
        f" · lg {bundle.league_hitting.get('avg'):.3f}"
        f"/{bundle.league_hitting.get('obp'):.3f}"
        f"/{bundle.league_hitting.get('slg'):.3f}",
        file=sys.stderr,
    )

    if not args.no_store:
        from guards_report.store.bq import BigQueryStore

        try:
            store = BigQueryStore(settings)
            store.ensure_dataset()
            created = store.ensure_tables()
            if created:
                print(f"  created BigQuery tables: {', '.join(created)}", file=sys.stderr)
        except Exception as exc:
            # Persistence is not required to produce a report. Say so plainly
            # rather than discarding a report that was built successfully.
            print(
                f"  warning: BigQuery persistence skipped ({type(exc).__name__}: {exc})",
                file=sys.stderr,
            )

    print(path)
    return 0


def cmd_setup_bq(args: argparse.Namespace) -> int:
    from guards_report.store.bq import BigQueryStore

    settings = load_settings()
    store = BigQueryStore(settings)
    dataset = store.ensure_dataset()
    created = store.ensure_tables()
    print(f"dataset ready: {dataset.full_dataset_id}")
    print(f"tables created: {', '.join(created) if created else '(all present)'}")
    return 0


def cmd_cost(args: argparse.Namespace) -> int:
    from guards_report.store.bq import BigQueryStore

    settings = load_settings()
    store = BigQueryStore(settings)

    print("storage by table:")
    total = 0.0
    for row in store.storage_summary():
        total += float(row.get("mib") or 0)
        print(f"  {row['table_name']:<28} {row['total_rows']:>10,} rows  {row['mib']:>8} MiB")
    print(f"  {'TOTAL':<28} {'':>10}       {total:>8.3f} MiB")
    print("\n  free tier is 10 GiB storage + 1 TiB queries per month")

    print("\nquery cost, last 7 days:")
    for row in store.recent_job_costs(days=7):
        print(
            f"  {row['day']}  {row['jobs']:>4} jobs  "
            f"{(row['bytes_billed'] or 0) / 1048576:>10.2f} MiB billed  "
            f"~${row['est_usd_at_on_demand']}"
        )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from guards_report.app.main import serve

    serve(host=args.host, port=args.port)
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    from pathlib import Path

    from guards_report.publish import gcs

    settings = load_settings()
    path = Path(args.path)
    if not path.is_absolute():
        candidate = settings.output_dir / path.name
        if candidate.is_file():
            path = candidate

    try:
        result = gcs.publish(
            path, bucket_name=settings.gcs_bucket, project=settings.gcp_project
        )
    except gcs.PublishError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(result.url)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="guards-report")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build a game preview")
    build.add_argument("--date", default="today", help="today|tomorrow|YYYY-MM-DD")
    build.add_argument("--team", type=int, default=CLEVELAND_GUARDIANS_TEAM_ID)
    build.add_argument(
        "--no-store", action="store_true", help="skip BigQuery persistence"
    )
    build.add_argument(
        "--no-statcast",
        action="store_true",
        help=(
            "skip per-player pitch-level fetches. Drops the handedness-split "
            "zone maps and spray charts, but cuts a run from minutes to seconds "
            "-- useful when iterating on layout."
        ),
    )
    build.add_argument(
        "--no-analysis", action="store_true",
        help="skip the computed written analysis, leaving the slots empty",
    )
    build.add_argument(
        "--with-analysis",
        action="store_true",
        help=(
            "generate written analysis for the matchup, both starters and both "
            "lineups via the Claude Agent SDK (uses your subscription's Agent "
            "SDK credit; requires `claude login`)"
        ),
    )
    build.add_argument(
        "--model",
        default="sonnet",
        help="model for written analysis: sonnet (default), opus, or haiku",
    )
    build.set_defaults(func=cmd_build)

    serve = sub.add_parser("serve", help="run the local web app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(func=cmd_serve)

    publish = sub.add_parser("publish", help="upload a report and print its link")
    publish.add_argument("path", help="report file, or just its name in out/")
    publish.set_defaults(func=cmd_publish)

    setup = sub.add_parser("setup-bq", help="create dataset and tables")
    setup.set_defaults(func=cmd_setup_bq)

    cost = sub.add_parser("cost", help="report storage and query cost")
    cost.set_defaults(func=cmd_cost)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
