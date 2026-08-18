"""Command line entry point.

    guards-report build [--date today|YYYY-MM-DD] [--no-store]
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


def cmd_build(args: argparse.Namespace) -> int:
    from guards_report.ingest.preview import build_preview
    from guards_report.report.render_html import render

    settings = load_settings()
    on = _parse_date(args.date)

    print(f"Building preview for {on.isoformat()} ...", file=sys.stderr)
    try:
        bundle = build_preview(settings, on=on, team_id=args.team)
    except LookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="guards-report")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build a game preview")
    build.add_argument("--date", default="today", help="today|tomorrow|YYYY-MM-DD")
    build.add_argument("--team", type=int, default=CLEVELAND_GUARDIANS_TEAM_ID)
    build.add_argument(
        "--no-store", action="store_true", help="skip BigQuery persistence"
    )
    build.set_defaults(func=cmd_build)

    setup = sub.add_parser("setup-bq", help="create dataset and tables")
    setup.set_defaults(func=cmd_setup_bq)

    cost = sub.add_parser("cost", help="report storage and query cost")
    cost.set_defaults(func=cmd_cost)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
