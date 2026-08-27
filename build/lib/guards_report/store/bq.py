"""BigQuery access, with cost control built into the client rather than bolted on.

Three rules are enforced structurally here, so that a mistake becomes an error
instead of a bill:

1. Every query job carries `maximum_bytes_billed`. A runaway query fails.
2. Ingestion uses batch load jobs, which are free. The streaming API is never
   used -- that is the path that actually costs money per byte.
3. Writes are MERGE upserts on a natural key, so re-running a date overwrites
   rather than appends. Idempotency is what makes a daily job safe to retry.

At the measured data volume (~40 MB per league-wide season) this all sits
inside BigQuery's free tier. The guardrails exist because "should be free" and
"cannot cost anything" are different guarantees.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any

from google.api_core.exceptions import NotFound
from google.cloud import bigquery

from guards_report.config import Settings
from guards_report.store.schemas import TABLES, TableSpec


class BigQueryStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = bigquery.Client(
            project=settings.gcp_project, location=settings.bq_location
        )

    # -- setup --------------------------------------------------------------

    def ensure_dataset(self) -> bigquery.Dataset:
        dataset_id = self.settings.dataset_ref
        try:
            return self.client.get_dataset(dataset_id)
        except NotFound:
            dataset = bigquery.Dataset(dataset_id)
            dataset.location = self.settings.bq_location
            dataset.description = (
                "Cleveland Guardians daily scouting report. Stores immutable "
                "game logs plus source snapshots; recent-form windows are "
                "views, not tables."
            )
            return self.client.create_dataset(dataset)

    def ensure_tables(self) -> list[str]:
        """Create any missing tables. Existing tables are left untouched."""
        created: list[str] = []
        for spec in TABLES.values():
            table_id = self.settings.table(spec.name)
            try:
                self.client.get_table(table_id)
            except NotFound:
                self.client.create_table(spec.to_table(table_id))
                created.append(spec.name)
        return created

    # -- reads --------------------------------------------------------------

    def query(
        self, sql: str, params: Sequence[bigquery.ScalarQueryParameter] | None = None
    ) -> bigquery.table.RowIterator:
        """Run a query with a hard cap on bytes billed.

        `maximum_bytes_billed` makes BigQuery reject the job outright if it
        would scan more than the configured limit, rather than running it and
        charging for it.
        """
        job_config = bigquery.QueryJobConfig(
            maximum_bytes_billed=self.settings.bq_max_bytes_billed,
            query_parameters=list(params or []),
            use_legacy_sql=False,
        )
        return self.client.query(sql, job_config=job_config).result()

    def dry_run_bytes(self, sql: str) -> int:
        """Bytes a query *would* scan, without running it. Always free."""
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        job = self.client.query(sql, job_config=job_config)
        return job.total_bytes_processed or 0

    def max_game_date(self, table: str, *, player_ids: Iterable[int]) -> dict[int, Any]:
        """Latest stored game date per player, to drive incremental ingestion.

        Without this a daily run would re-fetch every player's whole season.
        With it, a run fetches only games it does not already have.
        """
        ids = list(player_ids)
        if not ids:
            return {}
        sql = f"""
            SELECT player_id, MAX(game_date) AS last_date
            FROM `{self.settings.table(table)}`
            WHERE game_date > DATE_SUB(CURRENT_DATE(), INTERVAL 400 DAY)
              AND player_id IN UNNEST(@player_ids)
            GROUP BY player_id
        """
        rows = self.query(
            sql,
            [bigquery.ArrayQueryParameter("player_ids", "INT64", ids)],
        )
        return {row["player_id"]: row["last_date"] for row in rows}

    # -- writes -------------------------------------------------------------

    def load_rows(self, spec: TableSpec, rows: Sequence[dict[str, Any]]) -> int:
        """Batch-load rows into a temp table, then MERGE into the target.

        Uses a load job rather than `insert_rows_json`: load jobs are free,
        streaming inserts are billed per byte. For a dataset this size the
        difference is small in absolute terms, but the streaming path is the
        one that turns a retry loop into a real bill.
        """
        if not rows:
            return 0

        target_id = self.settings.table(spec.name)
        if not spec.key_fields:
            self._load(target_id, spec, rows, disposition="WRITE_APPEND")
            return len(rows)

        staging_id = f"{target_id}__staging"
        self._load(staging_id, spec, rows, disposition="WRITE_TRUNCATE")
        try:
            self._merge(target_id, staging_id, spec)
        finally:
            self.client.delete_table(staging_id, not_found_ok=True)
        return len(rows)

    def _load(
        self,
        table_id: str,
        spec: TableSpec,
        rows: Sequence[dict[str, Any]],
        *,
        disposition: str,
    ) -> None:
        job_config = bigquery.LoadJobConfig(
            schema=spec.schema,
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            write_disposition=disposition,
        )
        # Staging tables inherit partitioning but never the partition-filter
        # requirement, which would otherwise make the MERGE below fail.
        if disposition == "WRITE_APPEND" and spec.partition_field:
            job_config.time_partitioning = bigquery.TimePartitioning(
                type_=getattr(bigquery.TimePartitioningType, spec.partition_type),
                field=spec.partition_field,
            )

        payload = "\n".join(json.dumps(row, default=str) for row in rows)
        job = self.client.load_table_from_file(
            file_obj=_as_stream(payload),
            destination=table_id,
            job_config=job_config,
        )
        job.result()

    def _merge(self, target_id: str, staging_id: str, spec: TableSpec) -> None:
        on_clause = " AND ".join(
            f"T.{field} = S.{field}" for field in spec.key_fields
        )
        columns = [field.name for field in spec.schema]
        update_clause = ", ".join(
            f"T.{col} = S.{col}" for col in columns if col not in spec.key_fields
        )
        column_list = ", ".join(columns)
        values_list = ", ".join(f"S.{col}" for col in columns)

        sql = f"""
            MERGE `{target_id}` T
            USING `{staging_id}` S
            ON {on_clause}
            WHEN MATCHED THEN UPDATE SET {update_clause}
            WHEN NOT MATCHED THEN INSERT ({column_list}) VALUES ({values_list})
        """
        # DML against a require_partition_filter table needs the filter to be
        # satisfiable from the ON clause, which it is: the key includes the
        # partition column for every partitioned table with a natural key.
        job_config = bigquery.QueryJobConfig(
            maximum_bytes_billed=self.settings.bq_max_bytes_billed
        )
        self.client.query(sql, job_config=job_config).result()

    # -- cost reporting -----------------------------------------------------

    def storage_summary(self) -> list[dict[str, Any]]:
        """Per-table row counts and logical bytes, from table metadata.

        Reads INFORMATION_SCHEMA.TABLE_STORAGE, which is metadata rather than
        table data and therefore free to query.
        """
        sql = f"""
            SELECT table_name,
                   total_rows,
                   total_logical_bytes,
                   ROUND(total_logical_bytes / 1048576, 3) AS mib
            FROM `{self.settings.gcp_project}`.`region-{self.settings.bq_location.lower()}`.INFORMATION_SCHEMA.TABLE_STORAGE
            WHERE table_schema = @dataset
            ORDER BY total_logical_bytes DESC
        """
        rows = self.query(
            sql,
            [
                bigquery.ScalarQueryParameter(
                    "dataset", "STRING", self.settings.bq_dataset
                )
            ],
        )
        return [dict(row) for row in rows]

    def recent_job_costs(self, *, days: int = 7) -> list[dict[str, Any]]:
        """Bytes billed by this project's recent query jobs."""
        sql = f"""
            SELECT DATE(creation_time) AS day,
                   COUNT(*) AS jobs,
                   SUM(total_bytes_billed) AS bytes_billed,
                   ROUND(SUM(total_bytes_billed) / 1099511627776 * 6.25, 4)
                       AS est_usd_at_on_demand
            FROM `{self.settings.gcp_project}`.`region-{self.settings.bq_location.lower()}`.INFORMATION_SCHEMA.JOBS
            WHERE creation_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
              AND job_type = 'QUERY'
            GROUP BY day
            ORDER BY day DESC
        """
        rows = self.query(
            sql, [bigquery.ScalarQueryParameter("days", "INT64", days)]
        )
        return [dict(row) for row in rows]


def _as_stream(text: str):
    import io

    return io.BytesIO(text.encode("utf-8"))
