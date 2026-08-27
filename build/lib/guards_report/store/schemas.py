"""BigQuery table definitions.

Sizing drove the design here. Measured against real payloads, a full season of
league-wide data is roughly 40 MB, against BigQuery's 10 GiB free storage tier.
That is only true because of one rule:

    Store the atom, derive the rest.

`fact_player_game_log` is the atom. Game logs are immutable once a game is
final, and season / L5 / L15 / L30 are all deterministic aggregations of them,
so the windows are SQL views costing zero storage rather than daily snapshots.

The two things that are stored despite being derivable-looking are stored for
concrete reasons:
  * platoon splits -- a game log does not break plate appearances out by the
    handedness of the opposing pitcher, so these genuinely cannot be derived;
  * Savant leaderboards -- pre-aggregated by the source, and cheaper to keep
    than to recompute from pitch-level data we deliberately do not store.

Both are upserted in place with no daily history.
"""

from __future__ import annotations

from google.cloud import bigquery

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

RAW_API_CALL = [
    bigquery.SchemaField("fetched_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("source", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("url", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("http_status", "INT64", mode="REQUIRED"),
    # sha256 of the response body. The body itself lives in a gzipped file on
    # disk, not here -- storing payloads in the warehouse would cost ~2 GB a
    # season (the live game feed alone is ~800 KB per game) to hold blobs that
    # are never queried. The hash preserves auditability at ~250 bytes a row.
    bigquery.SchemaField("sha256", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("byte_count", "INT64"),
    bigquery.SchemaField("archive_path", "STRING"),
    bigquery.SchemaField("elapsed_seconds", "FLOAT64"),
    bigquery.SchemaField("run_id", "STRING"),
]

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

DIM_PLAYER = [
    bigquery.SchemaField("player_id", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("full_name", "STRING"),
    # Handedness drives the platoon matchups the report is built around.
    bigquery.SchemaField("bats", "STRING"),
    bigquery.SchemaField("throws", "STRING"),
    bigquery.SchemaField("primary_position", "STRING"),
    bigquery.SchemaField("position_type", "STRING"),
    bigquery.SchemaField("current_team_id", "INT64"),
    bigquery.SchemaField("jersey_number", "STRING"),
    bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
]

DIM_TEAM = [
    bigquery.SchemaField("team_id", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("name", "STRING"),
    bigquery.SchemaField("abbreviation", "STRING"),
    bigquery.SchemaField("venue_id", "INT64"),
    bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
]

DIM_VENUE = [
    bigquery.SchemaField("venue_id", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("name", "STRING"),
    bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
]

# The FIP constant is derived per season from actual league totals rather than
# hardcoded, because it drifts with the run environment. The inputs are kept
# alongside it so the report's audit appendix can show the derivation instead
# of asking the reader to trust a bare number.
DIM_LEAGUE_CONSTANT = [
    bigquery.SchemaField("season", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("fip_constant", "FLOAT64", mode="REQUIRED"),
    bigquery.SchemaField("league_era", "FLOAT64", mode="REQUIRED"),
    bigquery.SchemaField("lg_outs", "INT64"),
    bigquery.SchemaField("lg_earned_runs", "INT64"),
    bigquery.SchemaField("lg_home_runs", "INT64"),
    bigquery.SchemaField("lg_walks", "INT64"),
    bigquery.SchemaField("lg_hit_by_pitch", "INT64"),
    bigquery.SchemaField("lg_strikeouts", "INT64"),
    bigquery.SchemaField("lg_batters_faced", "INT64"),
    bigquery.SchemaField("computed_at", "TIMESTAMP", mode="REQUIRED"),
]

# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------

FACT_GAME_SCHEDULE = [
    bigquery.SchemaField("game_pk", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("game_date", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("game_datetime", "TIMESTAMP"),
    bigquery.SchemaField("season", "INT64"),
    bigquery.SchemaField("game_type", "STRING"),
    bigquery.SchemaField("status", "STRING"),
    bigquery.SchemaField("home_team_id", "INT64"),
    bigquery.SchemaField("away_team_id", "INT64"),
    bigquery.SchemaField("home_probable_pitcher_id", "INT64"),
    bigquery.SchemaField("away_probable_pitcher_id", "INT64"),
    bigquery.SchemaField("venue_id", "INT64"),
    bigquery.SchemaField("weather_condition", "STRING"),
    bigquery.SchemaField("weather_temp_f", "INT64"),
    bigquery.SchemaField("weather_wind", "STRING"),
    bigquery.SchemaField("home_plate_umpire", "STRING"),
    bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
]

# The atom. One row per player per game. Everything else is derived from here.
FACT_PLAYER_GAME_LOG = [
    bigquery.SchemaField("player_id", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("game_pk", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("game_date", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("stat_group", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("season", "INT64"),
    # Payload order from the source, which is chronological. This is the
    # correct tiebreaker for doubleheaders -- gamePk is not reliably ordered
    # within a date (observed 2026-07-28, where the higher id was game one).
    bigquery.SchemaField("sequence", "INT64"),
    bigquery.SchemaField("is_home", "BOOL"),
    bigquery.SchemaField("opponent_team_id", "INT64"),
    bigquery.SchemaField("team_id", "INT64"),
    # Shared counting stats
    bigquery.SchemaField("games_played", "INT64"),
    bigquery.SchemaField("plate_appearances", "INT64"),
    bigquery.SchemaField("at_bats", "INT64"),
    bigquery.SchemaField("runs", "INT64"),
    bigquery.SchemaField("hits", "INT64"),
    bigquery.SchemaField("doubles", "INT64"),
    bigquery.SchemaField("triples", "INT64"),
    bigquery.SchemaField("home_runs", "INT64"),
    bigquery.SchemaField("rbi", "INT64"),
    bigquery.SchemaField("walks", "INT64"),
    bigquery.SchemaField("intentional_walks", "INT64"),
    bigquery.SchemaField("strikeouts", "INT64"),
    bigquery.SchemaField("hit_by_pitch", "INT64"),
    bigquery.SchemaField("sac_flies", "INT64"),
    bigquery.SchemaField("sac_bunts", "INT64"),
    bigquery.SchemaField("stolen_bases", "INT64"),
    bigquery.SchemaField("caught_stealing", "INT64"),
    bigquery.SchemaField("total_bases", "INT64"),
    bigquery.SchemaField("ground_outs", "INT64"),
    bigquery.SchemaField("air_outs", "INT64"),
    bigquery.SchemaField("number_of_pitches", "INT64"),
    # Pitching-only. `outs` is stored directly rather than an innings string,
    # which keeps the "5.1 innings" notation out of the warehouse entirely.
    bigquery.SchemaField("games_started", "INT64"),
    bigquery.SchemaField("batters_faced", "INT64"),
    bigquery.SchemaField("outs", "INT64"),
    bigquery.SchemaField("earned_runs", "INT64"),
    bigquery.SchemaField("wins", "INT64"),
    bigquery.SchemaField("losses", "INT64"),
    bigquery.SchemaField("saves", "INT64"),
    bigquery.SchemaField("holds", "INT64"),
    bigquery.SchemaField("blown_saves", "INT64"),
    bigquery.SchemaField("pitches_thrown", "INT64"),
    bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
]

FACT_PLAYER_SPLIT = [
    bigquery.SchemaField("player_id", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("season", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("stat_group", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("split_code", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("split_description", "STRING"),
    bigquery.SchemaField("plate_appearances", "INT64"),
    bigquery.SchemaField("at_bats", "INT64"),
    bigquery.SchemaField("hits", "INT64"),
    bigquery.SchemaField("doubles", "INT64"),
    bigquery.SchemaField("triples", "INT64"),
    bigquery.SchemaField("home_runs", "INT64"),
    bigquery.SchemaField("walks", "INT64"),
    bigquery.SchemaField("strikeouts", "INT64"),
    bigquery.SchemaField("hit_by_pitch", "INT64"),
    bigquery.SchemaField("sac_flies", "INT64"),
    bigquery.SchemaField("total_bases", "INT64"),
    bigquery.SchemaField("avg", "STRING"),
    bigquery.SchemaField("obp", "STRING"),
    bigquery.SchemaField("slg", "STRING"),
    bigquery.SchemaField("ops", "STRING"),
    bigquery.SchemaField("as_of_date", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
]

# Savant metrics are stored as key/value pairs rather than fixed columns.
# Savant adds and renames leaderboard columns between seasons; a repeated
# key/value field absorbs that without a schema migration, and these tables are
# read by explicit key lookup rather than SELECT *.
FACT_SAVANT_LEADERBOARD = [
    bigquery.SchemaField("leaderboard", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("season", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("player_id", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("player_type", "STRING"),
    # Discriminator for leaderboards with several rows per player, such as the
    # arsenal board which emits one row per pitch type.
    bigquery.SchemaField("row_key", "STRING"),
    bigquery.SchemaField(
        "metrics",
        "RECORD",
        mode="REPEATED",
        fields=[
            bigquery.SchemaField("key", "STRING"),
            bigquery.SchemaField("value_num", "FLOAT64"),
            bigquery.SchemaField("value_str", "STRING"),
        ],
    ),
    bigquery.SchemaField("as_of_date", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
]

REPORT_RUN = [
    bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("generated_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("game_pk", "INT64"),
    bigquery.SchemaField("game_date", "DATE"),
    bigquery.SchemaField("git_sha", "STRING"),
    bigquery.SchemaField("api_call_count", "INT64"),
    bigquery.SchemaField("bytes_fetched", "INT64"),
    bigquery.SchemaField("output_path", "STRING"),
    bigquery.SchemaField("lineup_source", "STRING"),
    bigquery.SchemaField("notes", "STRING"),
]


# ---------------------------------------------------------------------------
# Table configuration
# ---------------------------------------------------------------------------


class TableSpec:
    """A table plus how it is partitioned, clustered, and keyed for upsert."""

    def __init__(
        self,
        name: str,
        schema: list[bigquery.SchemaField],
        *,
        partition_field: str | None = None,
        partition_type: str = "DAY",
        clustering: list[str] | None = None,
        key_fields: list[str] | None = None,
        require_partition_filter: bool = False,
    ) -> None:
        self.name = name
        self.schema = schema
        self.partition_field = partition_field
        self.partition_type = partition_type
        self.clustering = clustering or []
        # Natural key. MERGE on these makes re-running a date idempotent
        # instead of appending duplicate rows.
        self.key_fields = key_fields or []
        self.require_partition_filter = require_partition_filter

    def to_table(self, table_id: str) -> bigquery.Table:
        table = bigquery.Table(table_id, schema=self.schema)
        if self.partition_field:
            table.time_partitioning = bigquery.TimePartitioning(
                type_=getattr(
                    bigquery.TimePartitioningType, self.partition_type
                ),
                field=self.partition_field,
                # Makes BigQuery itself reject an unfiltered scan, so a cost
                # mistake becomes a query error rather than a bill.
                require_partition_filter=self.require_partition_filter,
            )
        if self.clustering:
            table.clustering_fields = self.clustering
        return table


TABLES: dict[str, TableSpec] = {
    "raw_api_call": TableSpec(
        "raw_api_call",
        RAW_API_CALL,
        partition_field="fetched_at",
        clustering=["source"],
        key_fields=["url", "fetched_at"],
    ),
    "dim_player": TableSpec(
        "dim_player", DIM_PLAYER, clustering=["player_id"], key_fields=["player_id"]
    ),
    "dim_team": TableSpec("dim_team", DIM_TEAM, key_fields=["team_id"]),
    "dim_venue": TableSpec("dim_venue", DIM_VENUE, key_fields=["venue_id"]),
    "dim_league_constant": TableSpec(
        "dim_league_constant", DIM_LEAGUE_CONSTANT, key_fields=["season"]
    ),
    "fact_game_schedule": TableSpec(
        "fact_game_schedule",
        FACT_GAME_SCHEDULE,
        partition_field="game_date",
        clustering=["home_team_id", "away_team_id"],
        key_fields=["game_pk"],
    ),
    "fact_player_game_log": TableSpec(
        "fact_player_game_log",
        FACT_PLAYER_GAME_LOG,
        partition_field="game_date",
        clustering=["player_id", "stat_group"],
        key_fields=["player_id", "game_pk", "stat_group"],
        require_partition_filter=True,
    ),
    "fact_player_split": TableSpec(
        "fact_player_split",
        FACT_PLAYER_SPLIT,
        clustering=["player_id", "split_code"],
        key_fields=["player_id", "season", "stat_group", "split_code"],
    ),
    "fact_savant_leaderboard": TableSpec(
        "fact_savant_leaderboard",
        FACT_SAVANT_LEADERBOARD,
        clustering=["leaderboard", "player_id"],
        key_fields=["leaderboard", "season", "player_id", "row_key"],
    ),
    "report_run": TableSpec(
        "report_run",
        REPORT_RUN,
        partition_field="generated_at",
        key_fields=["run_id"],
    ),
}
