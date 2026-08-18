"""Configuration and constants.

Anything that is a real-world fact (team ids, API hosts) is a constant here.
Anything environment-specific (GCP project, paths) comes from .env so the repo
stays shareable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(REPO_ROOT / ".env")

# ---------------------------------------------------------------------------
# Baseball constants
# ---------------------------------------------------------------------------

CLEVELAND_GUARDIANS_TEAM_ID = 114
MLB_SPORT_ID = 1

STATSAPI_BASE = "https://statsapi.mlb.com/api"
SAVANT_BASE = "https://baseballsavant.mlb.com"

# Recent-form windows, in games. Season is handled separately since it is a
# date range rather than a game count.
FORM_WINDOWS = (5, 15, 30)

# Situational split codes understood by statsapi's statSplits endpoint.
SPLIT_VS_LHP = "vl"
SPLIT_VS_RHP = "vr"
SPLIT_HOME = "h"
SPLIT_AWAY = "a"
DEFAULT_SPLIT_CODES = (SPLIT_VS_LHP, SPLIT_VS_RHP, SPLIT_HOME, SPLIT_AWAY)


# ---------------------------------------------------------------------------
# Politeness
# ---------------------------------------------------------------------------
# Neither source publishes a rate limit, and neither requires auth. We stay
# well under anything that could look like abuse: these are somebody else's
# servers and the whole project depends on continued access.

REQUESTS_PER_SECOND = 4.0
REQUEST_TIMEOUT_SECONDS = 45
MAX_RETRIES = 4
USER_AGENT = (
    "guards-report/0.1 (personal scouting report; contact via repo owner)"
)


@dataclass(frozen=True)
class Settings:
    gcp_project: str
    bq_dataset: str
    bq_location: str
    bq_max_bytes_billed: int
    raw_archive_dir: Path
    output_dir: Path

    @property
    def dataset_ref(self) -> str:
        return f"{self.gcp_project}.{self.bq_dataset}"

    def table(self, name: str) -> str:
        return f"{self.dataset_ref}.{name}"


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in."
        )
    return value


def load_settings() -> Settings:
    raw_archive = Path(os.environ.get("RAW_ARCHIVE_DIR", "data/raw"))
    if not raw_archive.is_absolute():
        raw_archive = REPO_ROOT / raw_archive

    return Settings(
        gcp_project=_require("GCP_PROJECT"),
        bq_dataset=os.environ.get("BQ_DATASET", "guards_report"),
        bq_location=os.environ.get("BQ_LOCATION", "US"),
        # Hard cap on bytes billed per query. A normal daily run scans single
        # digit megabytes; 1 GiB is a wide margin that still turns a runaway
        # query into a failure rather than a bill.
        bq_max_bytes_billed=int(
            os.environ.get("BQ_MAX_BYTES_BILLED", str(1024**3))
        ),
        raw_archive_dir=raw_archive,
        output_dir=REPO_ROOT / "out",
    )


# ---------------------------------------------------------------------------
# BigQuery table names
# ---------------------------------------------------------------------------

TABLE_RAW_API_CALL = "raw_api_call"
TABLE_DIM_PLAYER = "dim_player"
TABLE_DIM_TEAM = "dim_team"
TABLE_DIM_VENUE = "dim_venue"
TABLE_DIM_LEAGUE_CONSTANT = "dim_league_constant"
TABLE_GAME_SCHEDULE = "fact_game_schedule"
TABLE_PLAYER_GAME_LOG = "fact_player_game_log"
TABLE_PLAYER_SPLIT = "fact_player_split"
TABLE_SAVANT_LEADERBOARD = "fact_savant_leaderboard"
TABLE_REPORT_RUN = "report_run"
