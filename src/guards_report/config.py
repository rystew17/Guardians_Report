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

# Game types that count toward a club's record. Spring training ("S"),
# exhibition ("E") and the All-Star game ("A") are excluded: the schedule
# endpoint returns them alongside real games, and counting them makes two
# clubs look like they have already met before their first actual meeting.
#   R regular season | F wild card | D division series
#   L championship series | W World Series | P playoff or tiebreaker
COMPETITIVE_GAME_TYPES = ("R", "F", "D", "L", "W", "P")
REGULAR_SEASON_GAME_TYPE = "R"

STATSAPI_BASE = "https://statsapi.mlb.com/api"
SAVANT_BASE = "https://baseballsavant.mlb.com"

# Recent-form windows, in games. Season is handled separately since it is a
# date range rather than a game count.
FORM_WINDOWS = (5, 15, 30)

# Situational split codes understood by statsapi's statSplits endpoint. The
# full catalogue runs to 602 codes; these are the ones that change a decision.
SPLIT_VS_LHP = "vl"
SPLIT_VS_RHP = "vr"
SPLIT_HOME = "h"
SPLIT_AWAY = "a"
DEFAULT_SPLIT_CODES = (SPLIT_VS_LHP, SPLIT_VS_RHP, SPLIT_HOME, SPLIT_AWAY)

# For hitters: platoon and venue, then the situations where approach shows --
# scoring position, late and close, and the count states that separate a hitter
# who can survive two strikes from one who must do damage early.
HITTER_SPLIT_CODES = (
    "vl", "vr", "h", "a",
    "risp",   # runners in scoring position
    "risp2",  # scoring position, two out
    "lc",     # late and close
    "2s",     # two strikes
    "ac",     # ahead in count
    "bc",     # behind in count
)

# For pitchers the same idea, plus a times-through-the-order proxy: statsapi
# has no TTO split, but pitch-count buckets stand in for it, and the gap
# between a starter's first 75 pitches and everything after is where the
# third-time-through penalty shows up.
PITCHER_SPLIT_CODES = (
    "vl", "vr", "h", "a",
    "risp",
    "lc",
    "2s",
    "pi000",  # first 75 pitches
    "pi760",  # pitch 76 onward
)

# Situation codes rendered with a friendly label in the report.
SPLIT_LABELS = {
    "vl": "vs LHP", "vr": "vs RHP", "h": "Home", "a": "Away",
    "risp": "RISP", "risp2": "RISP, 2 out", "lc": "Late & close",
    "2s": "Two strikes", "ac": "Ahead in count", "bc": "Behind in count",
    "pi000": "Pitches 1-75", "pi760": "Pitches 76+",
}


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
    # Cloud Storage bucket that published reports are copied to, so a report
    # can be handed to someone as a link instead of a file. Empty means
    # publishing is simply switched off.
    gcs_bucket: str = ""

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
        gcs_bucket=os.environ.get("GCS_BUCKET", ""),
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
