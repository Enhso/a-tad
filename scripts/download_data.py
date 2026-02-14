"""Download and cache StatsBomb match data for the tactical analysis project.

Downloads all matches (events + lineups) for the configured
competition-seasons and prints a data-quality summary log.

Usage::

    python scripts/download_data.py
"""

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure the src package is importable when running as a standalone script.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from statsbombpy.api_client import (  # type: ignore[import-untyped]  # noqa: E402
    NoAuthWarning,
)

warnings.filterwarnings("ignore", category=NoAuthWarning)

from tactical.adapters import MatchInfo, NormalizedEvent, StatsBombAdapter  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CACHE_DIR = _PROJECT_ROOT / "data" / "statsbomb_cache"


# ------------------------------------------------------------------
# Dataset definitions
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Dataset:
    """A single competition-season to download.

    Attributes:
        label: Human-readable label for log output.
        competition_id: StatsBomb competition identifier.
        season_id: StatsBomb season identifier.
    """

    label: str
    competition_id: int
    season_id: int


DATASETS: tuple[Dataset, ...] = (
    Dataset("Arsenal 15-16 Premier League", competition_id=2, season_id=27),
    Dataset("Leverkusen 23-24 Bundesliga", competition_id=9, season_id=281),
    Dataset("Arsenal Invincibles 03-04 Premier League", competition_id=2, season_id=44),
)


# ------------------------------------------------------------------
# Per-dataset statistics
# ------------------------------------------------------------------


@dataclass(slots=True)
class DatasetStats:
    """Mutable accumulator for one dataset's download statistics.

    Attributes:
        label: Human-readable dataset label.
        match_count: Number of matches downloaded.
        event_count: Total normalised events across all matches.
        matches_with_360: Matches where at least one event has 360 data.
        matches_without_360: Matches with no 360 freeze-frame data.
    """

    label: str
    match_count: int = 0
    event_count: int = 0
    matches_with_360: int = 0
    matches_without_360: int = 0


# ------------------------------------------------------------------
# Download helpers
# ------------------------------------------------------------------


def _has_360(events: list[NormalizedEvent]) -> bool:
    """Return True if any event contains a non-None freeze frame.

    Args:
        events: Normalised events for a single match.

    Returns:
        Whether 360 data is present in at least one event.
    """
    return any(e.freeze_frame != [] for e in events)


def _download_dataset(
    adapter: StatsBombAdapter,
    dataset: Dataset,
) -> DatasetStats:
    """Download all matches for one competition-season.

    Fetches events and lineups for every match, caching both to disk.

    Args:
        adapter: Configured StatsBomb adapter (with cache).
        dataset: Competition-season to download.

    Returns:
        Accumulated statistics for this dataset.
    """
    stats = DatasetStats(label=dataset.label)

    logger.info("Listing matches for %s ...", dataset.label)
    matches: list[MatchInfo] = adapter.list_matches(
        competition_id=dataset.competition_id,
        season_id=dataset.season_id,
    )
    stats.match_count = len(matches)
    logger.info("Found %d matches", stats.match_count)

    for match in tqdm(matches, desc=dataset.label, unit="match"):
        events = adapter.load_match_events(match.match_id)
        adapter.load_match_lineups(match.match_id)

        stats.event_count += len(events)
        if _has_360(events):
            stats.matches_with_360 += 1
        else:
            stats.matches_without_360 += 1

    return stats


# ------------------------------------------------------------------
# Summary log
# ------------------------------------------------------------------

_HEADER = f"{'Dataset':<50} {'Matches':>8} {'Events':>9} {'w/ 360':>8} {'w/o 360':>9}"
_SEPARATOR = "-" * len(_HEADER)


def _print_summary(all_stats: list[DatasetStats]) -> None:
    """Print a formatted data-quality summary table.

    Args:
        all_stats: Statistics for each downloaded dataset.
    """
    logger.info("")
    logger.info("=" * len(_HEADER))
    logger.info("DATA QUALITY SUMMARY")
    logger.info("=" * len(_HEADER))
    logger.info(_HEADER)
    logger.info(_SEPARATOR)

    total_matches = 0
    total_events = 0
    total_360 = 0
    total_no_360 = 0

    for ds in all_stats:
        logger.info(
            "%-50s %8d %9d %8d %9d",
            ds.label,
            ds.match_count,
            ds.event_count,
            ds.matches_with_360,
            ds.matches_without_360,
        )
        total_matches += ds.match_count
        total_events += ds.event_count
        total_360 += ds.matches_with_360
        total_no_360 += ds.matches_without_360

    logger.info(_SEPARATOR)
    logger.info(
        "%-50s %8d %9d %8d %9d",
        "TOTAL",
        total_matches,
        total_events,
        total_360,
        total_no_360,
    )
    logger.info("=" * len(_HEADER))
    logger.info("Cache directory: %s", CACHE_DIR)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main() -> None:
    """Download and cache all configured competition-seasons."""
    adapter = StatsBombAdapter(cache_dir=CACHE_DIR)
    all_stats: list[DatasetStats] = []

    for dataset in DATASETS:
        stats = _download_dataset(adapter, dataset)
        all_stats.append(stats)

    _print_summary(all_stats)


if __name__ == "__main__":
    main()
