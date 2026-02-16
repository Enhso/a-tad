"""Extract features for every cached match and write to Parquet.

Loads all cached matches via the StatsBomb adapter, runs the full
feature-extraction pipeline (both time-window and possession-based
segmentation), and writes the result as a single Parquet file.

Logs a data-quality summary including total rows, feature count,
null percentage per column, and output file size.

Usage::

    python scripts/run_feature_extraction.py
"""

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import polars as pl
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

from tactical.adapters import MatchContext, MatchInfo, StatsBombAdapter  # noqa: E402
from tactical.config import PipelineConfig  # noqa: E402
from tactical.exceptions import FeatureExtractionError, SegmentationError  # noqa: E402
from tactical.features.pipeline import extract_match_features  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CACHE_DIR = _PROJECT_ROOT / "data" / "statsbomb_cache"
OUTPUT_DIR = _PROJECT_ROOT / "data" / "output"
OUTPUT_FILE = OUTPUT_DIR / "features.parquet"


# ------------------------------------------------------------------
# Dataset definitions (mirrors download_data.py)
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Dataset:
    """A single competition-season to process.

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
# Helpers
# ------------------------------------------------------------------


def _build_contexts(
    match_info: MatchInfo,
    has_360: bool,
) -> tuple[MatchContext, MatchContext]:
    """Build home-team and away-team match contexts.

    Args:
        match_info: High-level metadata for the match.
        has_360: Whether 360 freeze-frame data exists.

    Returns:
        ``(home_context, away_context)`` pair.
    """
    home = MatchContext(
        match_id=match_info.match_id,
        team_id=match_info.home_team_id,
        opponent_id=match_info.away_team_id,
        team_is_home=True,
        has_360=has_360,
    )
    away = MatchContext(
        match_id=match_info.match_id,
        team_id=match_info.away_team_id,
        opponent_id=match_info.home_team_id,
        team_is_home=False,
        has_360=has_360,
    )
    return home, away


def _format_file_size(size_bytes: int) -> str:
    """Return a human-readable file-size string.

    Args:
        size_bytes: File size in bytes.

    Returns:
        Formatted string (e.g. ``"12.34 MB"``).
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def _log_dataframe_summary(df: pl.DataFrame, output_path: Path) -> None:
    """Log a data-quality summary for the output DataFrame.

    Reports total rows, feature column count, per-column null
    percentage (only columns with any nulls), and file size.

    Args:
        df: The combined feature DataFrame.
        output_path: Path to the written Parquet file.
    """
    metadata_cols = {
        "match_id",
        "team_id",
        "segment_type",
        "start_time",
        "end_time",
        "period",
        "match_minute",
    }
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    sep = "=" * 72
    logger.info("")
    logger.info(sep)
    logger.info("FEATURE EXTRACTION SUMMARY")
    logger.info(sep)
    logger.info("Output file      : %s", output_path)
    logger.info(
        "File size        : %s",
        _format_file_size(output_path.stat().st_size),
    )
    logger.info("Total rows       : %d", df.height)
    logger.info("Feature columns  : %d", len(feature_cols))
    logger.info("Metadata columns : %d", len(metadata_cols & set(df.columns)))
    logger.info("Total columns    : %d", df.width)

    # Segment type breakdown
    for seg_type in ("window", "possession"):
        count = df.filter(pl.col("segment_type") == seg_type).height
        logger.info("  segment_type=%-12s  rows=%d", seg_type, count)

    # Null percentage per column
    logger.info("")
    logger.info("--- Null percentage per feature column ---")

    null_stats: list[tuple[str, float]] = []
    for col in feature_cols:
        null_count = df[col].null_count()
        null_pct = 100.0 * null_count / df.height if df.height > 0 else 0.0
        null_stats.append((col, null_pct))

    # Sort by null % descending for readability
    null_stats.sort(key=lambda x: x[1], reverse=True)

    cols_with_nulls = [(name, pct) for name, pct in null_stats if pct > 0.0]
    cols_fully_populated = len(null_stats) - len(cols_with_nulls)

    if cols_with_nulls:
        for name, pct in cols_with_nulls:
            logger.info("  %-55s %6.2f%%", name, pct)
    logger.info(
        "  ... %d feature column(s) are fully populated (0%% null)",
        cols_fully_populated,
    )

    # Overall null percentage across all feature cells
    total_cells = df.height * len(feature_cols)
    total_nulls = sum(df[c].null_count() for c in feature_cols)
    overall_pct = 100.0 * total_nulls / total_cells if total_cells > 0 else 0.0
    logger.info("")
    logger.info(
        "Overall null rate : %.2f%% (%d / %d feature cells)",
        overall_pct,
        total_nulls,
        total_cells,
    )
    logger.info(sep)


# ------------------------------------------------------------------
# Core extraction loop
# ------------------------------------------------------------------


def run_extraction(
    adapter: StatsBombAdapter,
    datasets: tuple[Dataset, ...],
    config: PipelineConfig,
) -> pl.DataFrame:
    """Run feature extraction on all matches across all datasets.

    For each match, features are extracted from both the home-team
    and away-team perspectives.

    Args:
        adapter: Configured StatsBomb adapter (with cache).
        datasets: Competition-seasons to process.
        config: Pipeline configuration (controls tier, window, etc.).

    Returns:
        Concatenated polars DataFrame with features for every
        (match, team, segment) triple.
    """
    all_frames: list[pl.DataFrame] = []
    skipped = 0
    processed = 0

    for dataset in datasets:
        logger.info("Listing matches for %s ...", dataset.label)
        matches: list[MatchInfo] = adapter.list_matches(
            competition_id=dataset.competition_id,
            season_id=dataset.season_id,
        )
        logger.info("Found %d matches", len(matches))

        for match_info in tqdm(
            matches,
            desc=f"Features: {dataset.label}",
            unit="match",
        ):
            try:
                events = adapter.load_match_events(match_info.match_id)
            except Exception:
                logger.exception(
                    "Failed to load events for match %s",
                    match_info.match_id,
                )
                skipped += 1
                continue

            if not events:
                logger.warning(
                    "No events for match %s -- skipping",
                    match_info.match_id,
                )
                skipped += 1
                continue

            has_360 = adapter.supports_360(match_info.match_id)
            home_ctx, away_ctx = _build_contexts(match_info, has_360)

            for ctx in (home_ctx, away_ctx):
                try:
                    df = extract_match_features(events, ctx, config)
                    all_frames.append(df)
                except (FeatureExtractionError, SegmentationError) as exc:
                    logger.warning(
                        "Extraction failed for match %s team %s: %s",
                        ctx.match_id,
                        ctx.team_id,
                        exc,
                    )
                    skipped += 1
                    continue

            processed += 1

    logger.info(
        "Processed %d matches, skipped %d",
        processed,
        skipped,
    )

    if not all_frames:
        msg = "No feature rows produced across all datasets"
        raise FeatureExtractionError(msg)

    return pl.concat(all_frames, how="diagonal_relaxed")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main() -> None:
    """Run the full feature extraction pipeline and write Parquet."""
    adapter = StatsBombAdapter(cache_dir=CACHE_DIR)

    config = PipelineConfig(
        max_feature_tier=3,
        output_dir=OUTPUT_DIR,
    )

    logger.info("Feature extraction pipeline starting")
    logger.info("  Cache dir  : %s", CACHE_DIR)
    logger.info("  Output dir : %s", OUTPUT_DIR)
    logger.info("  Max tier   : %d", config.max_feature_tier)
    logger.info(
        "  Window     : %.0fs size, %.0fs overlap, min %d events",
        config.window.window_seconds,
        config.window.overlap_seconds,
        config.window.min_events,
    )
    logger.info(
        "  Possession : min %d events",
        config.possession.min_events,
    )

    combined = run_extraction(adapter, DATASETS, config)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(OUTPUT_FILE)
    logger.info("Parquet written to %s", OUTPUT_FILE)

    _log_dataframe_summary(combined, OUTPUT_FILE)


if __name__ == "__main__":
    main()
