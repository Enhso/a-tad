"""Run segmentation on all cached matches and report statistics.

Loads every cached match via the StatsBomb adapter, applies both
time-based windowing and possession-based sequencing with default
configs, then logs aggregate statistics and a halftime sanity check.

Usage::

    python scripts/run_segmentation.py
"""

from __future__ import annotations

import logging
import re
import sys
import warnings
from pathlib import Path

import numpy as np
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

from tactical.adapters import StatsBombAdapter  # noqa: E402
from tactical.config import PossessionConfig, WindowConfig  # noqa: E402
from tactical.exceptions import SegmentationError  # noqa: E402
from tactical.segmentation.possession import create_possession_sequences  # noqa: E402
from tactical.segmentation.windows import create_time_windows  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CACHE_DIR = _PROJECT_ROOT / "data" / "statsbomb_cache"

_MATCH_FILE_RE = re.compile(r"^match_(\d+)\.pkl$")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _discover_match_ids(cache_dir: Path) -> list[str]:
    """Extract sorted match IDs from cached pickle filenames.

    Args:
        cache_dir: Directory containing ``match_<id>.pkl`` files.

    Returns:
        Sorted list of match-ID strings.
    """
    ids: list[str] = []
    for path in cache_dir.iterdir():
        m = _MATCH_FILE_RE.match(path.name)
        if m:
            ids.append(m.group(1))
    ids.sort()
    return ids


def _format_distribution(values: np.ndarray) -> str:
    """Return a one-line percentile summary string.

    Args:
        values: 1-D numeric array.

    Returns:
        Formatted string with mean, std, min, p25, p50, p75, max.
    """
    if len(values) == 0:
        return "(no data)"
    p25, p50, p75 = np.percentile(values, [25, 50, 75])
    return (
        f"mean={np.mean(values):.2f}  std={np.std(values):.2f}  "
        f"min={np.min(values)}  p25={p25:.0f}  p50={p50:.0f}  "
        f"p75={p75:.0f}  max={np.max(values)}"
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    """Segment every cached match and log aggregate statistics."""
    adapter = StatsBombAdapter(cache_dir=CACHE_DIR)
    w_config = WindowConfig()
    p_config = PossessionConfig()

    match_ids = _discover_match_ids(CACHE_DIR)
    logger.info("Discovered %d cached matches in %s", len(match_ids), CACHE_DIR)

    if not match_ids:
        logger.error("No cached matches found -- run download_data.py first")
        return

    # -- accumulators --------------------------------------------------
    windows_per_match: list[int] = []
    possessions_per_match: list[int] = []
    events_per_window_all: list[int] = []
    events_per_possession_all: list[int] = []
    possession_outcome_counts: dict[str, int] = {}
    halftime_violations: list[tuple[str, float, float, set[int]]] = []
    skipped: int = 0

    # -- process each match --------------------------------------------
    for match_id in tqdm(match_ids, desc="Segmenting matches", unit="match"):
        try:
            events = adapter.load_match_events(match_id)
        except Exception:
            logger.exception("Failed to load events for match %s", match_id)
            skipped += 1
            continue

        if not events:
            skipped += 1
            continue

        # --- time windows ---
        try:
            windows = create_time_windows(events, w_config)
        except SegmentationError:
            logger.warning("Windowing failed for match %s (empty events?)", match_id)
            skipped += 1
            continue

        windows_per_match.append(len(windows))
        for w in windows:
            events_per_window_all.append(len(w.events))
            periods_in_window = {e.period for e in w.events}
            if len(periods_in_window) > 1:
                halftime_violations.append(
                    (match_id, w.start_time, w.end_time, periods_in_window)
                )

        # --- possession sequences ---
        try:
            possessions = create_possession_sequences(events, p_config)
        except SegmentationError:
            logger.warning("Possession segmentation failed for match %s", match_id)
            continue

        possessions_per_match.append(len(possessions))
        for seq in possessions:
            events_per_possession_all.append(len(seq.events))
            possession_outcome_counts[seq.outcome] = (
                possession_outcome_counts.get(seq.outcome, 0) + 1
            )

    # -- convert to numpy for stats ------------------------------------
    wpm = np.array(windows_per_match)
    ppm = np.array(possessions_per_match)
    epw = np.array(events_per_window_all)
    epp = np.array(events_per_possession_all)

    # -- log summary ---------------------------------------------------
    sep = "=" * 72
    logger.info("")
    logger.info(sep)
    logger.info("SEGMENTATION STATISTICS")
    logger.info(sep)
    logger.info(
        "Matches processed: %d  |  Skipped: %d  |  Total cached: %d",
        len(wpm),
        skipped,
        len(match_ids),
    )
    logger.info("")

    logger.info("--- Windows per match ---")
    logger.info("  %s", _format_distribution(wpm))
    logger.info("  Total windows: %d", int(np.sum(wpm)))
    logger.info("")

    logger.info("--- Possessions per match ---")
    logger.info("  %s", _format_distribution(ppm))
    logger.info("  Total possessions: %d", int(np.sum(ppm)))
    logger.info("")

    logger.info("--- Events per window ---")
    logger.info("  %s", _format_distribution(epw))
    logger.info("")

    logger.info("--- Events per possession ---")
    logger.info("  %s", _format_distribution(epp))
    logger.info("")

    logger.info("--- Possession outcome distribution ---")
    for outcome in sorted(possession_outcome_counts):
        logger.info("  %-20s %d", outcome, possession_outcome_counts[outcome])
    logger.info("")

    # -- halftime sanity check -----------------------------------------
    logger.info(sep)
    logger.info("HALFTIME SANITY CHECK")
    logger.info(sep)
    if halftime_violations:
        logger.error(
            "FAIL: %d window(s) span multiple periods!", len(halftime_violations)
        )
        for match_id, start, end, periods in halftime_violations[:10]:
            logger.error(
                "  match=%s  start=%.1f  end=%.1f  periods=%s",
                match_id,
                start,
                end,
                periods,
            )
    else:
        logger.info(
            "PASS: No windows span halftime (0 violations across %d windows)",
            int(np.sum(wpm)),
        )
    logger.info(sep)


if __name__ == "__main__":
    main()
