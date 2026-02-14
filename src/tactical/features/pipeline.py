"""End-to-end feature extraction pipeline.

Wires segmentation (time windows and possession sequences) to the
feature registry, producing a single :class:`polars.DataFrame` with
one row per segment and all extracted feature columns.

Public API
----------
.. function:: extract_match_features

    Extract features for every segment of a match and return a tidy
    polars DataFrame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from tqdm import tqdm

from tactical.exceptions import FeatureExtractionError
from tactical.features.registry import create_default_registry
from tactical.segmentation.possession import create_possession_sequences
from tactical.segmentation.windows import create_time_windows

if TYPE_CHECKING:
    from tactical.adapters.schemas import MatchContext, NormalizedEvent
    from tactical.config import PipelineConfig
    from tactical.segmentation.possession import PossessionSequence
    from tactical.segmentation.windows import TimeWindow


def extract_match_features(
    events: list[NormalizedEvent],
    match_context: MatchContext,
    config: PipelineConfig,
    windows: list[TimeWindow] | None = None,
    possessions: list[PossessionSequence] | None = None,
) -> pl.DataFrame:
    """Extract features for all segments of a match.

    Runs feature extraction on the provided windows and/or possession
    sequences.  Returns a polars DataFrame with one row per segment.

    Args:
        events: Full list of normalized events for the match.
        match_context: Context for feature extraction.
        config: Pipeline configuration.
        windows: Pre-computed time windows.  If ``None``, windows are
            created via :func:`create_time_windows`.
        possessions: Pre-computed possession sequences.  If ``None``,
            sequences are created via
            :func:`create_possession_sequences`.

    Returns:
        A :class:`polars.DataFrame` with metadata columns
        (``match_id``, ``team_id``, ``segment_type``, ``start_time``,
        ``end_time``, ``period``, ``match_minute``) followed by all
        feature columns from the active tiers.

    Raises:
        FeatureExtractionError: If no valid segments are produced.
    """
    if windows is None:
        windows = create_time_windows(events, config.window)
    if possessions is None:
        possessions = create_possession_sequences(events, config.possession)

    registry = create_default_registry()
    max_tier: int = config.max_feature_tier

    rows: list[dict[str, object]] = []

    segments: list[
        tuple[str, tuple[NormalizedEvent, ...], float, float, int, float]
    ] = []

    for win in windows:
        segments.append(
            (
                "window",
                win.events,
                win.start_time,
                win.end_time,
                win.period,
                win.match_minute_start,
            )
        )

    for poss in possessions:
        segments.append(
            (
                "possession",
                poss.events,
                poss.start_time,
                poss.end_time,
                poss.period,
                poss.match_minute_start,
            )
        )

    for seg_type, seg_events, start, end, period, mm in tqdm(
        segments,
        desc="Extracting features",
        unit="seg",
    ):
        features = registry.extract_all(seg_events, match_context, max_tier=max_tier)

        row: dict[str, object] = {
            "match_id": match_context.match_id,
            "team_id": match_context.team_id,
            "segment_type": seg_type,
            "start_time": start,
            "end_time": end,
            "period": period,
            "match_minute": mm,
        }
        row.update(features)
        rows.append(row)

    if not rows:
        msg = (
            f"No valid segments produced for match "
            f"{match_context.match_id!r}, team {match_context.team_id!r}"
        )
        raise FeatureExtractionError(msg)

    return pl.DataFrame(rows)
