"""Segmentation utilities for the Tactical State Discovery Engine.

Provides time-based and possession-based windowing of normalised match
events.
"""

from tactical.segmentation.possession import (
    DEAD_BALL_EVENTS,
    SET_PIECE_STARTS,
    TURNOVER_EVENTS,
    PossessionSequence,
    create_possession_sequences,
)
from tactical.segmentation.windows import TimeWindow, create_time_windows

__all__ = [
    "DEAD_BALL_EVENTS",
    "SET_PIECE_STARTS",
    "TURNOVER_EVENTS",
    "PossessionSequence",
    "TimeWindow",
    "create_possession_sequences",
    "create_time_windows",
]
