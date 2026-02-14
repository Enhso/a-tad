"""Time-based sliding-window segmentation for match events.

Segments a match's normalised event stream into fixed-duration,
optionally overlapping windows that never span period boundaries.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from itertools import groupby
from operator import attrgetter
from typing import TYPE_CHECKING

from tactical.exceptions import SegmentationError

if TYPE_CHECKING:
    from tactical.adapters.schemas import NormalizedEvent
    from tactical.config import WindowConfig


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """A fixed-duration segment of match events.

    Attributes:
        match_id: Identifier of the match.
        period: Match period this window belongs to.
        start_time: Window start in seconds from period start.
        end_time: Window end in seconds from period start.
        match_minute_start: Earliest match minute among contained events.
        match_minute_end: Latest match minute among contained events.
        events: Immutable sequence of events inside this window.
    """

    match_id: str
    period: int
    start_time: float
    end_time: float
    match_minute_start: float
    match_minute_end: float
    events: tuple[NormalizedEvent, ...]


def create_time_windows(
    events: list[NormalizedEvent],
    config: WindowConfig,
) -> list[TimeWindow]:
    """Segment a match's events into fixed-duration sliding windows.

    Windows do not span period boundaries. Windows with fewer than
    ``config.min_events`` events are dropped.

    Args:
        events: Sorted list of normalized events for a single match.
        config: Window configuration (size, overlap, min events).

    Returns:
        List of TimeWindow objects, sorted by (period, start_time).

    Raises:
        SegmentationError: If events list is empty.
    """
    if not events:
        msg = "Cannot create windows from an empty event list"
        raise SegmentationError(msg)

    step_size = config.window_seconds - config.overlap_seconds

    period_key = attrgetter("period")
    sorted_events = sorted(events, key=lambda e: (e.period, e.timestamp))

    windows: list[TimeWindow] = []

    for period, group in groupby(sorted_events, key=period_key):
        period_events = list(group)
        timestamps = [e.timestamp for e in period_events]

        min_ts = timestamps[0]
        max_ts = timestamps[-1]

        start = min_ts
        while start <= max_ts:
            end = start + config.window_seconds

            lo = bisect_left(timestamps, start)
            hi = bisect_left(timestamps, end)

            if hi - lo >= config.min_events:
                window_slice = period_events[lo:hi]
                match_minutes = [e.match_minute for e in window_slice]
                windows.append(
                    TimeWindow(
                        match_id=window_slice[0].match_id,
                        period=period,
                        start_time=start,
                        end_time=end,
                        match_minute_start=min(match_minutes),
                        match_minute_end=max(match_minutes),
                        events=tuple(window_slice),
                    )
                )

            start += step_size

    return windows
