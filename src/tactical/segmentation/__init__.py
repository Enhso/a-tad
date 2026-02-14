"""Segmentation utilities for the Tactical State Discovery Engine.

Provides time-based (and future possession-based) windowing of
normalised match events.
"""

from tactical.segmentation.windows import TimeWindow, create_time_windows

__all__ = ["TimeWindow", "create_time_windows"]
