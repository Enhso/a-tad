"""Tier 1 spatial and temporal feature extractors.

Provides :class:`SpatialFeatureExtractor` and
:class:`TemporalFeatureExtractor`, both satisfying the
:class:`~tactical.features.base.FeatureExtractor` protocol.

Tier 1 features require only the normalised event stream -- no derived
calculations or 360 freeze-frame data.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tactical.adapters.schemas import MatchContext, NormalizedEvent

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_GOAL_LOCATION: tuple[float, float] = (100.0, 50.0)
_THIRD_BOUNDARY_LOW: float = 33.3
_THIRD_BOUNDARY_HIGH: float = 66.7

# ------------------------------------------------------------------
# Spatial extractor
# ------------------------------------------------------------------

_SPATIAL_NAMES: tuple[str, ...] = (
    "t1_spatial_event_centroid_x",
    "t1_spatial_event_centroid_y",
    "t1_spatial_event_spread_x",
    "t1_spatial_event_spread_y",
    "t1_spatial_avg_distance_to_goal",
    "t1_spatial_attacking_third_pct",
    "t1_spatial_middle_third_pct",
    "t1_spatial_defensive_third_pct",
    "t1_spatial_left_channel_pct",
    "t1_spatial_center_channel_pct",
    "t1_spatial_right_channel_pct",
)


class SpatialFeatureExtractor:
    """Extract spatial features from event locations.

    Computes centroid, spread, distance-to-goal, and zone
    distribution features from the ``(x, y)`` locations of events
    belonging to the focal team.
    """

    __slots__ = ()

    @property
    def tier(self) -> int:
        """Feature tier: 1 (event-only)."""
        return 1

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names of spatial features produced by this extractor."""
        return _SPATIAL_NAMES

    def extract(
        self,
        events: tuple[NormalizedEvent, ...],
        context: MatchContext,
    ) -> dict[str, float | None]:
        """Extract spatial features from located team events.

        Args:
            events: Tuple of normalised events in the segment.
            context: Per-team match context.

        Returns:
            Dict mapping each spatial feature name to its value,
            or ``None`` when no located events exist.
        """
        locs = _team_locations(events, context.team_id)
        if locs.shape[0] == 0:
            return {name: None for name in _SPATIAL_NAMES}

        xs: NDArray[np.floating] = locs[:, 0]
        ys: NDArray[np.floating] = locs[:, 1]
        n = float(locs.shape[0])

        dx = xs - _GOAL_LOCATION[0]
        dy = ys - _GOAL_LOCATION[1]
        distances: NDArray[np.floating] = np.sqrt(dx * dx + dy * dy)

        return {
            "t1_spatial_event_centroid_x": float(np.mean(xs)),
            "t1_spatial_event_centroid_y": float(np.mean(ys)),
            "t1_spatial_event_spread_x": float(np.std(xs)),
            "t1_spatial_event_spread_y": float(np.std(ys)),
            "t1_spatial_avg_distance_to_goal": float(np.mean(distances)),
            "t1_spatial_attacking_third_pct": float(
                np.count_nonzero(xs > _THIRD_BOUNDARY_HIGH) / n
            ),
            "t1_spatial_middle_third_pct": float(
                np.count_nonzero(
                    (xs >= _THIRD_BOUNDARY_LOW) & (xs <= _THIRD_BOUNDARY_HIGH)
                )
                / n
            ),
            "t1_spatial_defensive_third_pct": float(
                np.count_nonzero(xs < _THIRD_BOUNDARY_LOW) / n
            ),
            "t1_spatial_left_channel_pct": float(
                np.count_nonzero(ys > _THIRD_BOUNDARY_HIGH) / n
            ),
            "t1_spatial_center_channel_pct": float(
                np.count_nonzero(
                    (ys >= _THIRD_BOUNDARY_LOW) & (ys <= _THIRD_BOUNDARY_HIGH)
                )
                / n
            ),
            "t1_spatial_right_channel_pct": float(
                np.count_nonzero(ys < _THIRD_BOUNDARY_LOW) / n
            ),
        }


# ------------------------------------------------------------------
# Temporal extractor
# ------------------------------------------------------------------

_TEMPORAL_NAMES: tuple[str, ...] = (
    "t1_temporal_event_rate",
    "t1_temporal_mean_inter_event_time",
    "t1_temporal_std_inter_event_time",
    "t1_temporal_max_inter_event_time",
    "t1_temporal_match_minute",
    "t1_temporal_period",
)


class TemporalFeatureExtractor:
    """Extract temporal features from event timestamps.

    Computes event rate, inter-event time statistics, average match
    minute, and dominant period for the focal team's events within
    a segment.
    """

    __slots__ = ()

    @property
    def tier(self) -> int:
        """Feature tier: 1 (event-only)."""
        return 1

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names of temporal features produced by this extractor."""
        return _TEMPORAL_NAMES

    def extract(
        self,
        events: tuple[NormalizedEvent, ...],
        context: MatchContext,
    ) -> dict[str, float | None]:
        """Extract temporal features from team events.

        Args:
            events: Tuple of normalised events in the segment.
            context: Per-team match context.

        Returns:
            Dict mapping each temporal feature name to its value,
            or ``None`` when the feature cannot be computed.
        """
        team_events = tuple(e for e in events if e.team_id == context.team_id)
        n = len(team_events)
        if n == 0:
            return {name: None for name in _TEMPORAL_NAMES}

        timestamps = np.array([e.timestamp for e in team_events], dtype=np.float64)
        minutes = np.array([e.match_minute for e in team_events], dtype=np.float64)
        periods = [e.period for e in team_events]

        duration = float(timestamps.max() - timestamps.min())
        most_common_period = Counter(periods).most_common(1)[0][0]

        # Single event or zero duration: rate and inter-event stats
        # cannot be meaningfully computed.
        if n < 2 or duration == 0.0:
            return {
                "t1_temporal_event_rate": None,
                "t1_temporal_mean_inter_event_time": None,
                "t1_temporal_std_inter_event_time": None,
                "t1_temporal_max_inter_event_time": None,
                "t1_temporal_match_minute": float(np.mean(minutes)),
                "t1_temporal_period": float(most_common_period),
            }

        timestamps.sort()
        deltas: NDArray[np.floating] = np.diff(timestamps)

        return {
            "t1_temporal_event_rate": n / duration,
            "t1_temporal_mean_inter_event_time": float(np.mean(deltas)),
            "t1_temporal_std_inter_event_time": float(np.std(deltas)),
            "t1_temporal_max_inter_event_time": float(np.max(deltas)),
            "t1_temporal_match_minute": float(np.mean(minutes)),
            "t1_temporal_period": float(most_common_period),
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _team_locations(
    events: tuple[NormalizedEvent, ...],
    team_id: str,
) -> NDArray[np.floating]:
    """Return an ``(N, 2)`` array of located team-event coordinates.

    Args:
        events: Full segment events (may include both teams).
        team_id: Focal team identifier.

    Returns:
        Numpy array of shape ``(N, 2)`` where *N* is the number of
        team events with a non-``None`` location.  Returns an empty
        ``(0, 2)`` array when no located events exist.
    """
    locs = [
        e.location for e in events if e.team_id == team_id and e.location is not None
    ]
    if not locs:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(locs, dtype=np.float64)
