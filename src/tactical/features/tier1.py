"""Tier 1 feature extractors for the Tactical State Discovery Engine.

Provides seven extractor classes satisfying the
:class:`~tactical.features.base.FeatureExtractor` protocol:

* :class:`SpatialFeatureExtractor` -- centroid, spread, zone distributions
* :class:`TemporalFeatureExtractor` -- event rate, inter-event timing
* :class:`PassingFeatureExtractor` -- pass counts, completion, geometry
* :class:`CarryingFeatureExtractor` -- carry counts, distance, directness
* :class:`DefendingFeatureExtractor` -- defensive action counts and height
* :class:`ShootingFeatureExtractor` -- shot counts and distance
* :class:`ContextFeatureExtractor` -- score, possession share

All are Tier 1 (event-only): no derived calculations or 360 freeze-frame
data required.
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
_PROGRESSIVE_THRESHOLD: float = 10.0
_SWITCH_THRESHOLD: float = 50.0

_DEFENSIVE_ACTION_TYPES: frozenset[str] = frozenset(
    {"tackle", "interception", "clearance", "block"}
)

_TACKLE_SUCCESS_OUTCOMES: frozenset[str] = frozenset(
    {"won", "success", "success_in_play", "success_out"}
)

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
# Passing extractor
# ------------------------------------------------------------------

_PASS_NAMES: tuple[str, ...] = (
    "t1_pass_count",
    "t1_pass_completion_rate",
    "t1_pass_length_mean",
    "t1_pass_length_std",
    "t1_pass_angle_std",
    "t1_pass_progressive_count",
    "t1_pass_backward_ratio",
    "t1_pass_height_ground_pct",
    "t1_pass_height_low_pct",
    "t1_pass_height_high_pct",
    "t1_pass_switch_count",
    "t1_pass_directness",
)


class PassingFeatureExtractor:
    """Extract passing features from pass events.

    Computes pass count, completion rate, geometric properties
    (length, angle, directness), progressive and switch counts,
    and backward ratio for the focal team's passes.
    """

    __slots__ = ()

    @property
    def tier(self) -> int:
        """Feature tier: 1 (event-only)."""
        return 1

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names of passing features produced by this extractor."""
        return _PASS_NAMES

    def extract(
        self,
        events: tuple[NormalizedEvent, ...],
        context: MatchContext,
    ) -> dict[str, float | None]:
        """Extract passing features from team pass events.

        Args:
            events: Tuple of normalised events in the segment.
            context: Per-team match context.

        Returns:
            Dict mapping each passing feature name to its value.
            Returns ``count=0`` and all others ``None`` when no
            passes exist.
        """
        passes = [
            e for e in events if e.team_id == context.team_id and e.event_type == "pass"
        ]
        n = len(passes)
        if n == 0:
            return _pass_none_result(count=0.0)

        completion_rate = sum(1 for p in passes if p.event_outcome == "complete") / n

        # Geometric features require both location and end_location.
        geo = [
            p for p in passes if p.location is not None and p.end_location is not None
        ]
        if not geo:
            result = _pass_none_result(count=float(n))
            result["t1_pass_completion_rate"] = completion_rate
            return result

        starts = np.array([p.location for p in geo], dtype=np.float64)
        ends = np.array([p.end_location for p in geo], dtype=np.float64)
        dx: NDArray[np.floating] = ends[:, 0] - starts[:, 0]
        dy: NDArray[np.floating] = ends[:, 1] - starts[:, 1]
        lengths: NDArray[np.floating] = np.sqrt(dx * dx + dy * dy)
        angles: NDArray[np.floating] = np.arctan2(dy, dx)

        n_geo = float(len(geo))

        # Directness: mean(x_displacement / length), skip zero-length.
        nonzero = lengths > 0.0
        directness: float | None = (
            float(np.mean(dx[nonzero] / lengths[nonzero])) if np.any(nonzero) else None
        )

        return {
            "t1_pass_count": float(n),
            "t1_pass_completion_rate": completion_rate,
            "t1_pass_length_mean": float(np.mean(lengths)),
            "t1_pass_length_std": float(np.std(lengths)),
            "t1_pass_angle_std": float(np.std(angles)),
            "t1_pass_progressive_count": float(
                np.count_nonzero(dx > _PROGRESSIVE_THRESHOLD)
            ),
            "t1_pass_backward_ratio": float(np.count_nonzero(dx < 0.0) / n_geo),
            "t1_pass_height_ground_pct": None,
            "t1_pass_height_low_pct": None,
            "t1_pass_height_high_pct": None,
            "t1_pass_switch_count": float(
                np.count_nonzero(np.abs(dy) > _SWITCH_THRESHOLD)
            ),
            "t1_pass_directness": directness,
        }


def _pass_none_result(*, count: float) -> dict[str, float | None]:
    """Return a pass feature dict with *count* set and all else ``None``."""
    result: dict[str, float | None] = {name: None for name in _PASS_NAMES}
    result["t1_pass_count"] = count
    return result


# ------------------------------------------------------------------
# Carrying extractor
# ------------------------------------------------------------------

_CARRY_NAMES: tuple[str, ...] = (
    "t1_carry_count",
    "t1_carry_distance_total",
    "t1_carry_distance_mean",
    "t1_carry_directness",
    "t1_carry_progressive_count",
    "t1_carry_dribble_attempt_count",
    "t1_carry_dribble_success_rate",
)


class CarryingFeatureExtractor:
    """Extract carrying features from carry events.

    Computes carry count, distance statistics, directness, and
    progressive carry count for the focal team.  Dribble attempts
    are proxied by carry count in v1.
    """

    __slots__ = ()

    @property
    def tier(self) -> int:
        """Feature tier: 1 (event-only)."""
        return 1

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names of carrying features produced by this extractor."""
        return _CARRY_NAMES

    def extract(
        self,
        events: tuple[NormalizedEvent, ...],
        context: MatchContext,
    ) -> dict[str, float | None]:
        """Extract carrying features from team carry events.

        Args:
            events: Tuple of normalised events in the segment.
            context: Per-team match context.

        Returns:
            Dict mapping each carrying feature name to its value.
            Returns ``count=0`` and all others ``None`` when no
            carries exist.
        """
        carries = [
            e
            for e in events
            if e.team_id == context.team_id and e.event_type == "carry"
        ]
        n = len(carries)
        if n == 0:
            return {
                "t1_carry_count": 0.0,
                "t1_carry_distance_total": None,
                "t1_carry_distance_mean": None,
                "t1_carry_directness": None,
                "t1_carry_progressive_count": 0.0,
                "t1_carry_dribble_attempt_count": 0.0,
                "t1_carry_dribble_success_rate": None,
            }

        # Geometric features require both location and end_location.
        geo = [
            c for c in carries if c.location is not None and c.end_location is not None
        ]
        if not geo:
            return {
                "t1_carry_count": float(n),
                "t1_carry_distance_total": None,
                "t1_carry_distance_mean": None,
                "t1_carry_directness": None,
                "t1_carry_progressive_count": 0.0,
                "t1_carry_dribble_attempt_count": float(n),
                "t1_carry_dribble_success_rate": None,
            }

        starts = np.array([c.location for c in geo], dtype=np.float64)
        ends = np.array([c.end_location for c in geo], dtype=np.float64)
        dx: NDArray[np.floating] = ends[:, 0] - starts[:, 0]
        dy: NDArray[np.floating] = ends[:, 1] - starts[:, 1]
        dists: NDArray[np.floating] = np.sqrt(dx * dx + dy * dy)

        nonzero = dists > 0.0
        directness: float | None = (
            float(np.mean(dx[nonzero] / dists[nonzero])) if np.any(nonzero) else None
        )

        return {
            "t1_carry_count": float(n),
            "t1_carry_distance_total": float(np.sum(dists)),
            "t1_carry_distance_mean": float(np.mean(dists)),
            "t1_carry_directness": directness,
            "t1_carry_progressive_count": float(
                np.count_nonzero(dx > _PROGRESSIVE_THRESHOLD)
            ),
            "t1_carry_dribble_attempt_count": float(n),
            "t1_carry_dribble_success_rate": None,
        }


# ------------------------------------------------------------------
# Defending extractor
# ------------------------------------------------------------------

_DEFEND_NAMES: tuple[str, ...] = (
    "t1_defend_pressure_count",
    "t1_defend_pressure_rate",
    "t1_defend_tackle_count",
    "t1_defend_tackle_success_rate",
    "t1_defend_interception_count",
    "t1_defend_clearance_count",
    "t1_defend_foul_count",
    "t1_defend_under_pressure_pct",
    "t1_defend_defensive_action_height",
)


class DefendingFeatureExtractor:
    """Extract defending features from defensive events.

    Computes counts of defensive actions (pressures, tackles,
    interceptions, clearances, fouls), tackle success rate,
    an under-pressure percentage, and the mean flipped
    x-coordinate of defensive actions.

    Tackle success is determined by outcomes ``"won"``,
    ``"success"``, ``"success_in_play"``, or ``"success_out"``.
    """

    __slots__ = ()

    @property
    def tier(self) -> int:
        """Feature tier: 1 (event-only)."""
        return 1

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names of defending features produced by this extractor."""
        return _DEFEND_NAMES

    def extract(
        self,
        events: tuple[NormalizedEvent, ...],
        context: MatchContext,
    ) -> dict[str, float | None]:
        """Extract defending features from team defensive events.

        Args:
            events: Tuple of normalised events in the segment.
            context: Per-team match context.

        Returns:
            Dict mapping each defending feature name to its value.
        """
        team_events = [e for e in events if e.team_id == context.team_id]
        n_team = len(team_events)
        n_total = len(events)

        pressure_count = sum(1 for e in team_events if e.event_type == "pressure")
        tackles = [e for e in team_events if e.event_type in {"tackle", "duel"}]
        tackle_count = len(tackles)
        tackle_success_rate: float | None = (
            sum(1 for t in tackles if t.event_outcome in _TACKLE_SUCCESS_OUTCOMES)
            / tackle_count
            if tackle_count > 0
            else None
        )
        interception_count = sum(
            1 for e in team_events if e.event_type == "interception"
        )
        clearance_count = sum(1 for e in team_events if e.event_type == "clearance")
        foul_count = sum(1 for e in team_events if e.event_type == "foul_committed")

        under_pressure_pct: float | None = (
            sum(1 for e in team_events if e.under_pressure) / n_team
            if n_team > 0
            else None
        )

        pressure_rate: float | None = pressure_count / n_total if n_total > 0 else None

        # Defensive action height: mean flipped x of tackles,
        # interceptions, clearances, blocks.
        def_xs = [
            100.0 - e.location[0]
            for e in team_events
            if e.event_type in _DEFENSIVE_ACTION_TYPES and e.location is not None
        ]
        defensive_action_height: float | None = (
            float(np.mean(np.array(def_xs, dtype=np.float64))) if def_xs else None
        )

        return {
            "t1_defend_pressure_count": float(pressure_count),
            "t1_defend_pressure_rate": pressure_rate,
            "t1_defend_tackle_count": float(tackle_count),
            "t1_defend_tackle_success_rate": tackle_success_rate,
            "t1_defend_interception_count": float(interception_count),
            "t1_defend_clearance_count": float(clearance_count),
            "t1_defend_foul_count": float(foul_count),
            "t1_defend_under_pressure_pct": under_pressure_pct,
            "t1_defend_defensive_action_height": defensive_action_height,
        }


# ------------------------------------------------------------------
# Shooting extractor
# ------------------------------------------------------------------

_SHOOT_NAMES: tuple[str, ...] = (
    "t1_shoot_count",
    "t1_shoot_distance_mean",
)


class ShootingFeatureExtractor:
    """Extract shooting features from shot events.

    Computes shot count and mean distance from shot location to the
    goal centre ``(100, 50)`` for the focal team.
    """

    __slots__ = ()

    @property
    def tier(self) -> int:
        """Feature tier: 1 (event-only)."""
        return 1

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names of shooting features produced by this extractor."""
        return _SHOOT_NAMES

    def extract(
        self,
        events: tuple[NormalizedEvent, ...],
        context: MatchContext,
    ) -> dict[str, float | None]:
        """Extract shooting features from team shot events.

        Args:
            events: Tuple of normalised events in the segment.
            context: Per-team match context.

        Returns:
            Dict mapping each shooting feature name to its value.
            Returns ``count=0`` and ``distance_mean=None`` when no
            shots exist.
        """
        shots = [
            e for e in events if e.team_id == context.team_id and e.event_type == "shot"
        ]
        n = len(shots)
        if n == 0:
            return {
                "t1_shoot_count": 0.0,
                "t1_shoot_distance_mean": None,
            }

        locs = [s.location for s in shots if s.location is not None]
        if not locs:
            return {
                "t1_shoot_count": float(n),
                "t1_shoot_distance_mean": None,
            }

        arr = np.array(locs, dtype=np.float64)
        dx = arr[:, 0] - _GOAL_LOCATION[0]
        dy = arr[:, 1] - _GOAL_LOCATION[1]
        distances: NDArray[np.floating] = np.sqrt(dx * dx + dy * dy)

        return {
            "t1_shoot_count": float(n),
            "t1_shoot_distance_mean": float(np.mean(distances)),
        }


# ------------------------------------------------------------------
# Context extractor
# ------------------------------------------------------------------

_CONTEXT_NAMES: tuple[str, ...] = (
    "t1_context_score_differential",
    "t1_context_possession_share",
    "t1_context_possession_team",
)


class ContextFeatureExtractor:
    """Extract match-context features from the event window.

    Computes score differential, possession share, and current
    possession indicator for the focal team.
    """

    __slots__ = ()

    @property
    def tier(self) -> int:
        """Feature tier: 1 (event-only)."""
        return 1

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names of context features produced by this extractor."""
        return _CONTEXT_NAMES

    def extract(
        self,
        events: tuple[NormalizedEvent, ...],
        context: MatchContext,
    ) -> dict[str, float | None]:
        """Extract context features from the event window.

        Args:
            events: Tuple of normalised events in the segment.
            context: Per-team match context.

        Returns:
            Dict mapping each context feature name to its value,
            or all ``None`` when no events exist.
        """
        if not events:
            return {name: None for name in _CONTEXT_NAMES}

        last = events[-1]

        # Score differential from the focal team's perspective.
        if context.team_is_home:
            score_diff = float(last.score_home - last.score_away)
        else:
            score_diff = float(last.score_away - last.score_home)

        n_total = len(events)
        n_team = sum(1 for e in events if e.team_id == context.team_id)
        possession_share = n_team / n_total

        possession_team = 1.0 if last.team_id == context.team_id else 0.0

        return {
            "t1_context_score_differential": score_diff,
            "t1_context_possession_share": possession_share,
            "t1_context_possession_team": possession_team,
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
