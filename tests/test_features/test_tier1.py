"""Tests for Tier 1 feature extractors.

Validates spatial, temporal, passing, carrying, defending, shooting,
and context extractors including edge cases, naming conventions, and
tier declarations.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tactical.adapters.schemas import MatchContext, NormalizedEvent
from tactical.features.tier1 import (
    CarryingFeatureExtractor,
    ContextFeatureExtractor,
    DefendingFeatureExtractor,
    PassingFeatureExtractor,
    ShootingFeatureExtractor,
    SpatialFeatureExtractor,
    TemporalFeatureExtractor,
)

if TYPE_CHECKING:
    from tactical.features.base import FeatureExtractor

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _evt(
    event_id: str,
    *,
    team_id: str = "team_a",
    timestamp: float,
    match_minute: float,
    location: tuple[float, float] | None,
    end_location: tuple[float, float] | None = None,
    period: int = 1,
    event_type: str = "pass",
    event_outcome: str = "complete",
    under_pressure: bool = False,
    body_part: str = "right_foot",
    score_home: int = 0,
    score_away: int = 0,
    team_is_home: bool = True,
) -> NormalizedEvent:
    """Build a :class:`NormalizedEvent` with sensible defaults."""
    return NormalizedEvent(
        event_id=event_id,
        match_id="match_001",
        team_id=team_id,
        player_id="player_01",
        period=period,
        timestamp=timestamp,
        match_minute=match_minute,
        location=location,
        end_location=end_location,
        event_type=event_type,
        event_outcome=event_outcome,
        under_pressure=under_pressure,
        body_part=body_part,
        freeze_frame=None,
        score_home=score_home,
        score_away=score_away,
        team_is_home=team_is_home,
    )


# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------

_CONTEXT = MatchContext(
    match_id="match_001",
    team_id="team_a",
    opponent_id="team_b",
    team_is_home=True,
    has_360=False,
)

_AWAY_CONTEXT = MatchContext(
    match_id="match_001",
    team_id="team_b",
    opponent_id="team_a",
    team_is_home=False,
    has_360=False,
)


@pytest.fixture()
def ten_events() -> tuple[NormalizedEvent, ...]:
    """Ten team_a events with known locations and timestamps.

    Locations are spread across all thirds and channels so zone
    percentages can be verified deterministically.

    Layout (x, y):
      - 3 events in defensive third   (x < 33.3)
      - 4 events in middle third      (33.3 <= x <= 66.7)
      - 3 events in attacking third   (x > 66.7)

      - 3 events in right channel     (y < 33.3)
      - 4 events in center channel    (33.3 <= y <= 66.7)
      - 3 events in left channel      (y > 66.7)

    Timestamps: 0, 1, 2, ..., 9 seconds (duration = 9 s).
    """
    specs: list[tuple[str, float, float, float, float]] = [
        # (id, timestamp, minute, x, y)
        ("e01", 0.0, 0.0, 10.0, 10.0),  # def, right
        ("e02", 1.0, 0.02, 20.0, 20.0),  # def, right
        ("e03", 2.0, 0.03, 30.0, 30.0),  # def, right
        ("e04", 3.0, 0.05, 40.0, 40.0),  # mid, center
        ("e05", 4.0, 0.07, 50.0, 50.0),  # mid, center
        ("e06", 5.0, 0.08, 55.0, 55.0),  # mid, center
        ("e07", 6.0, 0.10, 60.0, 60.0),  # mid, center
        ("e08", 7.0, 0.12, 70.0, 70.0),  # att, left
        ("e09", 8.0, 0.13, 80.0, 80.0),  # att, left
        ("e10", 9.0, 0.15, 90.0, 90.0),  # att, left
    ]
    return tuple(
        _evt(eid, timestamp=ts, match_minute=mm, location=(x, y))
        for eid, ts, mm, x, y in specs
    )


# ------------------------------------------------------------------
# Spatial tests
# ------------------------------------------------------------------


class TestSpatialFeatureExtractor:
    """Tests for :class:`SpatialFeatureExtractor`."""

    def test_spatial_centroid(self, ten_events: tuple[NormalizedEvent, ...]) -> None:
        """Centroid equals the arithmetic mean of x and y locations."""
        ext = SpatialFeatureExtractor()
        result = ext.extract(ten_events, _CONTEXT)

        xs = [10.0, 20.0, 30.0, 40.0, 50.0, 55.0, 60.0, 70.0, 80.0, 90.0]
        ys = [10.0, 20.0, 30.0, 40.0, 50.0, 55.0, 60.0, 70.0, 80.0, 90.0]
        expected_cx = sum(xs) / len(xs)
        expected_cy = sum(ys) / len(ys)

        assert result["t1_spatial_event_centroid_x"] == pytest.approx(expected_cx)
        assert result["t1_spatial_event_centroid_y"] == pytest.approx(expected_cy)

    def test_spatial_spread(self, ten_events: tuple[NormalizedEvent, ...]) -> None:
        """Spread equals population std of x and y locations."""
        ext = SpatialFeatureExtractor()
        result = ext.extract(ten_events, _CONTEXT)

        xs = np.array([10, 20, 30, 40, 50, 55, 60, 70, 80, 90], dtype=np.float64)
        ys = xs.copy()

        assert result["t1_spatial_event_spread_x"] == pytest.approx(float(np.std(xs)))
        assert result["t1_spatial_event_spread_y"] == pytest.approx(float(np.std(ys)))

    def test_spatial_thirds(self, ten_events: tuple[NormalizedEvent, ...]) -> None:
        """Zone percentages match the known event layout."""
        ext = SpatialFeatureExtractor()
        result = ext.extract(ten_events, _CONTEXT)

        assert result["t1_spatial_defensive_third_pct"] == pytest.approx(0.3)
        assert result["t1_spatial_middle_third_pct"] == pytest.approx(0.4)
        assert result["t1_spatial_attacking_third_pct"] == pytest.approx(0.3)

    def test_spatial_channels(self, ten_events: tuple[NormalizedEvent, ...]) -> None:
        """Channel percentages match the known event layout."""
        ext = SpatialFeatureExtractor()
        result = ext.extract(ten_events, _CONTEXT)

        assert result["t1_spatial_right_channel_pct"] == pytest.approx(0.3)
        assert result["t1_spatial_center_channel_pct"] == pytest.approx(0.4)
        assert result["t1_spatial_left_channel_pct"] == pytest.approx(0.3)

    def test_spatial_distance_to_goal(
        self, ten_events: tuple[NormalizedEvent, ...]
    ) -> None:
        """Average distance to goal matches hand-computed value."""
        ext = SpatialFeatureExtractor()
        result = ext.extract(ten_events, _CONTEXT)

        xs = [10, 20, 30, 40, 50, 55, 60, 70, 80, 90]
        ys = [10, 20, 30, 40, 50, 55, 60, 70, 80, 90]
        dists = [
            math.sqrt((x - 100) ** 2 + (y - 50) ** 2)
            for x, y in zip(xs, ys, strict=True)
        ]
        expected = sum(dists) / len(dists)

        assert result["t1_spatial_avg_distance_to_goal"] == pytest.approx(expected)

    def test_spatial_no_locations(self) -> None:
        """All features are None when every event lacks a location."""
        events = tuple(
            _evt(f"e{i:02d}", timestamp=float(i), match_minute=0.0, location=None)
            for i in range(5)
        )
        ext = SpatialFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert all(v is None for v in result.values())

    def test_spatial_filters_by_team(self) -> None:
        """Only events from the focal team contribute to features."""
        events = (
            _evt(
                "e01",
                team_id="team_a",
                timestamp=0.0,
                match_minute=0.0,
                location=(80.0, 50.0),
            ),
            _evt(
                "e02",
                team_id="team_b",
                timestamp=1.0,
                match_minute=0.0,
                location=(20.0, 50.0),
            ),
        )
        ext = SpatialFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        # Only team_a event at x=80 should be used
        assert result["t1_spatial_event_centroid_x"] == pytest.approx(80.0)


# ------------------------------------------------------------------
# Temporal tests
# ------------------------------------------------------------------


class TestTemporalFeatureExtractor:
    """Tests for :class:`TemporalFeatureExtractor`."""

    def test_temporal_event_rate(self, ten_events: tuple[NormalizedEvent, ...]) -> None:
        """10 events over 9 seconds yields rate of 10/9."""
        ext = TemporalFeatureExtractor()
        result = ext.extract(ten_events, _CONTEXT)

        assert result["t1_temporal_event_rate"] == pytest.approx(10.0 / 9.0)

    def test_temporal_inter_event_times(
        self, ten_events: tuple[NormalizedEvent, ...]
    ) -> None:
        """Known timestamps produce correct mean, std, and max gaps."""
        ext = TemporalFeatureExtractor()
        result = ext.extract(ten_events, _CONTEXT)

        # Timestamps 0..9 with step 1 -> all deltas are 1.0
        assert result["t1_temporal_mean_inter_event_time"] == pytest.approx(1.0)
        assert result["t1_temporal_std_inter_event_time"] == pytest.approx(0.0)
        assert result["t1_temporal_max_inter_event_time"] == pytest.approx(1.0)

    def test_temporal_inter_event_uneven(self) -> None:
        """Uneven timestamp gaps produce correct statistics."""
        events = (
            _evt("e01", timestamp=0.0, match_minute=0.0, location=(50.0, 50.0)),
            _evt("e02", timestamp=2.0, match_minute=0.03, location=(50.0, 50.0)),
            _evt("e03", timestamp=5.0, match_minute=0.08, location=(50.0, 50.0)),
            _evt("e04", timestamp=10.0, match_minute=0.17, location=(50.0, 50.0)),
        )
        ext = TemporalFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        deltas = np.array([2.0, 3.0, 5.0])
        assert result["t1_temporal_mean_inter_event_time"] == pytest.approx(
            float(np.mean(deltas))
        )
        assert result["t1_temporal_std_inter_event_time"] == pytest.approx(
            float(np.std(deltas))
        )
        assert result["t1_temporal_max_inter_event_time"] == pytest.approx(5.0)

    def test_temporal_match_minute(
        self, ten_events: tuple[NormalizedEvent, ...]
    ) -> None:
        """Match minute is the mean of all event match minutes."""
        ext = TemporalFeatureExtractor()
        result = ext.extract(ten_events, _CONTEXT)

        minutes = [0.0, 0.02, 0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.13, 0.15]
        expected = sum(minutes) / len(minutes)

        assert result["t1_temporal_match_minute"] == pytest.approx(expected)

    def test_temporal_period(self, ten_events: tuple[NormalizedEvent, ...]) -> None:
        """Period is the most common period value."""
        ext = TemporalFeatureExtractor()
        result = ext.extract(ten_events, _CONTEXT)

        # All events are period 1
        assert result["t1_temporal_period"] == pytest.approx(1.0)

    def test_temporal_single_event(self) -> None:
        """Single event returns None for rate and inter-event stats."""
        events = (_evt("e01", timestamp=5.0, match_minute=0.08, location=(50.0, 50.0)),)
        ext = TemporalFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t1_temporal_event_rate"] is None
        assert result["t1_temporal_mean_inter_event_time"] is None
        assert result["t1_temporal_std_inter_event_time"] is None
        assert result["t1_temporal_max_inter_event_time"] is None
        # match_minute and period are still computable
        assert result["t1_temporal_match_minute"] == pytest.approx(0.08)
        assert result["t1_temporal_period"] == pytest.approx(1.0)

    def test_temporal_zero_events(self) -> None:
        """No team events yields all None values."""
        # Only opponent events
        events = (
            _evt(
                "e01",
                team_id="team_b",
                timestamp=0.0,
                match_minute=0.0,
                location=(50.0, 50.0),
            ),
        )
        ext = TemporalFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert all(v is None for v in result.values())

    def test_temporal_filters_by_team(self) -> None:
        """Only events from the focal team contribute to features."""
        events = (
            _evt(
                "e01",
                team_id="team_a",
                timestamp=0.0,
                match_minute=0.0,
                location=(50.0, 50.0),
            ),
            _evt(
                "e02",
                team_id="team_b",
                timestamp=5.0,
                match_minute=0.08,
                location=(50.0, 50.0),
            ),
            _evt(
                "e03",
                team_id="team_a",
                timestamp=10.0,
                match_minute=0.17,
                location=(50.0, 50.0),
            ),
        )
        ext = TemporalFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        # Duration is 10s with 2 team_a events
        assert result["t1_temporal_event_rate"] == pytest.approx(2.0 / 10.0)


# ------------------------------------------------------------------
# Passing tests
# ------------------------------------------------------------------


class TestPassingFeatureExtractor:
    """Tests for :class:`PassingFeatureExtractor`."""

    def test_pass_completion_rate(self) -> None:
        """3 complete + 1 incomplete = 0.75 completion rate."""
        events = (
            _evt(
                "p01",
                timestamp=0.0,
                match_minute=0.0,
                location=(30.0, 40.0),
                end_location=(50.0, 40.0),
                event_outcome="complete",
            ),
            _evt(
                "p02",
                timestamp=1.0,
                match_minute=0.02,
                location=(40.0, 50.0),
                end_location=(55.0, 50.0),
                event_outcome="complete",
            ),
            _evt(
                "p03",
                timestamp=2.0,
                match_minute=0.03,
                location=(50.0, 45.0),
                end_location=(60.0, 45.0),
                event_outcome="complete",
            ),
            _evt(
                "p04",
                timestamp=3.0,
                match_minute=0.05,
                location=(55.0, 50.0),
                end_location=(65.0, 50.0),
                event_outcome="incomplete",
            ),
        )
        ext = PassingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t1_pass_count"] == pytest.approx(4.0)
        assert result["t1_pass_completion_rate"] == pytest.approx(0.75)

    def test_pass_no_passes_returns_none(self) -> None:
        """No passes in window returns count=0 and all others None."""
        events = (
            _evt(
                "c01",
                timestamp=0.0,
                match_minute=0.0,
                location=(50.0, 50.0),
                event_type="carry",
            ),
        )
        ext = PassingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t1_pass_count"] == pytest.approx(0.0)
        for name in ext.feature_names:
            if name != "t1_pass_count":
                assert result[name] is None, f"{name} should be None"

    def test_pass_progressive_counting(self) -> None:
        """Only passes with forward x-displacement > 10.0 are counted."""
        events = (
            # Forward 20 units -> progressive
            _evt(
                "p01",
                timestamp=0.0,
                match_minute=0.0,
                location=(30.0, 50.0),
                end_location=(50.0, 50.0),
            ),
            # Forward 5 units -> NOT progressive
            _evt(
                "p02",
                timestamp=1.0,
                match_minute=0.02,
                location=(40.0, 50.0),
                end_location=(45.0, 50.0),
            ),
            # Forward 15 units -> progressive
            _evt(
                "p03",
                timestamp=2.0,
                match_minute=0.03,
                location=(50.0, 50.0),
                end_location=(65.0, 50.0),
            ),
            # Backward -10 units -> NOT progressive
            _evt(
                "p04",
                timestamp=3.0,
                match_minute=0.05,
                location=(60.0, 50.0),
                end_location=(50.0, 50.0),
            ),
        )
        ext = PassingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t1_pass_progressive_count"] == pytest.approx(2.0)

    def test_pass_directness(self) -> None:
        """Known displacements produce correct directness value."""
        events = (
            # Pure forward pass: dx=20, dy=0, length=20 -> dir=1.0
            _evt(
                "p01",
                timestamp=0.0,
                match_minute=0.0,
                location=(30.0, 50.0),
                end_location=(50.0, 50.0),
            ),
            # Pure lateral pass: dx=0, dy=20, length=20 -> dir=0.0
            _evt(
                "p02",
                timestamp=1.0,
                match_minute=0.02,
                location=(50.0, 30.0),
                end_location=(50.0, 50.0),
            ),
            # Pure backward pass: dx=-20, dy=0, length=20 -> dir=-1.0
            _evt(
                "p03",
                timestamp=2.0,
                match_minute=0.03,
                location=(70.0, 50.0),
                end_location=(50.0, 50.0),
            ),
        )
        ext = PassingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        # mean(1.0, 0.0, -1.0) = 0.0
        assert result["t1_pass_directness"] == pytest.approx(0.0)

    def test_pass_backward_ratio(self) -> None:
        """Backward ratio counts only passes with negative x-displacement."""
        events = (
            # Forward
            _evt(
                "p01",
                timestamp=0.0,
                match_minute=0.0,
                location=(30.0, 50.0),
                end_location=(50.0, 50.0),
            ),
            # Backward
            _evt(
                "p02",
                timestamp=1.0,
                match_minute=0.02,
                location=(50.0, 50.0),
                end_location=(40.0, 50.0),
            ),
            # Lateral (dx=0) -> NOT backward
            _evt(
                "p03",
                timestamp=2.0,
                match_minute=0.03,
                location=(50.0, 30.0),
                end_location=(50.0, 50.0),
            ),
            # Backward
            _evt(
                "p04",
                timestamp=3.0,
                match_minute=0.05,
                location=(60.0, 50.0),
                end_location=(45.0, 50.0),
            ),
        )
        ext = PassingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        # 2 backward out of 4 = 0.5
        assert result["t1_pass_backward_ratio"] == pytest.approx(0.5)

    def test_pass_switch_count(self) -> None:
        """Only passes with abs(y-displacement) > 50 are switches."""
        events = (
            # dy = 55 -> switch
            _evt(
                "p01",
                timestamp=0.0,
                match_minute=0.0,
                location=(50.0, 10.0),
                end_location=(55.0, 65.0),
            ),
            # dy = 30 -> NOT a switch
            _evt(
                "p02",
                timestamp=1.0,
                match_minute=0.02,
                location=(50.0, 30.0),
                end_location=(55.0, 60.0),
            ),
            # dy = -60 -> switch
            _evt(
                "p03",
                timestamp=2.0,
                match_minute=0.03,
                location=(50.0, 80.0),
                end_location=(55.0, 20.0),
            ),
        )
        ext = PassingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t1_pass_switch_count"] == pytest.approx(2.0)

    def test_pass_height_features_are_none(self) -> None:
        """Height features are None (pass_height not in schema)."""
        events = (
            _evt(
                "p01",
                timestamp=0.0,
                match_minute=0.0,
                location=(30.0, 50.0),
                end_location=(50.0, 50.0),
            ),
        )
        ext = PassingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t1_pass_height_ground_pct"] is None
        assert result["t1_pass_height_low_pct"] is None
        assert result["t1_pass_height_high_pct"] is None

    def test_pass_length_mean_and_std(self) -> None:
        """Pass length statistics match hand-computed values."""
        events = (
            # length = sqrt(20^2 + 0^2) = 20
            _evt(
                "p01",
                timestamp=0.0,
                match_minute=0.0,
                location=(30.0, 50.0),
                end_location=(50.0, 50.0),
            ),
            # length = sqrt(0^2 + 10^2) = 10
            _evt(
                "p02",
                timestamp=1.0,
                match_minute=0.02,
                location=(50.0, 40.0),
                end_location=(50.0, 50.0),
            ),
        )
        ext = PassingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        lengths = np.array([20.0, 10.0])
        assert result["t1_pass_length_mean"] == pytest.approx(float(np.mean(lengths)))
        assert result["t1_pass_length_std"] == pytest.approx(float(np.std(lengths)))

    def test_pass_filters_by_team(self) -> None:
        """Only focal team passes contribute."""
        events = (
            _evt(
                "p01",
                team_id="team_a",
                timestamp=0.0,
                match_minute=0.0,
                location=(30.0, 50.0),
                end_location=(50.0, 50.0),
            ),
            _evt(
                "p02",
                team_id="team_b",
                timestamp=1.0,
                match_minute=0.02,
                location=(70.0, 50.0),
                end_location=(80.0, 50.0),
            ),
        )
        ext = PassingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t1_pass_count"] == pytest.approx(1.0)


# ------------------------------------------------------------------
# Carrying tests
# ------------------------------------------------------------------


class TestCarryingFeatureExtractor:
    """Tests for :class:`CarryingFeatureExtractor`."""

    def test_carry_distance(self) -> None:
        """Known carry locations produce correct total and mean distance."""
        events = (
            # distance = sqrt(10^2 + 0^2) = 10
            _evt(
                "c01",
                timestamp=0.0,
                match_minute=0.0,
                location=(40.0, 50.0),
                end_location=(50.0, 50.0),
                event_type="carry",
            ),
            # distance = sqrt(0^2 + 20^2) = 20
            _evt(
                "c02",
                timestamp=1.0,
                match_minute=0.02,
                location=(50.0, 30.0),
                end_location=(50.0, 50.0),
                event_type="carry",
            ),
            # distance = sqrt(30^2 + 40^2) = 50
            _evt(
                "c03",
                timestamp=2.0,
                match_minute=0.03,
                location=(20.0, 10.0),
                end_location=(50.0, 50.0),
                event_type="carry",
            ),
        )
        ext = CarryingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t1_carry_count"] == pytest.approx(3.0)
        assert result["t1_carry_distance_total"] == pytest.approx(80.0)
        assert result["t1_carry_distance_mean"] == pytest.approx(80.0 / 3.0)

    def test_carry_progressive_count(self) -> None:
        """Only carries with forward x-displacement > 10.0 are counted."""
        events = (
            # dx = 15 -> progressive
            _evt(
                "c01",
                timestamp=0.0,
                match_minute=0.0,
                location=(30.0, 50.0),
                end_location=(45.0, 50.0),
                event_type="carry",
            ),
            # dx = 5 -> NOT progressive
            _evt(
                "c02",
                timestamp=1.0,
                match_minute=0.02,
                location=(45.0, 50.0),
                end_location=(50.0, 50.0),
                event_type="carry",
            ),
        )
        ext = CarryingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t1_carry_progressive_count"] == pytest.approx(1.0)

    def test_carry_directness(self) -> None:
        """Directness is mean(x_displacement / distance)."""
        events = (
            # dx=20, dy=0, dist=20 -> dir=1.0
            _evt(
                "c01",
                timestamp=0.0,
                match_minute=0.0,
                location=(30.0, 50.0),
                end_location=(50.0, 50.0),
                event_type="carry",
            ),
            # dx=0, dy=20, dist=20 -> dir=0.0
            _evt(
                "c02",
                timestamp=1.0,
                match_minute=0.02,
                location=(50.0, 30.0),
                end_location=(50.0, 50.0),
                event_type="carry",
            ),
        )
        ext = CarryingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        # mean(1.0, 0.0) = 0.5
        assert result["t1_carry_directness"] == pytest.approx(0.5)

    def test_carry_no_carries_returns_zero_count(self) -> None:
        """No carries produces count=0 and distance features as None."""
        events = (
            _evt(
                "p01",
                timestamp=0.0,
                match_minute=0.0,
                location=(50.0, 50.0),
            ),
        )
        ext = CarryingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t1_carry_count"] == pytest.approx(0.0)
        assert result["t1_carry_distance_total"] is None
        assert result["t1_carry_distance_mean"] is None
        assert result["t1_carry_directness"] is None
        assert result["t1_carry_progressive_count"] == pytest.approx(0.0)
        assert result["t1_carry_dribble_attempt_count"] == pytest.approx(0.0)

    def test_carry_dribble_attempt_is_carry_count(self) -> None:
        """Dribble attempt count equals carry count (v1 proxy)."""
        events = (
            _evt(
                "c01",
                timestamp=0.0,
                match_minute=0.0,
                location=(40.0, 50.0),
                end_location=(50.0, 50.0),
                event_type="carry",
            ),
            _evt(
                "c02",
                timestamp=1.0,
                match_minute=0.02,
                location=(50.0, 50.0),
                end_location=(55.0, 50.0),
                event_type="carry",
            ),
        )
        ext = CarryingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t1_carry_dribble_attempt_count"] == pytest.approx(2.0)
        assert result["t1_carry_dribble_success_rate"] is None


# ------------------------------------------------------------------
# Defending tests
# ------------------------------------------------------------------


class TestDefendingFeatureExtractor:
    """Tests for :class:`DefendingFeatureExtractor`."""

    def test_defend_action_height_flipped(self) -> None:
        """Defensive event at x=80 yields action height 20.0 (flipped)."""
        events = (
            _evt(
                "d01",
                timestamp=0.0,
                match_minute=0.0,
                location=(80.0, 50.0),
                event_type="tackle",
            ),
        )
        ext = DefendingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t1_defend_defensive_action_height"] == pytest.approx(20.0)

    def test_defend_action_height_multiple(self) -> None:
        """Action height is mean of flipped x across multiple types."""
        events = (
            # tackle at x=80 -> flipped 20
            _evt(
                "d01",
                timestamp=0.0,
                match_minute=0.0,
                location=(80.0, 50.0),
                event_type="tackle",
            ),
            # interception at x=60 -> flipped 40
            _evt(
                "d02",
                timestamp=1.0,
                match_minute=0.02,
                location=(60.0, 50.0),
                event_type="interception",
            ),
            # clearance at x=90 -> flipped 10
            _evt(
                "d03",
                timestamp=2.0,
                match_minute=0.03,
                location=(90.0, 50.0),
                event_type="clearance",
            ),
            # block at x=70 -> flipped 30
            _evt(
                "d04",
                timestamp=3.0,
                match_minute=0.05,
                location=(70.0, 50.0),
                event_type="block",
            ),
        )
        ext = DefendingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        # mean(20, 40, 10, 30) = 25.0
        assert result["t1_defend_defensive_action_height"] == pytest.approx(25.0)

    def test_defend_pressure_rate(self) -> None:
        """3 pressures out of 10 total events = 0.3 pressure rate."""
        pressures = [
            _evt(
                f"pr{i}",
                timestamp=float(i),
                match_minute=float(i) / 60.0,
                location=(50.0, 50.0),
                event_type="pressure",
            )
            for i in range(3)
        ]
        passes = [
            _evt(
                f"pa{i}",
                timestamp=float(i + 3),
                match_minute=float(i + 3) / 60.0,
                location=(50.0, 50.0),
            )
            for i in range(4)
        ]
        opp_passes = [
            _evt(
                f"op{i}",
                team_id="team_b",
                timestamp=float(i + 7),
                match_minute=float(i + 7) / 60.0,
                location=(50.0, 50.0),
            )
            for i in range(3)
        ]
        events = tuple(pressures + passes + opp_passes)
        ext = DefendingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t1_defend_pressure_count"] == pytest.approx(3.0)
        assert result["t1_defend_pressure_rate"] == pytest.approx(0.3)

    def test_defend_counts(self) -> None:
        """Individual defensive action counts are correct."""
        events = (
            _evt(
                "d01",
                timestamp=0.0,
                match_minute=0.0,
                location=(50.0, 50.0),
                event_type="tackle",
                event_outcome="won",
            ),
            _evt(
                "d02",
                timestamp=1.0,
                match_minute=0.02,
                location=(50.0, 50.0),
                event_type="duel",
                event_outcome="lost",
            ),
            _evt(
                "d03",
                timestamp=2.0,
                match_minute=0.03,
                location=(50.0, 50.0),
                event_type="interception",
            ),
            _evt(
                "d04",
                timestamp=3.0,
                match_minute=0.05,
                location=(50.0, 50.0),
                event_type="clearance",
            ),
            _evt(
                "d05",
                timestamp=4.0,
                match_minute=0.07,
                location=(50.0, 50.0),
                event_type="foul_committed",
            ),
        )
        ext = DefendingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        # tackle + duel both counted as tackles
        assert result["t1_defend_tackle_count"] == pytest.approx(2.0)
        assert result["t1_defend_interception_count"] == pytest.approx(1.0)
        assert result["t1_defend_clearance_count"] == pytest.approx(1.0)
        assert result["t1_defend_foul_count"] == pytest.approx(1.0)

    def test_defend_tackle_success_rate(self) -> None:
        """Tackle success rate counts Won and Success* outcomes."""
        events = (
            # won -> success
            _evt(
                "d01",
                timestamp=0.0,
                match_minute=0.0,
                location=(50.0, 50.0),
                event_type="tackle",
                event_outcome="won",
            ),
            # success_in_play -> success
            _evt(
                "d02",
                timestamp=1.0,
                match_minute=0.02,
                location=(50.0, 50.0),
                event_type="duel",
                event_outcome="success_in_play",
            ),
            # lost -> NOT success
            _evt(
                "d03",
                timestamp=2.0,
                match_minute=0.03,
                location=(50.0, 50.0),
                event_type="tackle",
                event_outcome="lost",
            ),
            # success_out -> success
            _evt(
                "d04",
                timestamp=3.0,
                match_minute=0.05,
                location=(50.0, 50.0),
                event_type="duel",
                event_outcome="success_out",
            ),
        )
        ext = DefendingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        # 3 successful out of 4 tackles/duels
        assert result["t1_defend_tackle_success_rate"] == pytest.approx(0.75)

    def test_defend_tackle_success_rate_all_outcomes(self) -> None:
        """All four recognised success outcomes are counted."""
        events = (
            _evt(
                "d01",
                timestamp=0.0,
                match_minute=0.0,
                location=(50.0, 50.0),
                event_type="tackle",
                event_outcome="won",
            ),
            _evt(
                "d02",
                timestamp=1.0,
                match_minute=0.02,
                location=(50.0, 50.0),
                event_type="tackle",
                event_outcome="success",
            ),
            _evt(
                "d03",
                timestamp=2.0,
                match_minute=0.03,
                location=(50.0, 50.0),
                event_type="duel",
                event_outcome="success_in_play",
            ),
            _evt(
                "d04",
                timestamp=3.0,
                match_minute=0.05,
                location=(50.0, 50.0),
                event_type="duel",
                event_outcome="success_out",
            ),
        )
        ext = DefendingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t1_defend_tackle_success_rate"] == pytest.approx(1.0)

    def test_defend_tackle_success_rate_no_tackles(self) -> None:
        """No tackles yields None for tackle success rate."""
        events = (
            _evt(
                "p01",
                timestamp=0.0,
                match_minute=0.0,
                location=(50.0, 50.0),
            ),
        )
        ext = DefendingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t1_defend_tackle_success_rate"] is None

    def test_defend_under_pressure_pct(self) -> None:
        """Under-pressure percentage computed from team events only."""
        events = (
            _evt(
                "d01",
                timestamp=0.0,
                match_minute=0.0,
                location=(50.0, 50.0),
                under_pressure=True,
            ),
            _evt(
                "d02",
                timestamp=1.0,
                match_minute=0.02,
                location=(50.0, 50.0),
                under_pressure=False,
            ),
            _evt(
                "d03",
                timestamp=2.0,
                match_minute=0.03,
                location=(50.0, 50.0),
                under_pressure=True,
            ),
            _evt(
                "d04",
                timestamp=3.0,
                match_minute=0.05,
                location=(50.0, 50.0),
                under_pressure=False,
            ),
        )
        ext = DefendingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        # 2 of 4 under pressure
        assert result["t1_defend_under_pressure_pct"] == pytest.approx(0.5)

    def test_defend_no_defensive_actions_height_none(self) -> None:
        """No defensive actions yields None for action height."""
        events = (
            _evt(
                "p01",
                timestamp=0.0,
                match_minute=0.0,
                location=(50.0, 50.0),
            ),
        )
        ext = DefendingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t1_defend_defensive_action_height"] is None


# ------------------------------------------------------------------
# Shooting tests
# ------------------------------------------------------------------


class TestShootingFeatureExtractor:
    """Tests for :class:`ShootingFeatureExtractor`."""

    def test_shoot_count(self) -> None:
        """Two shots are counted correctly."""
        events = (
            _evt(
                "s01",
                timestamp=0.0,
                match_minute=0.0,
                location=(85.0, 50.0),
                event_type="shot",
                event_outcome="saved",
            ),
            _evt(
                "s02",
                timestamp=1.0,
                match_minute=0.02,
                location=(80.0, 45.0),
                event_type="shot",
                event_outcome="goal",
            ),
        )
        ext = ShootingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t1_shoot_count"] == pytest.approx(2.0)

    def test_shoot_distance_mean(self) -> None:
        """Mean shot distance matches hand-computed Euclidean values."""
        events = (
            # distance to (100,50) = sqrt(15^2 + 0^2) = 15.0
            _evt(
                "s01",
                timestamp=0.0,
                match_minute=0.0,
                location=(85.0, 50.0),
                event_type="shot",
            ),
            # distance to (100,50) = sqrt(20^2 + 10^2) = sqrt(500)
            _evt(
                "s02",
                timestamp=1.0,
                match_minute=0.02,
                location=(80.0, 40.0),
                event_type="shot",
            ),
        )
        ext = ShootingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        d1 = 15.0
        d2 = math.sqrt(500.0)
        expected = (d1 + d2) / 2.0
        assert result["t1_shoot_distance_mean"] == pytest.approx(expected)

    def test_shoot_no_shots(self) -> None:
        """No shots returns count=0 and distance_mean=None."""
        events = (
            _evt(
                "p01",
                timestamp=0.0,
                match_minute=0.0,
                location=(50.0, 50.0),
            ),
        )
        ext = ShootingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t1_shoot_count"] == pytest.approx(0.0)
        assert result["t1_shoot_distance_mean"] is None

    def test_shoot_filters_by_team(self) -> None:
        """Only focal team shots are counted."""
        events = (
            _evt(
                "s01",
                team_id="team_a",
                timestamp=0.0,
                match_minute=0.0,
                location=(85.0, 50.0),
                event_type="shot",
            ),
            _evt(
                "s02",
                team_id="team_b",
                timestamp=1.0,
                match_minute=0.02,
                location=(85.0, 50.0),
                event_type="shot",
            ),
        )
        ext = ShootingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t1_shoot_count"] == pytest.approx(1.0)


# ------------------------------------------------------------------
# Context tests
# ------------------------------------------------------------------


class TestContextFeatureExtractor:
    """Tests for :class:`ContextFeatureExtractor`."""

    def test_context_score_differential_home(self) -> None:
        """Home team winning 2-1 yields differential +1."""
        events = (
            _evt(
                "e01",
                timestamp=0.0,
                match_minute=0.0,
                location=(50.0, 50.0),
                score_home=2,
                score_away=1,
            ),
        )
        ext = ContextFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t1_context_score_differential"] == pytest.approx(1.0)

    def test_context_score_differential_away(self) -> None:
        """Away team perspective on 2-1 home lead yields differential -1."""
        events = (
            _evt(
                "e01",
                team_id="team_b",
                timestamp=0.0,
                match_minute=0.0,
                location=(50.0, 50.0),
                score_home=2,
                score_away=1,
                team_is_home=False,
            ),
        )
        ext = ContextFeatureExtractor()
        result = ext.extract(events, _AWAY_CONTEXT)

        assert result["t1_context_score_differential"] == pytest.approx(-1.0)

    def test_context_possession_share(self) -> None:
        """6 team_a events out of 10 total = 0.6 possession share."""
        team_a_events = [
            _evt(
                f"a{i}",
                team_id="team_a",
                timestamp=float(i),
                match_minute=float(i) / 60.0,
                location=(50.0, 50.0),
            )
            for i in range(6)
        ]
        team_b_events = [
            _evt(
                f"b{i}",
                team_id="team_b",
                timestamp=float(i + 6),
                match_minute=float(i + 6) / 60.0,
                location=(50.0, 50.0),
            )
            for i in range(4)
        ]
        events = tuple(team_a_events + team_b_events)
        ext = ContextFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t1_context_possession_share"] == pytest.approx(0.6)

    def test_context_possession_team(self) -> None:
        """Possession team indicator reflects the last event's team."""
        events_own = (
            _evt(
                "e01",
                team_id="team_b",
                timestamp=0.0,
                match_minute=0.0,
                location=(50.0, 50.0),
            ),
            _evt(
                "e02",
                team_id="team_a",
                timestamp=1.0,
                match_minute=0.02,
                location=(50.0, 50.0),
            ),
        )
        ext = ContextFeatureExtractor()
        result_own = ext.extract(events_own, _CONTEXT)
        assert result_own["t1_context_possession_team"] == pytest.approx(1.0)

        events_opp = (
            _evt(
                "e01",
                team_id="team_a",
                timestamp=0.0,
                match_minute=0.0,
                location=(50.0, 50.0),
            ),
            _evt(
                "e02",
                team_id="team_b",
                timestamp=1.0,
                match_minute=0.02,
                location=(50.0, 50.0),
            ),
        )
        result_opp = ext.extract(events_opp, _CONTEXT)
        assert result_opp["t1_context_possession_team"] == pytest.approx(0.0)

    def test_context_empty_events(self) -> None:
        """No events yields all None."""
        events: tuple[NormalizedEvent, ...] = ()
        ext = ContextFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert all(v is None for v in result.values())

    def test_context_score_at_last_event(self) -> None:
        """Score differential uses the last event's score state."""
        events = (
            _evt(
                "e01",
                timestamp=0.0,
                match_minute=0.0,
                location=(50.0, 50.0),
                score_home=0,
                score_away=0,
            ),
            _evt(
                "e02",
                timestamp=1.0,
                match_minute=0.02,
                location=(50.0, 50.0),
                score_home=1,
                score_away=0,
            ),
            _evt(
                "e03",
                timestamp=2.0,
                match_minute=0.03,
                location=(50.0, 50.0),
                score_home=1,
                score_away=2,
            ),
        )
        ext = ContextFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        # Last event: home=1, away=2, home team -> -1
        assert result["t1_context_score_differential"] == pytest.approx(-1.0)


# ------------------------------------------------------------------
# Naming & tier tests (all 7 extractors)
# ------------------------------------------------------------------

_ALL_EXTRACTORS = [
    SpatialFeatureExtractor,
    TemporalFeatureExtractor,
    PassingFeatureExtractor,
    CarryingFeatureExtractor,
    DefendingFeatureExtractor,
    ShootingFeatureExtractor,
    ContextFeatureExtractor,
]

_EXPECTED_PREFIXES = {
    "SpatialFeatureExtractor": "t1_spatial_",
    "TemporalFeatureExtractor": "t1_temporal_",
    "PassingFeatureExtractor": "t1_pass_",
    "CarryingFeatureExtractor": "t1_carry_",
    "DefendingFeatureExtractor": "t1_defend_",
    "ShootingFeatureExtractor": "t1_shoot_",
    "ContextFeatureExtractor": "t1_context_",
}


class TestNamingAndTier:
    """Verify naming conventions and tier declarations for all extractors."""

    @pytest.mark.parametrize(
        "extractor_cls",
        _ALL_EXTRACTORS,
        ids=[c.__name__ for c in _ALL_EXTRACTORS],
    )
    def test_tier_is_one(self, extractor_cls: type) -> None:
        """All Tier 1 extractors report tier=1."""
        assert extractor_cls().tier == 1

    @pytest.mark.parametrize(
        "extractor_cls",
        _ALL_EXTRACTORS,
        ids=[c.__name__ for c in _ALL_EXTRACTORS],
    )
    def test_feature_names_prefixed(self, extractor_cls: type) -> None:
        """All feature names use the correct t1_ prefix."""
        ext = extractor_cls()
        prefix = _EXPECTED_PREFIXES[extractor_cls.__name__]
        for name in ext.feature_names:
            assert name.startswith(prefix), (
                f"{extractor_cls.__name__}: {name!r} missing prefix {prefix!r}"
            )

    @pytest.mark.parametrize(
        "extractor_cls",
        _ALL_EXTRACTORS,
        ids=[c.__name__ for c in _ALL_EXTRACTORS],
    )
    def test_extract_keys_match_feature_names(self, extractor_cls: type) -> None:
        """Extract keys match declared feature_names for each extractor."""
        ext = extractor_cls()
        # Provide a minimal event of each needed type
        events = (
            _evt(
                "e01",
                timestamp=0.0,
                match_minute=0.0,
                location=(50.0, 50.0),
                end_location=(55.0, 50.0),
            ),
            _evt(
                "e02",
                timestamp=1.0,
                match_minute=0.02,
                location=(50.0, 50.0),
                end_location=(55.0, 50.0),
                event_type="carry",
            ),
            _evt(
                "e03",
                timestamp=2.0,
                match_minute=0.03,
                location=(50.0, 50.0),
                event_type="shot",
                event_outcome="saved",
            ),
            _evt(
                "e04",
                timestamp=3.0,
                match_minute=0.05,
                location=(50.0, 50.0),
                event_type="pressure",
            ),
            _evt(
                "e05",
                timestamp=4.0,
                match_minute=0.07,
                location=(50.0, 50.0),
                event_type="tackle",
            ),
        )
        result = ext.extract(events, _CONTEXT)
        assert set(result.keys()) == set(ext.feature_names)

    def test_no_duplicate_names_across_extractors(self) -> None:
        """No two extractors share a feature name."""
        instances: list[FeatureExtractor] = [
            SpatialFeatureExtractor(),
            TemporalFeatureExtractor(),
            PassingFeatureExtractor(),
            CarryingFeatureExtractor(),
            DefendingFeatureExtractor(),
            ShootingFeatureExtractor(),
            ContextFeatureExtractor(),
        ]
        all_names: list[str] = []
        for ext in instances:
            all_names.extend(ext.feature_names)
        assert len(all_names) == len(set(all_names))
