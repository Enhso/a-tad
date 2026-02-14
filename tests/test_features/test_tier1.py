"""Tests for Tier 1 spatial and temporal feature extractors.

Validates centroid computation, third/channel distributions, temporal
event-rate statistics, edge cases (no locations, single event), naming
conventions, and tier declarations.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tactical.adapters.schemas import MatchContext, NormalizedEvent
from tactical.features.tier1 import SpatialFeatureExtractor, TemporalFeatureExtractor

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
    period: int = 1,
    event_type: str = "pass",
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
        end_location=None,
        event_type=event_type,
        event_outcome="complete",
        under_pressure=False,
        body_part="right_foot",
        freeze_frame=None,
        score_home=0,
        score_away=0,
        team_is_home=True,
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
# Naming & tier tests
# ------------------------------------------------------------------


class TestNamingAndTier:
    """Verify naming conventions and tier declarations."""

    def test_feature_names_prefixed_spatial(self) -> None:
        """All spatial feature names start with ``t1_spatial_``."""
        ext = SpatialFeatureExtractor()
        for name in ext.feature_names:
            assert name.startswith("t1_spatial_"), name

    def test_feature_names_prefixed_temporal(self) -> None:
        """All temporal feature names start with ``t1_temporal_``."""
        ext = TemporalFeatureExtractor()
        for name in ext.feature_names:
            assert name.startswith("t1_temporal_"), name

    def test_tier_is_one_spatial(self) -> None:
        """Spatial extractor reports tier 1."""
        assert SpatialFeatureExtractor().tier == 1

    def test_tier_is_one_temporal(self) -> None:
        """Temporal extractor reports tier 1."""
        assert TemporalFeatureExtractor().tier == 1

    def test_extract_keys_match_feature_names_spatial(self) -> None:
        """Spatial extract() keys match declared feature_names."""
        ext = SpatialFeatureExtractor()
        events = (_evt("e01", timestamp=0.0, match_minute=0.0, location=(50.0, 50.0)),)
        result = ext.extract(events, _CONTEXT)
        assert set(result.keys()) == set(ext.feature_names)

    def test_extract_keys_match_feature_names_temporal(self) -> None:
        """Temporal extract() keys match declared feature_names."""
        ext = TemporalFeatureExtractor()
        events = (_evt("e01", timestamp=0.0, match_minute=0.0, location=(50.0, 50.0)),)
        result = ext.extract(events, _CONTEXT)
        assert set(result.keys()) == set(ext.feature_names)
