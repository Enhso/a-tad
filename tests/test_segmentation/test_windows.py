"""Tests for time-based sliding-window segmentation.

Covers window creation, period-boundary isolation, overlap sharing,
minimum-event filtering, per-event assignment correctness, empty-input
error handling, and custom configuration behaviour.
"""

from __future__ import annotations

import pytest

from tactical.adapters.schemas import NormalizedEvent
from tactical.config import WindowConfig
from tactical.exceptions import SegmentationError
from tactical.segmentation.windows import TimeWindow, create_time_windows

# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------


def _evt(
    event_id: str,
    *,
    match_id: str = "match_001",
    team_id: str = "team_a",
    player_id: str = "player_01",
    period: int = 1,
    timestamp: float,
    match_minute: float,
    location: tuple[float, float] = (50.0, 50.0),
    end_location: tuple[float, float] | None = None,
    event_type: str = "pass",
    event_outcome: str = "complete",
    under_pressure: bool = False,
    body_part: str = "right_foot",
) -> NormalizedEvent:
    """Build a :class:`NormalizedEvent` with sensible defaults."""
    return NormalizedEvent(
        event_id=event_id,
        match_id=match_id,
        team_id=team_id,
        player_id=player_id,
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
        score_home=0,
        score_away=0,
        team_is_home=True,
    )


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def default_config() -> WindowConfig:
    """Default window config: 15 s window, 5 s overlap, min 3 events."""
    return WindowConfig()


@pytest.fixture()
def twenty_events_60s() -> list[NormalizedEvent]:
    """20 events evenly spaced over 60 s (0, 3, 6, ..., 57) in period 1."""
    return [
        _evt(
            f"e_{i:02d}",
            timestamp=float(i * 3),
            match_minute=float(i * 3) / 60.0,
        )
        for i in range(20)
    ]


@pytest.fixture()
def two_period_events() -> list[NormalizedEvent]:
    """Events split across two periods, 5 per period, 2 s apart."""
    period_1 = [
        _evt(
            f"p1_{i}",
            period=1,
            timestamp=float(i * 2),
            match_minute=float(i * 2) / 60.0,
        )
        for i in range(5)
    ]
    period_2 = [
        _evt(
            f"p2_{i}",
            period=2,
            timestamp=float(i * 2),
            match_minute=45.0 + float(i * 2) / 60.0,
        )
        for i in range(5)
    ]
    return period_1 + period_2


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestBasicWindowing:
    """Verify window count and boundary values with default config."""

    def test_basic_windowing(
        self,
        twenty_events_60s: list[NormalizedEvent],
        default_config: WindowConfig,
    ) -> None:
        """20 events over 60 s with 15 s/5 s overlap -> 6 windows."""
        # step = 15 - 5 = 10
        # min_ts=0, max_ts=57
        # starts: 0, 10, 20, 30, 40, 50  (50 <= 57)
        # Window [0,15):  ts 0,3,6,9,12       -> 5 events
        # Window [10,25): ts 12,15,18,21,24   -> 5 events
        # Window [20,35): ts 21,24,27,30,33   -> 5 events
        # Window [30,45): ts 30,33,36,39,42   -> 5 events
        # Window [40,55): ts 42,45,48,51,54   -> 5 events
        # Window [50,65): ts 51,54,57          -> 3 events (>= min_events)
        windows = create_time_windows(twenty_events_60s, default_config)

        assert len(windows) == 6

        # All windows in period 1
        assert all(w.period == 1 for w in windows)

        # First window boundaries
        assert windows[0].start_time == pytest.approx(0.0)
        assert windows[0].end_time == pytest.approx(15.0)

        # Last window boundaries
        assert windows[-1].start_time == pytest.approx(50.0)
        assert windows[-1].end_time == pytest.approx(65.0)

        # Monotonic start times
        for i in range(len(windows) - 1):
            assert windows[i].start_time < windows[i + 1].start_time


class TestPeriodBoundaries:
    """Windows must never span period boundaries."""

    def test_window_boundaries_dont_span_periods(
        self,
        two_period_events: list[NormalizedEvent],
    ) -> None:
        """No window contains events from both period 1 and period 2."""
        config = WindowConfig(window_seconds=10.0, overlap_seconds=0.0, min_events=3)
        windows = create_time_windows(two_period_events, config)

        for window in windows:
            periods_in_window = {e.period for e in window.events}
            assert len(periods_in_window) == 1
            assert window.period in periods_in_window


class TestWindowOverlap:
    """Consecutive windows share events in the overlap region."""

    def test_window_overlap(
        self,
        twenty_events_60s: list[NormalizedEvent],
        default_config: WindowConfig,
    ) -> None:
        """Adjacent windows share the event at t=12 (overlap [10,15))."""
        windows = create_time_windows(twenty_events_60s, default_config)

        first_ids = {e.event_id for e in windows[0].events}
        second_ids = {e.event_id for e in windows[1].events}
        shared = first_ids & second_ids

        # Event at t=12 (e_04) should appear in both [0,15) and [10,25)
        assert len(shared) > 0
        assert "e_04" in shared


class TestMinEvents:
    """Sparse regions with fewer than min_events produce no window."""

    def test_window_min_events(self) -> None:
        """Two isolated events cannot form a window with min_events=3."""
        sparse = [
            _evt("s_0", timestamp=0.0, match_minute=0.0),
            _evt("s_1", timestamp=100.0, match_minute=1.67),
        ]
        config = WindowConfig(window_seconds=15.0, overlap_seconds=5.0, min_events=3)
        windows = create_time_windows(sparse, config)

        assert len(windows) == 0


class TestEventAssignment:
    """Each event appears in exactly the correct set of windows."""

    def test_window_event_assignment(self) -> None:
        """Events land in every window whose [start, end) contains them."""
        # 6 events at 0, 5, 10, 15, 20, 25 -- window 10 s, step 10 s
        events = [
            _evt(f"a_{i}", timestamp=float(i * 5), match_minute=float(i * 5) / 60.0)
            for i in range(6)
        ]
        config = WindowConfig(window_seconds=10.0, overlap_seconds=0.0, min_events=2)
        # step = 10, starts at 0, 10, 20
        # [0,10): ts 0, 5       -> 2 events
        # [10,20): ts 10, 15    -> 2 events
        # [20,30): ts 20, 25    -> 2 events
        windows = create_time_windows(events, config)

        assert len(windows) == 3

        assert {e.event_id for e in windows[0].events} == {"a_0", "a_1"}
        assert {e.event_id for e in windows[1].events} == {"a_2", "a_3"}
        assert {e.event_id for e in windows[2].events} == {"a_4", "a_5"}

    def test_event_in_multiple_overlapping_windows(self) -> None:
        """An event in the overlap zone appears in two consecutive windows."""
        # 10 events at 0..9, window=6, overlap=3, step=3
        events = [
            _evt(f"m_{i}", timestamp=float(i), match_minute=float(i) / 60.0)
            for i in range(10)
        ]
        config = WindowConfig(window_seconds=6.0, overlap_seconds=3.0, min_events=3)
        windows = create_time_windows(events, config)

        # Event at t=3 is in [0,6) and [3,9)
        windows_with_m3 = [
            w for w in windows if "m_3" in {e.event_id for e in w.events}
        ]
        assert len(windows_with_m3) >= 2


class TestEmptyEvents:
    """An empty event list must raise SegmentationError."""

    def test_empty_events_raises(self, default_config: WindowConfig) -> None:
        """Passing an empty list raises SegmentationError."""
        with pytest.raises(SegmentationError):
            create_time_windows([], default_config)


class TestCustomConfig:
    """Non-default configuration produces the expected window count."""

    def test_custom_config(self) -> None:
        """30 s window, 10 s overlap -> step 20 s over 60 s of events."""
        events = [
            _evt(
                f"c_{i:02d}",
                timestamp=float(i * 3),
                match_minute=float(i * 3) / 60.0,
            )
            for i in range(20)
        ]
        config = WindowConfig(window_seconds=30.0, overlap_seconds=10.0, min_events=3)
        # step = 20, min_ts=0, max_ts=57
        # starts: 0, 20, 40  (40 <= 57, 60 > 57 stop)
        # [0,30):  ts 0,3,...,27 -> 10 events
        # [20,50): ts 21,24,...,48 -> 10 events
        # [40,70): ts 42,45,48,51,54,57 -> 6 events
        windows = create_time_windows(events, config)

        assert len(windows) == 3
        assert windows[0].start_time == pytest.approx(0.0)
        assert windows[0].end_time == pytest.approx(30.0)
        assert windows[1].start_time == pytest.approx(20.0)
        assert windows[2].start_time == pytest.approx(40.0)


class TestTimeWindowDataclass:
    """Verify TimeWindow is frozen and slotted."""

    def test_frozen(self) -> None:
        """TimeWindow instances are immutable."""
        event = _evt("f_0", timestamp=0.0, match_minute=0.0)
        tw = TimeWindow(
            match_id="m",
            period=1,
            start_time=0.0,
            end_time=15.0,
            match_minute_start=0.0,
            match_minute_end=0.0,
            events=(event,),
        )
        with pytest.raises(AttributeError):
            tw.period = 2  # type: ignore[misc]

    def test_slotted(self) -> None:
        """TimeWindow uses __slots__ for memory efficiency."""
        assert hasattr(TimeWindow, "__slots__")
