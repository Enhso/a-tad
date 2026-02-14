"""Integration tests for time-based and possession-based segmentation.

Verifies that both segmentation schemes produce correct, independent
results when applied to the same realistic event stream.
"""

from __future__ import annotations

import pytest

from tactical.adapters.schemas import NormalizedEvent
from tactical.config import PossessionConfig, WindowConfig
from tactical.segmentation.possession import (
    PossessionSequence,
    create_possession_sequences,
)
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
    team_is_home: bool = True,
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
        team_is_home=team_is_home,
    )


# ------------------------------------------------------------------
# Fixture: realistic 30-event stream
# ------------------------------------------------------------------

_TEAM_A = "team_a"
_TEAM_B = "team_b"


@pytest.fixture()
def match_events() -> list[NormalizedEvent]:
    """30 events spanning ~120 s in period 1.

    Layout
    ------
    Seq 1 (team_a, 10 events, ts 2-48):
        8 passes + 1 carry + 1 shot  ->  outcome "shot"
    Seq 2 (team_b, 10 events, ts 52-84):
        interception + mixed          ->  outcome "turnover"
    Seq 3 (team_a, 10 events, ts 88-118):
        ball_recovery + mixed         ->  outcome "end_of_period"
    """
    events: list[NormalizedEvent] = []

    # -- Sequence 1: team_a build-up ending with a shot ----------------
    seq1_specs: list[tuple[str, float, str]] = [
        ("a01", 2.0, "pass"),
        ("a02", 6.0, "pass"),
        ("a03", 10.0, "pass"),
        ("a04", 15.0, "carry"),
        ("a05", 20.0, "pass"),
        ("a06", 25.0, "pass"),
        ("a07", 30.0, "pass"),
        ("a08", 36.0, "ball_receipt"),
        ("a09", 42.0, "pass"),
        ("a10", 48.0, "shot"),
    ]
    for eid, ts, etype in seq1_specs:
        events.append(
            _evt(
                eid,
                team_id=_TEAM_A,
                timestamp=ts,
                match_minute=ts / 60.0,
                event_type=etype,
                event_outcome="saved" if etype == "shot" else "complete",
                location=(30.0 + ts * 0.4, 45.0),
            )
        )

    # -- Sequence 2: team_b possession after turnover ------------------
    seq2_specs: list[tuple[str, float, str]] = [
        ("b01", 52.0, "interception"),
        ("b02", 55.0, "ball_receipt"),
        ("b03", 58.0, "pass"),
        ("b04", 62.0, "carry"),
        ("b05", 66.0, "pass"),
        ("b06", 70.0, "pass"),
        ("b07", 73.0, "pass"),
        ("b08", 77.0, "carry"),
        ("b09", 80.0, "pass"),
        ("b10", 84.0, "pass"),
    ]
    for eid, ts, etype in seq2_specs:
        events.append(
            _evt(
                eid,
                team_id=_TEAM_B,
                player_id="player_11",
                timestamp=ts,
                match_minute=ts / 60.0,
                event_type=etype,
                location=(70.0 - ts * 0.3, 50.0),
                team_is_home=False,
            )
        )

    # -- Sequence 3: team_a wins possession back -----------------------
    seq3_specs: list[tuple[str, float, str]] = [
        ("a11", 88.0, "ball_recovery"),
        ("a12", 92.0, "ball_receipt"),
        ("a13", 96.0, "pass"),
        ("a14", 100.0, "carry"),
        ("a15", 103.0, "pass"),
        ("a16", 106.0, "pass"),
        ("a17", 109.0, "ball_receipt"),
        ("a18", 112.0, "pass"),
        ("a19", 115.0, "carry"),
        ("a20", 118.0, "pass"),
    ]
    for eid, ts, etype in seq3_specs:
        events.append(
            _evt(
                eid,
                team_id=_TEAM_A,
                timestamp=ts,
                match_minute=ts / 60.0,
                event_type=etype,
                location=(25.0 + ts * 0.3, 40.0),
            )
        )

    return events


# ------------------------------------------------------------------
# Integration test
# ------------------------------------------------------------------


class TestBothSegmentationsOnSameData:
    """Run time-window and possession segmentation on the same stream."""

    def test_time_windows_reasonable_count(
        self,
        match_events: list[NormalizedEvent],
    ) -> None:
        """Default config (15 s / 5 s overlap) produces ~11 windows."""
        config = WindowConfig()
        windows = create_time_windows(match_events, config)

        # step=10, min_ts=2, max_ts=118 -> starts 2,12,...,112
        # One or two may be dropped for <3 events in the gap regions
        assert 9 <= len(windows) <= 13
        assert all(isinstance(w, TimeWindow) for w in windows)

    def test_no_empty_windows(
        self,
        match_events: list[NormalizedEvent],
    ) -> None:
        """Every retained window has at least min_events events."""
        config = WindowConfig()
        windows = create_time_windows(match_events, config)

        for w in windows:
            assert len(w.events) >= config.min_events

    def test_possession_three_sequences(
        self,
        match_events: list[NormalizedEvent],
    ) -> None:
        """Possession segmentation yields 3 sequences."""
        config = PossessionConfig()
        seqs = create_possession_sequences(match_events, config)

        assert len(seqs) == 3
        assert all(isinstance(s, PossessionSequence) for s in seqs)

    def test_possession_outcomes(
        self,
        match_events: list[NormalizedEvent],
    ) -> None:
        """Outcomes are shot, turnover, end_of_period in order."""
        config = PossessionConfig()
        seqs = create_possession_sequences(match_events, config)

        assert seqs[0].team_id == _TEAM_A
        assert seqs[0].outcome == "shot"

        assert seqs[1].team_id == _TEAM_B
        assert seqs[1].outcome == "turnover"

        assert seqs[2].team_id == _TEAM_A
        assert seqs[2].outcome == "end_of_period"

    def test_every_event_in_at_least_one_window(
        self,
        match_events: list[NormalizedEvent],
    ) -> None:
        """Sliding windows with overlap cover every input event."""
        config = WindowConfig()
        windows = create_time_windows(match_events, config)

        windowed_ids: set[str] = set()
        for w in windows:
            windowed_ids.update(e.event_id for e in w.events)

        input_ids = {e.event_id for e in match_events}
        assert input_ids == windowed_ids

    def test_possessions_partition_input(
        self,
        match_events: list[NormalizedEvent],
    ) -> None:
        """Possession events partition the input (no skips here)."""
        config = PossessionConfig()
        seqs = create_possession_sequences(match_events, config)

        possession_ids: list[str] = []
        for s in seqs:
            possession_ids.extend(e.event_id for e in s.events)

        input_ids = [
            e.event_id
            for e in sorted(match_events, key=lambda e: (e.period, e.timestamp))
        ]

        # Every input event appears exactly once across all possessions
        assert sorted(possession_ids) == sorted(input_ids)
        # No duplicates
        assert len(possession_ids) == len(set(possession_ids))

    def test_schemes_independent(
        self,
        match_events: list[NormalizedEvent],
    ) -> None:
        """Running both schemes on the same list causes no interference."""
        w_cfg = WindowConfig()
        p_cfg = PossessionConfig()

        windows_first = create_time_windows(match_events, w_cfg)
        seqs = create_possession_sequences(match_events, p_cfg)
        windows_second = create_time_windows(match_events, w_cfg)

        # Window results identical before and after possession run
        assert len(windows_first) == len(windows_second)
        for w1, w2 in zip(windows_first, windows_second, strict=True):
            assert w1.start_time == pytest.approx(w2.start_time)
            assert w1.end_time == pytest.approx(w2.end_time)
            assert len(w1.events) == len(w2.events)
            assert tuple(e.event_id for e in w1.events) == tuple(
                e.event_id for e in w2.events
            )

        # Possession results still valid
        assert len(seqs) == 3
        assert [s.outcome for s in seqs] == [
            "shot",
            "turnover",
            "end_of_period",
        ]

    def test_input_list_unmodified(
        self,
        match_events: list[NormalizedEvent],
    ) -> None:
        """Neither segmentation mutates the original event list."""
        original_ids = [e.event_id for e in match_events]
        original_len = len(match_events)

        create_time_windows(match_events, WindowConfig())
        create_possession_sequences(match_events, PossessionConfig())

        assert len(match_events) == original_len
        assert [e.event_id for e in match_events] == original_ids
