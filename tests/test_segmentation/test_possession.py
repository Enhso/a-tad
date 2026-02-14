"""Tests for possession-based event sequencing.

Covers single-team possession, turnover splitting, shot termination,
dead-ball termination, period-boundary isolation, minimum-event
filtering, set-piece detection, outcome label correctness, and
empty-input error handling.
"""

from __future__ import annotations

import pytest

from tactical.adapters.schemas import NormalizedEvent
from tactical.config import PossessionConfig
from tactical.exceptions import SegmentationError
from tactical.segmentation.possession import (
    PossessionSequence,
    create_possession_sequences,
)

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
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def default_config() -> PossessionConfig:
    """Default possession config: min 3 events."""
    return PossessionConfig()


@pytest.fixture()
def simple_five_events() -> list[NormalizedEvent]:
    """Five consecutive passes by team_a in period 1."""
    return [
        _evt(f"s_{i}", timestamp=float(i * 2), match_minute=float(i * 2) / 60.0)
        for i in range(5)
    ]


@pytest.fixture()
def turnover_events() -> list[NormalizedEvent]:
    """Four events by team_a, then three by team_b via interception."""
    team_a = [
        _evt(f"ta_{i}", timestamp=float(i * 2), match_minute=float(i * 2) / 60.0)
        for i in range(4)
    ]
    team_b = [
        _evt(
            f"tb_{i}",
            team_id="team_b",
            player_id="player_11",
            timestamp=float(8 + i * 2),
            match_minute=float(8 + i * 2) / 60.0,
            event_type="interception" if i == 0 else "pass",
            team_is_home=False,
        )
        for i in range(3)
    ]
    return team_a + team_b


@pytest.fixture()
def shot_events() -> list[NormalizedEvent]:
    """Three passes then a shot by team_a, followed by two team_b events."""
    team_a = [
        _evt(f"sh_{i}", timestamp=float(i * 2), match_minute=float(i * 2) / 60.0)
        for i in range(3)
    ]
    shot = _evt(
        "sh_3",
        timestamp=6.0,
        match_minute=0.1,
        event_type="shot",
        event_outcome="saved",
    )
    team_b_after = [
        _evt(
            f"sb_{i}",
            team_id="team_b",
            player_id="player_11",
            timestamp=float(8 + i * 2),
            match_minute=float(8 + i * 2) / 60.0,
            event_type="ball_recovery" if i == 0 else "pass",
            team_is_home=False,
        )
        for i in range(3)
    ]
    return team_a + [shot] + team_b_after


@pytest.fixture()
def dead_ball_events() -> list[NormalizedEvent]:
    """Three passes by team_a, then a throw-in by team_b (dead ball)."""
    team_a = [
        _evt(f"db_{i}", timestamp=float(i * 2), match_minute=float(i * 2) / 60.0)
        for i in range(3)
    ]
    throw_in_and_after = [
        _evt(
            f"ti_{i}",
            team_id="team_b",
            player_id="player_11",
            timestamp=float(6 + i * 2),
            match_minute=float(6 + i * 2) / 60.0,
            event_type="set_piece_throw_in" if i == 0 else "pass",
            team_is_home=False,
        )
        for i in range(3)
    ]
    return team_a + throw_in_and_after


@pytest.fixture()
def two_period_events() -> list[NormalizedEvent]:
    """Four events in period 1, four in period 2, all team_a."""
    p1 = [
        _evt(
            f"p1_{i}",
            period=1,
            timestamp=float(i * 3),
            match_minute=float(i * 3) / 60.0,
        )
        for i in range(4)
    ]
    p2 = [
        _evt(
            f"p2_{i}",
            period=2,
            timestamp=float(i * 3),
            match_minute=45.0 + float(i * 3) / 60.0,
        )
        for i in range(4)
    ]
    return p1 + p2


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestSimplePossession:
    """A single team's uninterrupted passage produces one sequence."""

    def test_simple_possession(
        self,
        simple_five_events: list[NormalizedEvent],
        default_config: PossessionConfig,
    ) -> None:
        """Five team_a passes yield exactly one possession sequence."""
        seqs = create_possession_sequences(simple_five_events, default_config)

        assert len(seqs) == 1
        assert seqs[0].team_id == "team_a"
        assert len(seqs[0].events) == 5
        assert seqs[0].outcome == "end_of_period"


class TestTurnoverSplits:
    """A team change mid-stream splits into two sequences."""

    def test_turnover_splits(
        self,
        turnover_events: list[NormalizedEvent],
        default_config: PossessionConfig,
    ) -> None:
        """Team switch from team_a to team_b produces two sequences."""
        seqs = create_possession_sequences(turnover_events, default_config)

        assert len(seqs) == 2
        assert seqs[0].team_id == "team_a"
        assert seqs[0].outcome == "turnover"
        assert seqs[1].team_id == "team_b"
        assert seqs[1].outcome == "end_of_period"


class TestShotEndsPossession:
    """A shot terminates the current possession."""

    def test_shot_ends_possession(
        self,
        shot_events: list[NormalizedEvent],
        default_config: PossessionConfig,
    ) -> None:
        """Possession ending with a shot has outcome 'shot'."""
        seqs = create_possession_sequences(shot_events, default_config)

        shot_seq = seqs[0]
        assert shot_seq.outcome == "shot"
        assert shot_seq.events[-1].event_type == "shot"
        assert shot_seq.team_id == "team_a"
        assert len(shot_seq.events) == 4

        assert seqs[1].team_id == "team_b"
        assert seqs[1].outcome == "end_of_period"


class TestDeadBallEndsPossession:
    """A dead-ball restart by the opposing team terminates possession."""

    def test_dead_ball_ends_possession(
        self,
        dead_ball_events: list[NormalizedEvent],
        default_config: PossessionConfig,
    ) -> None:
        """Throw-in by team_b ends team_a possession with 'dead_ball'."""
        seqs = create_possession_sequences(dead_ball_events, default_config)

        assert seqs[0].team_id == "team_a"
        assert seqs[0].outcome == "dead_ball"

        assert seqs[1].team_id == "team_b"
        assert seqs[1].events[0].event_type == "set_piece_throw_in"


class TestPeriodBoundary:
    """Events in different periods produce separate sequences."""

    def test_period_boundary(
        self,
        two_period_events: list[NormalizedEvent],
        default_config: PossessionConfig,
    ) -> None:
        """Same team across period boundary yields two sequences."""
        seqs = create_possession_sequences(two_period_events, default_config)

        assert len(seqs) == 2
        assert seqs[0].period == 1
        assert seqs[0].outcome == "end_of_period"
        assert seqs[1].period == 2
        assert seqs[1].outcome == "end_of_period"

        p1_periods = {e.period for e in seqs[0].events}
        p2_periods = {e.period for e in seqs[1].events}
        assert p1_periods == {1}
        assert p2_periods == {2}


class TestMinEventsFilter:
    """Sequences with too few events are dropped."""

    def test_min_events_filter(self, default_config: PossessionConfig) -> None:
        """Two-event sequence dropped when min_events=3."""
        events = [
            _evt("mf_0", timestamp=0.0, match_minute=0.0),
            _evt("mf_1", timestamp=1.0, match_minute=1.0 / 60.0),
        ]
        seqs = create_possession_sequences(events, default_config)
        assert len(seqs) == 0

    def test_min_events_allows_exact_threshold(self) -> None:
        """Exactly min_events events are retained."""
        events = [
            _evt(f"ex_{i}", timestamp=float(i), match_minute=float(i) / 60.0)
            for i in range(3)
        ]
        config = PossessionConfig(min_events=3)
        seqs = create_possession_sequences(events, config)
        assert len(seqs) == 1
        assert len(seqs[0].events) == 3


class TestSetPieceDetection:
    """Sequences starting with a set piece are flagged."""

    def test_set_piece_detection(self) -> None:
        """Sequence starting with a corner is marked as set piece."""
        events: list[NormalizedEvent] = [
            _evt(
                "sp_0",
                timestamp=0.0,
                match_minute=0.0,
                event_type="set_piece_corner",
            ),
            _evt("sp_1", timestamp=2.0, match_minute=2.0 / 60.0),
            _evt("sp_2", timestamp=4.0, match_minute=4.0 / 60.0),
        ]
        config = PossessionConfig(min_events=3)
        seqs = create_possession_sequences(events, config)

        assert len(seqs) == 1
        assert seqs[0].is_set_piece is True
        assert seqs[0].set_piece_type == "set_piece_corner"

    def test_non_set_piece_start(self, default_config: PossessionConfig) -> None:
        """Sequence starting with a pass is not a set piece."""
        events = [
            _evt(f"ns_{i}", timestamp=float(i), match_minute=float(i) / 60.0)
            for i in range(4)
        ]
        seqs = create_possession_sequences(events, default_config)

        assert seqs[0].is_set_piece is False
        assert seqs[0].set_piece_type is None


class TestOutcomeLabels:
    """Verify correct outcome assigned for each termination type."""

    def test_outcome_labels(self) -> None:
        """Multiple termination types in one event stream."""
        events: list[NormalizedEvent] = [
            # Sequence 1: team_a pass x3, then shot -> "shot"
            _evt("o_0", timestamp=0.0, match_minute=0.0),
            _evt("o_1", timestamp=1.0, match_minute=1.0 / 60.0),
            _evt("o_2", timestamp=2.0, match_minute=2.0 / 60.0),
            _evt(
                "o_3",
                timestamp=3.0,
                match_minute=3.0 / 60.0,
                event_type="shot",
                event_outcome="saved",
            ),
            # Sequence 2: team_b x3, then team_a takes over -> "turnover"
            _evt(
                "o_4",
                team_id="team_b",
                player_id="player_11",
                timestamp=5.0,
                match_minute=5.0 / 60.0,
                event_type="ball_recovery",
                team_is_home=False,
            ),
            _evt(
                "o_5",
                team_id="team_b",
                player_id="player_11",
                timestamp=6.0,
                match_minute=6.0 / 60.0,
                team_is_home=False,
            ),
            _evt(
                "o_6",
                team_id="team_b",
                player_id="player_11",
                timestamp=7.0,
                match_minute=7.0 / 60.0,
                team_is_home=False,
            ),
            # Sequence 3: team_a x3, reaches end -> "end_of_period"
            _evt("o_7", timestamp=9.0, match_minute=9.0 / 60.0),
            _evt("o_8", timestamp=10.0, match_minute=10.0 / 60.0),
            _evt("o_9", timestamp=11.0, match_minute=11.0 / 60.0),
        ]
        config = PossessionConfig(min_events=3)
        seqs = create_possession_sequences(events, config)

        assert len(seqs) == 3
        assert seqs[0].outcome == "shot"
        assert seqs[1].outcome == "turnover"
        assert seqs[2].outcome == "end_of_period"


class TestEmptyEventsRaises:
    """An empty event list must raise SegmentationError."""

    def test_empty_events_raises(self, default_config: PossessionConfig) -> None:
        """Passing an empty list raises SegmentationError."""
        with pytest.raises(SegmentationError):
            create_possession_sequences([], default_config)


class TestSkippedEvents:
    """Substitution and goal_keeper events do not break possession."""

    def test_substitution_does_not_break(self) -> None:
        """A substitution mid-sequence keeps possession intact."""
        events: list[NormalizedEvent] = [
            _evt("sk_0", timestamp=0.0, match_minute=0.0),
            _evt("sk_1", timestamp=1.0, match_minute=1.0 / 60.0),
            _evt(
                "sk_sub",
                timestamp=2.0,
                match_minute=2.0 / 60.0,
                event_type="substitution",
            ),
            _evt("sk_2", timestamp=3.0, match_minute=3.0 / 60.0),
        ]
        config = PossessionConfig(min_events=3)
        seqs = create_possession_sequences(events, config)

        assert len(seqs) == 1
        # Substitution excluded from the sequence events
        assert all(e.event_type != "substitution" for e in seqs[0].events)
        assert len(seqs[0].events) == 3


class TestPossessionSequenceDataclass:
    """Verify PossessionSequence is frozen and slotted."""

    def test_frozen(self) -> None:
        """PossessionSequence instances are immutable."""
        event = _evt("f_0", timestamp=0.0, match_minute=0.0)
        ps = PossessionSequence(
            match_id="m",
            team_id="t",
            period=1,
            start_time=0.0,
            end_time=1.0,
            match_minute_start=0.0,
            match_minute_end=1.0,
            events=(event,),
            outcome="end_of_period",
            is_set_piece=False,
            set_piece_type=None,
        )
        with pytest.raises(AttributeError):
            ps.outcome = "shot"  # type: ignore[misc]

    def test_slotted(self) -> None:
        """PossessionSequence uses __slots__ for memory efficiency."""
        assert hasattr(PossessionSequence, "__slots__")
