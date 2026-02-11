"""Tests for the internal data schemas.

Validates creation, field accessibility, and immutability of all frozen
dataclasses defined in :mod:`tactical.adapters.schemas`.
"""

from __future__ import annotations

import dataclasses

import pytest

from tactical.adapters.schemas import (
    FreezeFramePlayer,
    MatchContext,
    MatchInfo,
    NormalizedEvent,
)


@pytest.fixture()
def freeze_frame_player() -> FreezeFramePlayer:
    """Return a valid FreezeFramePlayer instance."""
    return FreezeFramePlayer(
        player_id="p1",
        teammate=True,
        location=(50.0, 50.0),
        position="Center Back",
    )


@pytest.fixture()
def normalized_event(
    freeze_frame_player: FreezeFramePlayer,
) -> NormalizedEvent:
    """Return a valid NormalizedEvent instance with a freeze frame."""
    return NormalizedEvent(
        event_id="evt-001",
        match_id="match-001",
        team_id="team-a",
        player_id="player-10",
        period=1,
        timestamp=123.4,
        match_minute=2.06,
        location=(45.0, 55.0),
        end_location=(80.0, 30.0),
        event_type="pass",
        event_outcome="complete",
        under_pressure=False,
        body_part="right_foot",
        freeze_frame=(freeze_frame_player,),
        score_home=0,
        score_away=0,
        team_is_home=True,
    )


@pytest.fixture()
def match_info() -> MatchInfo:
    """Return a valid MatchInfo instance."""
    return MatchInfo(
        match_id="match-001",
        competition="Premier League",
        season="2023/2024",
        home_team_id="team-a",
        home_team_name="Arsenal",
        away_team_id="team-b",
        away_team_name="Chelsea",
        home_score=2,
        away_score=1,
        match_date="2024-01-15",
    )


@pytest.fixture()
def match_context() -> MatchContext:
    """Return a valid MatchContext instance."""
    return MatchContext(
        match_id="match-001",
        team_id="team-a",
        opponent_id="team-b",
        team_is_home=True,
        has_360=False,
    )


class TestNormalizedEventCreation:
    """Test NormalizedEvent field accessibility."""

    def test_all_fields_accessible(self, normalized_event: NormalizedEvent) -> None:
        """All fields on a valid NormalizedEvent are accessible."""
        assert normalized_event.event_id == "evt-001"
        assert normalized_event.match_id == "match-001"
        assert normalized_event.team_id == "team-a"
        assert normalized_event.player_id == "player-10"
        assert normalized_event.period == 1
        assert normalized_event.timestamp == 123.4
        assert normalized_event.match_minute == 2.06
        assert normalized_event.location == (45.0, 55.0)
        assert normalized_event.end_location == (80.0, 30.0)
        assert normalized_event.event_type == "pass"
        assert normalized_event.event_outcome == "complete"
        assert normalized_event.under_pressure is False
        assert normalized_event.body_part == "right_foot"
        assert normalized_event.freeze_frame is not None
        assert len(normalized_event.freeze_frame) == 1
        assert normalized_event.score_home == 0
        assert normalized_event.score_away == 0
        assert normalized_event.team_is_home is True

    def test_none_optional_fields(self) -> None:
        """NormalizedEvent accepts None for optional fields."""
        event = NormalizedEvent(
            event_id="evt-002",
            match_id="match-001",
            team_id="team-a",
            player_id="player-10",
            period=2,
            timestamp=0.0,
            match_minute=45.0,
            location=None,
            end_location=None,
            event_type="pressure",
            event_outcome="success",
            under_pressure=True,
            body_part="no_touch",
            freeze_frame=None,
            score_home=1,
            score_away=0,
            team_is_home=False,
        )
        assert event.location is None
        assert event.end_location is None
        assert event.freeze_frame is None


class TestNormalizedEventImmutable:
    """Test NormalizedEvent frozen immutability."""

    def test_cannot_modify_field(self, normalized_event: NormalizedEvent) -> None:
        """Assigning to a field raises FrozenInstanceError."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            normalized_event.event_id = "changed"  # type: ignore[misc]

    def test_cannot_modify_score(self, normalized_event: NormalizedEvent) -> None:
        """Assigning to an int field raises FrozenInstanceError."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            normalized_event.score_home = 99  # type: ignore[misc]


class TestFreezeFramePlayerCreation:
    """Test FreezeFramePlayer creation and immutability."""

    def test_all_fields_accessible(
        self, freeze_frame_player: FreezeFramePlayer
    ) -> None:
        """All fields on a valid FreezeFramePlayer are accessible."""
        assert freeze_frame_player.player_id == "p1"
        assert freeze_frame_player.teammate is True
        assert freeze_frame_player.location == (50.0, 50.0)
        assert freeze_frame_player.position == "Center Back"

    def test_immutable(self, freeze_frame_player: FreezeFramePlayer) -> None:
        """Assigning to a field raises FrozenInstanceError."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            freeze_frame_player.teammate = False  # type: ignore[misc]


class TestMatchInfoCreation:
    """Test MatchInfo creation and immutability."""

    def test_all_fields_accessible(self, match_info: MatchInfo) -> None:
        """All fields on a valid MatchInfo are accessible."""
        assert match_info.match_id == "match-001"
        assert match_info.competition == "Premier League"
        assert match_info.season == "2023/2024"
        assert match_info.home_team_id == "team-a"
        assert match_info.home_team_name == "Arsenal"
        assert match_info.away_team_id == "team-b"
        assert match_info.away_team_name == "Chelsea"
        assert match_info.home_score == 2
        assert match_info.away_score == 1
        assert match_info.match_date == "2024-01-15"

    def test_immutable(self, match_info: MatchInfo) -> None:
        """Assigning to a field raises FrozenInstanceError."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            match_info.home_score = 5  # type: ignore[misc]


class TestMatchContextCreation:
    """Test MatchContext creation and immutability."""

    def test_all_fields_accessible(self, match_context: MatchContext) -> None:
        """All fields on a valid MatchContext are accessible."""
        assert match_context.match_id == "match-001"
        assert match_context.team_id == "team-a"
        assert match_context.opponent_id == "team-b"
        assert match_context.team_is_home is True
        assert match_context.has_360 is False

    def test_immutable(self, match_context: MatchContext) -> None:
        """Assigning to a field raises FrozenInstanceError."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            match_context.has_360 = True  # type: ignore[misc]
