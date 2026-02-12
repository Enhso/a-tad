"""Tests for the StatsBomb data adapter.

Validates coordinate normalization, event type mapping, unknown event
type handling, cache-first loading, and match listing for
:class:`tactical.adapters.statsbomb.StatsBombAdapter`.

All tests mock ``statsbombpy.sb`` to avoid real API calls.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tactical.adapters.schemas import MatchInfo, NormalizedEvent
from tactical.adapters.statsbomb import (
    STATSBOMB_EVENT_TYPE_MAP,
    StatsBombAdapter,
    _parse_timestamp,
)

if TYPE_CHECKING:
    from pathlib import Path


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

_FAKE_EVENTS_DATA: dict[str, list[Any]] = {
    "id": [
        "start-home",
        "start-away",
        "half-start-1",
        "half-start-2",
        "evt-pass-1",
        "evt-shot-1",
        "evt-carry-1",
        "evt-pressure-1",
        "evt-unknown-1",
    ],
    "index": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "match_id": [100] * 9,
    "type": [
        "Starting XI",
        "Starting XI",
        "Half Start",
        "Half Start",
        "Pass",
        "Shot",
        "Carry",
        "Pressure",
        "InventedType",
    ],
    "period": [1] * 9,
    "timestamp": [
        "00:00:00.000",
        "00:00:00.000",
        "00:00:00.000",
        "00:00:00.000",
        "00:00:05.500",
        "00:01:30.000",
        "00:02:00.000",
        "00:02:10.000",
        "00:02:20.000",
    ],
    "minute": [0, 0, 0, 0, 0, 1, 2, 2, 2],
    "second": [0, 0, 0, 0, 5, 30, 0, 10, 20],
    "team": ["HomeFC"] * 5 + ["AwayFC", "HomeFC", "AwayFC", "HomeFC"],
    "team_id": [1, 2, 1, 2, 1, 2, 1, 2, 1],
    "player": [
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        "Alice",
        "Bob",
        "Alice",
        "Carol",
        "Dave",
    ],
    "player_id": [
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        10.0,
        20.0,
        10.0,
        30.0,
        40.0,
    ],
    "location": [
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        [60.0, 40.0],
        [108.0, 28.0],
        [30.0, 60.0],
        [90.0, 10.0],
        [0.0, 0.0],
    ],
    "under_pressure": [
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        True,
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
    ],
    "pass_outcome": [
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
    ],
    "pass_body_part": [
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        "Right Foot",
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
    ],
    "pass_end_location": [
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        [80.0, 50.0],
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
    ],
    "pass_type": [
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
    ],
    "shot_outcome": [
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        "Goal",
        float("nan"),
        float("nan"),
        float("nan"),
    ],
    "shot_body_part": [
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        "Left Foot",
        float("nan"),
        float("nan"),
        float("nan"),
    ],
    "shot_end_location": [
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        [120.0, 40.0, 1.0],
        float("nan"),
        float("nan"),
        float("nan"),
    ],
    "shot_type": [
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        "Open Play",
        float("nan"),
        float("nan"),
        float("nan"),
    ],
    "carry_end_location": [
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        [35.0, 55.0],
        float("nan"),
        float("nan"),
    ],
    "duel_type": [float("nan")] * 9,
    "duel_outcome": [float("nan")] * 9,
    "interception_outcome": [float("nan")] * 9,
    "ball_receipt_outcome": [float("nan")] * 9,
    "dribble_outcome": [float("nan")] * 9,
    "goalkeeper_outcome": [float("nan")] * 9,
    "goalkeeper_body_part": [float("nan")] * 9,
    "goalkeeper_end_location": [float("nan")] * 9,
    "clearance_body_part": [float("nan")] * 9,
    "position": [float("nan")] * 9,
    "play_pattern": ["Regular Play"] * 9,
    "possession": list(range(1, 10)),
    "possession_team": ["HomeFC"] * 9,
    "possession_team_id": [1] * 9,
    "duration": [0.0] * 9,
    "related_events": [float("nan")] * 9,
    "tactics": [float("nan")] * 9,
    "substitution_outcome": [float("nan")] * 9,
    "substitution_outcome_id": [float("nan")] * 9,
    "substitution_replacement": [float("nan")] * 9,
    "substitution_replacement_id": [float("nan")] * 9,
}


@pytest.fixture()
def fake_events_df() -> pd.DataFrame:
    """Return a small fake StatsBomb events DataFrame."""
    return pd.DataFrame(_FAKE_EVENTS_DATA)


@pytest.fixture()
def adapter(tmp_path: Path) -> StatsBombAdapter:
    """Return a StatsBombAdapter backed by a temporary cache directory."""
    return StatsBombAdapter(cache_dir=tmp_path / "cache")


# ------------------------------------------------------------------
# Coordinate normalization
# ------------------------------------------------------------------


class TestCoordinateNormalization:
    """Verify StatsBomb 120x80 -> 0-100 coordinate conversion."""

    def test_origin(self) -> None:
        """(0, 0) maps to (0.0, 100.0) -- top-left corner."""
        assert StatsBombAdapter._normalize_coordinates(0.0, 0.0) == (0.0, 100.0)

    def test_far_corner(self) -> None:
        """(120, 80) maps to (100.0, 0.0) -- bottom-right corner."""
        assert StatsBombAdapter._normalize_coordinates(120.0, 80.0) == (100.0, 0.0)

    def test_center(self) -> None:
        """(60, 40) maps to (50.0, 50.0) -- pitch center."""
        assert StatsBombAdapter._normalize_coordinates(60.0, 40.0) == (50.0, 50.0)


# ------------------------------------------------------------------
# Event type mapping
# ------------------------------------------------------------------


class TestEventTypeMapping:
    """Verify known StatsBomb types map to the controlled vocabulary."""

    @pytest.mark.parametrize(
        ("sb_type", "expected"),
        [
            ("Pass", "pass"),
            ("Ball Receipt*", "ball_receipt"),
            ("Carry", "carry"),
            ("Dribble", "carry"),
            ("Shot", "shot"),
            ("Pressure", "pressure"),
            ("Foul Committed", "foul_committed"),
            ("Foul Won", "foul_won"),
            ("Interception", "interception"),
            ("Block", "block"),
            ("Clearance", "clearance"),
            ("Ball Recovery", "ball_recovery"),
            ("Dispossessed", "dispossessed"),
            ("Miscontrol", "miscontrol"),
            ("Goal Keeper", "goal_keeper"),
            ("Substitution", "substitution"),
        ],
    )
    def test_basic_mapping(self, sb_type: str, expected: str) -> None:
        """Each basic StatsBomb type maps to the correct controlled type."""
        assert STATSBOMB_EVENT_TYPE_MAP[sb_type] == expected

    def test_pass_corner_maps_to_set_piece(self) -> None:
        """A Pass with pass_type 'Corner' resolves to set_piece_corner."""
        row = pd.Series({"type": "Pass", "pass_type": "Corner"})
        assert StatsBombAdapter._resolve_event_type(row) == "set_piece_corner"

    def test_pass_free_kick_maps_to_set_piece(self) -> None:
        """A Pass with pass_type 'Free Kick' resolves to set_piece_free_kick."""
        row = pd.Series({"type": "Pass", "pass_type": "Free Kick"})
        assert StatsBombAdapter._resolve_event_type(row) == "set_piece_free_kick"

    def test_pass_throw_in_maps_to_set_piece(self) -> None:
        """A Pass with pass_type 'Throw-in' resolves to set_piece_throw_in."""
        row = pd.Series({"type": "Pass", "pass_type": "Throw-in"})
        assert StatsBombAdapter._resolve_event_type(row) == "set_piece_throw_in"

    def test_pass_goal_kick_maps_to_set_piece(self) -> None:
        """A Pass with pass_type 'Goal Kick' resolves to set_piece_goal_kick."""
        row = pd.Series({"type": "Pass", "pass_type": "Goal Kick"})
        assert StatsBombAdapter._resolve_event_type(row) == "set_piece_goal_kick"

    def test_pass_kick_off_maps_to_set_piece(self) -> None:
        """A Pass with pass_type 'Kick Off' resolves to set_piece_kick_off."""
        row = pd.Series({"type": "Pass", "pass_type": "Kick Off"})
        assert StatsBombAdapter._resolve_event_type(row) == "set_piece_kick_off"

    def test_shot_penalty_maps_to_set_piece(self) -> None:
        """A Shot with shot_type 'Penalty' resolves to set_piece_penalty."""
        row = pd.Series({"type": "Shot", "shot_type": "Penalty"})
        assert StatsBombAdapter._resolve_event_type(row) == "set_piece_penalty"

    def test_duel_tackle_maps_to_tackle(self) -> None:
        """A Duel with duel_type 'Tackle' resolves to tackle."""
        row = pd.Series({"type": "Duel", "duel_type": "Tackle"})
        assert StatsBombAdapter._resolve_event_type(row) == "tackle"

    def test_duel_aerial_maps_to_duel(self) -> None:
        """A Duel with duel_type 'Aerial Lost' resolves to duel."""
        row = pd.Series({"type": "Duel", "duel_type": "Aerial Lost"})
        assert StatsBombAdapter._resolve_event_type(row) == "duel"

    def test_regular_pass_maps_to_pass(self) -> None:
        """A Pass with no special pass_type resolves to pass."""
        row = pd.Series({"type": "Pass", "pass_type": float("nan")})
        assert StatsBombAdapter._resolve_event_type(row) == "pass"

    def test_unmapped_type_returns_none(self) -> None:
        """An unknown event type returns None."""
        row = pd.Series({"type": "Tactical Shift"})
        assert StatsBombAdapter._resolve_event_type(row) is None


# ------------------------------------------------------------------
# Unknown event type skipping
# ------------------------------------------------------------------


class TestUnknownEventTypeSkipped:
    """Verify unknown event types are logged and dropped."""

    def test_unknown_type_logged_and_skipped(
        self,
        adapter: StatsBombAdapter,
        fake_events_df: pd.DataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Unknown event types produce a warning and are excluded."""
        with caplog.at_level(logging.WARNING):
            result = adapter._normalize_events(fake_events_df, "100")

        event_types = [e.event_type for e in result]
        assert "InventedType" not in event_types

        assert any("InventedType" in rec.message for rec in caplog.records)

    def test_known_skippable_types_not_warned(
        self,
        adapter: StatsBombAdapter,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Starting XI and Half Start are silently skipped (no warning)."""
        data = {
            "id": ["s1", "s2", "h1", "h2"],
            "index": [1, 2, 3, 4],
            "match_id": [100] * 4,
            "type": ["Starting XI", "Starting XI", "Half Start", "Half Start"],
            "period": [1] * 4,
            "timestamp": ["00:00:00.000"] * 4,
            "minute": [0] * 4,
            "second": [0] * 4,
            "team": ["HomeFC", "AwayFC", "HomeFC", "AwayFC"],
            "team_id": [1, 2, 1, 2],
            "player": [float("nan")] * 4,
            "player_id": [float("nan")] * 4,
            "location": [float("nan")] * 4,
            "under_pressure": [float("nan")] * 4,
        }
        df = pd.DataFrame(data)
        with caplog.at_level(logging.WARNING):
            result = adapter._normalize_events(df, "100")

        assert result == []
        assert not any("Starting XI" in rec.message for rec in caplog.records)
        assert not any("Half Start" in rec.message for rec in caplog.records)


# ------------------------------------------------------------------
# Cache integration
# ------------------------------------------------------------------


class TestCacheUsage:
    """Verify load_match_events uses the cache before calling the API."""

    @patch("tactical.adapters.statsbomb.sb")
    def test_load_match_events_uses_cache(
        self,
        mock_sb: MagicMock,
        adapter: StatsBombAdapter,
        fake_events_df: pd.DataFrame,
    ) -> None:
        """On cache miss the API is called; on cache hit it is not."""
        mock_sb.events.return_value = fake_events_df

        # First call: cache miss -> API called
        result_1 = adapter.load_match_events("100")
        mock_sb.events.assert_called_once_with(match_id=100, flatten_attrs=True)
        assert len(result_1) > 0

        # Second call: cache hit -> API NOT called again
        mock_sb.events.reset_mock()
        result_2 = adapter.load_match_events("100")
        mock_sb.events.assert_not_called()
        assert len(result_2) == len(result_1)


# ------------------------------------------------------------------
# list_matches
# ------------------------------------------------------------------


class TestListMatches:
    """Verify list_matches returns sorted MatchInfo objects."""

    @patch("tactical.adapters.statsbomb.sb")
    def test_list_matches_returns_match_info(
        self,
        mock_sb: MagicMock,
        adapter: StatsBombAdapter,
    ) -> None:
        """list_matches returns MatchInfo objects sorted by date."""
        mock_sb.matches.return_value = {
            "match_2": {
                "match_id": 2,
                "match_date": "2024-02-01",
                "competition": {
                    "competition_id": 11,
                    "competition_name": "La Liga",
                },
                "season": {"season_id": 90, "season_name": "2023/2024"},
                "home_team": {
                    "home_team_id": 100,
                    "home_team_name": "TeamA",
                },
                "away_team": {
                    "away_team_id": 200,
                    "away_team_name": "TeamB",
                },
                "home_score": 3,
                "away_score": 1,
            },
            "match_1": {
                "match_id": 1,
                "match_date": "2024-01-15",
                "competition": {
                    "competition_id": 11,
                    "competition_name": "La Liga",
                },
                "season": {"season_id": 90, "season_name": "2023/2024"},
                "home_team": {
                    "home_team_id": 300,
                    "home_team_name": "TeamC",
                },
                "away_team": {
                    "away_team_id": 400,
                    "away_team_name": "TeamD",
                },
                "home_score": 0,
                "away_score": 0,
            },
        }

        result = adapter.list_matches(competition_id=11, season_id=90)

        mock_sb.matches.assert_called_once_with(
            competition_id=11, season_id=90, fmt="json"
        )

        assert len(result) == 2
        assert all(isinstance(m, MatchInfo) for m in result)

        # Sorted by match_date ascending
        assert result[0].match_date == "2024-01-15"
        assert result[1].match_date == "2024-02-01"

        assert result[0].match_id == "1"
        assert result[0].home_team_id == "300"
        assert result[0].away_team_name == "TeamD"

        assert result[1].match_id == "2"
        assert result[1].home_score == 3
        assert result[1].away_score == 1


# ------------------------------------------------------------------
# Normalized event field correctness
# ------------------------------------------------------------------


class TestNormalizeEventsFields:
    """Verify field-level correctness of normalized events."""

    def test_normalized_event_fields(
        self,
        adapter: StatsBombAdapter,
        fake_events_df: pd.DataFrame,
    ) -> None:
        """All normalized events have correct field values."""
        events = adapter._normalize_events(fake_events_df, "100")

        # Should have 4 events: pass, shot, carry, pressure
        assert len(events) == 4
        assert all(isinstance(e, NormalizedEvent) for e in events)

        # Events sorted by (period, timestamp)
        timestamps = [e.timestamp for e in events]
        assert timestamps == sorted(timestamps)

    def test_pass_event(
        self,
        adapter: StatsBombAdapter,
        fake_events_df: pd.DataFrame,
    ) -> None:
        """Pass event has correct type, outcome, body part, and location."""
        events = adapter._normalize_events(fake_events_df, "100")
        pass_evt = next(e for e in events if e.event_type == "pass")

        assert pass_evt.event_outcome == "complete"
        assert pass_evt.body_part == "right_foot"
        assert pass_evt.under_pressure is True
        assert pass_evt.team_is_home is True
        assert pass_evt.team_id == "1"
        assert pass_evt.player_id == "10"
        # location: [60, 40] -> (50.0, 50.0)
        assert pass_evt.location == (50.0, 50.0)
        # end_location: [80, 50] -> (66.67, 37.5)
        assert pass_evt.end_location is not None
        assert abs(pass_evt.end_location[0] - 80.0 / 120.0 * 100.0) < 0.01
        assert abs(pass_evt.end_location[1] - (80.0 - 50.0) / 80.0 * 100.0) < 0.01

    def test_shot_goal_increments_score(
        self,
        adapter: StatsBombAdapter,
        fake_events_df: pd.DataFrame,
    ) -> None:
        """A goal by the away team increments score_away."""
        events = adapter._normalize_events(fake_events_df, "100")
        shot_evt = next(e for e in events if e.event_type == "shot")

        assert shot_evt.team_is_home is False
        # Goal scored by away team -> score_away = 1
        assert shot_evt.score_away == 1
        assert shot_evt.score_home == 0

    def test_carry_event(
        self,
        adapter: StatsBombAdapter,
        fake_events_df: pd.DataFrame,
    ) -> None:
        """Carry event has correct end location from carry_end_location."""
        events = adapter._normalize_events(fake_events_df, "100")
        carry_evt = next(e for e in events if e.event_type == "carry")

        assert carry_evt.end_location is not None
        assert abs(carry_evt.end_location[0] - 35.0 / 120.0 * 100.0) < 0.01
        assert abs(carry_evt.end_location[1] - (80.0 - 55.0) / 80.0 * 100.0) < 0.01

    def test_pressure_defaults(
        self,
        adapter: StatsBombAdapter,
        fake_events_df: pd.DataFrame,
    ) -> None:
        """Pressure event uses default outcome and body part."""
        events = adapter._normalize_events(fake_events_df, "100")
        pressure_evt = next(e for e in events if e.event_type == "pressure")

        assert pressure_evt.event_outcome == "success"
        assert pressure_evt.body_part == "unknown"
        assert pressure_evt.under_pressure is False
        assert pressure_evt.freeze_frame is None


# ------------------------------------------------------------------
# Timestamp parsing
# ------------------------------------------------------------------


class TestParseTimestamp:
    """Verify timestamp string parsing."""

    def test_zero(self) -> None:
        """Zero timestamp parses to 0.0 seconds."""
        assert _parse_timestamp("00:00:00.000") == 0.0

    def test_minutes_and_seconds(self) -> None:
        """Timestamp with minutes and seconds parses correctly."""
        assert _parse_timestamp("00:01:30.000") == 90.0

    def test_fractional_seconds(self) -> None:
        """Fractional seconds are preserved."""
        result = _parse_timestamp("00:00:05.500")
        assert abs(result - 5.5) < 1e-9
