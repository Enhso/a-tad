"""Integration test for the full adapter pipeline.

Exercises the end-to-end flow: raw StatsBomb-shaped data is fetched
(mocked), normalized into :class:`NormalizedEvent` instances, cached to
disk, and served from the cache on subsequent calls.  Also validates
that lineup data is loaded, normalized, and cached independently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pandas as pd
import requests

from tactical.adapters import (
    CONTROLLED_EVENT_TYPES,
    CardEvent,
    FreezeFramePlayer,
    NormalizedEvent,
    PlayerLineup,
    PositionSpell,
    StatsBombAdapter,
    TeamLineup,
)

if TYPE_CHECKING:
    from pathlib import Path

# ------------------------------------------------------------------
# Fixture data
# ------------------------------------------------------------------

_INTEGRATION_LINEUPS: dict[str, dict[str, Any]] = {
    "1": {
        "team_id": 1,
        "team_name": "HomeFC",
        "lineup": [
            {
                "player_id": 10,
                "player_name": "Alice",
                "player_nickname": None,
                "jersey_number": 1,
                "country": {"id": 1, "name": "Testland"},
                "cards": [],
                "positions": [
                    {
                        "position_id": 1,
                        "position": "Goalkeeper",
                        "from": "00:00",
                        "to": None,
                        "from_period": 1,
                        "to_period": None,
                        "start_reason": "Starting XI",
                        "end_reason": "Final Whistle",
                    },
                ],
            },
            {
                "player_id": 40,
                "player_name": "Dave",
                "player_nickname": None,
                "jersey_number": 7,
                "country": {"id": 1, "name": "Testland"},
                "cards": [
                    {
                        "time": "67:40",
                        "card_type": "Yellow Card",
                        "reason": "Foul Committed",
                        "period": 2,
                    },
                ],
                "positions": [
                    {
                        "position_id": 17,
                        "position": "Right Wing",
                        "from": "00:00",
                        "to": "60:00",
                        "from_period": 1,
                        "to_period": 2,
                        "start_reason": "Starting XI",
                        "end_reason": "Tactical Shift",
                    },
                    {
                        "position_id": 23,
                        "position": "Center Forward",
                        "from": "60:00",
                        "to": None,
                        "from_period": 2,
                        "to_period": None,
                        "start_reason": "Tactical Shift",
                        "end_reason": "Final Whistle",
                    },
                ],
            },
            {
                "player_id": 50,
                "player_name": "Eve",
                "player_nickname": None,
                "jersey_number": 14,
                "country": {"id": 1, "name": "Testland"},
                "cards": [],
                "positions": [],
            },
        ],
    },
    "2": {
        "team_id": 2,
        "team_name": "AwayFC",
        "lineup": [
            {
                "player_id": 20,
                "player_name": "Bob",
                "player_nickname": None,
                "jersey_number": 9,
                "country": {"id": 2, "name": "Otherland"},
                "cards": [],
                "positions": [
                    {
                        "position_id": 23,
                        "position": "Center Forward",
                        "from": "00:00",
                        "to": None,
                        "from_period": 1,
                        "to_period": None,
                        "start_reason": "Starting XI",
                        "end_reason": "Final Whistle",
                    },
                ],
            },
            {
                "player_id": 30,
                "player_name": "Carol",
                "player_nickname": None,
                "jersey_number": 5,
                "country": {"id": 2, "name": "Otherland"},
                "cards": [
                    {
                        "time": "35:10",
                        "card_type": "Yellow Card",
                        "reason": "Foul Committed",
                        "period": 1,
                    },
                    {
                        "time": "80:10",
                        "card_type": "Second Yellow",
                        "reason": "Foul Committed",
                        "period": 2,
                    },
                ],
                "positions": [
                    {
                        "position_id": 5,
                        "position": "Left Center Back",
                        "from": "00:00",
                        "to": None,
                        "from_period": 1,
                        "to_period": None,
                        "start_reason": "Starting XI",
                        "end_reason": "Final Whistle",
                    },
                ],
            },
        ],
    },
}

_INTEGRATION_EVENTS: dict[str, list[Any]] = {
    "id": [
        "xi-home",
        "xi-away",
        "half-start-1",
        "half-start-2",
        "evt-pass-1",
        "evt-pass-2",
        "evt-pass-3",
        "evt-carry-1",
        "evt-carry-2",
        "evt-shot-1",
        "evt-pressure-1",
        "evt-duel-1",
        "evt-clearance-1",
        "evt-unknown-1",
    ],
    "index": list(range(1, 15)),
    "match_id": [999] * 14,
    "type": [
        "Starting XI",
        "Starting XI",
        "Half Start",
        "Half Start",
        "Pass",
        "Pass",
        "Pass",
        "Carry",
        "Carry",
        "Shot",
        "Pressure",
        "Duel",
        "Clearance",
        "Wizard Spell",
    ],
    "period": [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
    "timestamp": [
        "00:00:00.000",
        "00:00:00.000",
        "00:00:00.000",
        "00:00:00.000",
        "00:01:10.200",
        "00:05:22.800",
        "00:12:45.100",
        "00:20:00.500",
        "00:03:15.000",
        "00:10:30.600",
        "00:15:00.000",
        "00:22:40.300",
        "00:35:10.000",
        "00:40:00.000",
    ],
    "minute": [0, 0, 0, 0, 1, 5, 12, 20, 48, 55, 60, 67, 80, 85],
    "second": [0, 0, 0, 0, 10, 22, 45, 0, 15, 30, 0, 40, 10, 0],
    "team": [
        "HomeFC",
        "AwayFC",
        "HomeFC",
        "AwayFC",
        "HomeFC",
        "AwayFC",
        "HomeFC",
        "AwayFC",
        "HomeFC",
        "HomeFC",
        "AwayFC",
        "HomeFC",
        "AwayFC",
        "HomeFC",
    ],
    "team_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1],
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
        "Alice",
        "Bob",
        "Dave",
        "Carol",
        "Eve",
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
        10.0,
        20.0,
        40.0,
        30.0,
        50.0,
    ],
    "location": [
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        [30.0, 40.0],
        [60.0, 20.0],
        [90.0, 60.0],
        [45.0, 50.0],
        [15.0, 10.0],
        [108.0, 36.0],
        [70.0, 55.0],
        [55.0, 30.0],
        [10.0, 70.0],
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
        True,
        float("nan"),
        True,
        float("nan"),
        float("nan"),
    ],
    "pass_outcome": [
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        "Incomplete",
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
        "Left Foot",
        "Head",
        float("nan"),
        float("nan"),
        float("nan"),
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
        [50.0, 45.0],
        [80.0, 30.0],
        [100.0, 65.0],
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
    ],
    "pass_type": [float("nan")] * 14,
    "shot_outcome": [
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        "Saved",
        float("nan"),
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
    "shot_end_location": [
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        [120.0, 38.0, 0.8],
        float("nan"),
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
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        "Open Play",
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
    ],
    "shot_freeze_frame": [
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        [
            {
                "location": [115.0, 40.0],
                "player": {"id": 201, "name": "Keeper"},
                "position": {"id": 1, "name": "Goalkeeper"},
                "teammate": False,
            },
            {
                "location": [110.0, 32.0],
                "player": {"id": 202, "name": "CB"},
                "position": {"id": 3, "name": "Center Back"},
                "teammate": False,
            },
            {
                "location": [105.0, 44.0],
                "player": {"id": 203, "name": "Winger"},
                "position": {"id": 7, "name": "Left Wing"},
                "teammate": True,
            },
        ],
        float("nan"),
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
        float("nan"),
        [50.0, 48.0],
        [25.0, 15.0],
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
    ],
    "duel_type": [
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        "Aerial Lost",
        float("nan"),
        float("nan"),
    ],
    "duel_outcome": [
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        "Lost",
        float("nan"),
        float("nan"),
    ],
    "clearance_body_part": [
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        "Head",
        float("nan"),
    ],
    "interception_outcome": [float("nan")] * 14,
    "ball_receipt_outcome": [float("nan")] * 14,
    "dribble_outcome": [float("nan")] * 14,
    "goalkeeper_outcome": [float("nan")] * 14,
    "goalkeeper_body_part": [float("nan")] * 14,
    "goalkeeper_end_location": [float("nan")] * 14,
    "position": [float("nan")] * 14,
    "play_pattern": ["Regular Play"] * 14,
    "possession": list(range(1, 15)),
    "possession_team": ["HomeFC"] * 14,
    "possession_team_id": [1] * 14,
    "duration": [0.0] * 14,
    "related_events": [float("nan")] * 14,
    "tactics": [float("nan")] * 14,
    "substitution_outcome": [float("nan")] * 14,
    "substitution_outcome_id": [float("nan")] * 14,
    "substitution_replacement": [float("nan")] * 14,
    "substitution_replacement_id": [float("nan")] * 14,
}


# ------------------------------------------------------------------
# Integration test
# ------------------------------------------------------------------


class TestFullAdapterPipeline:
    """End-to-end integration test for the adapter layer."""

    @patch("tactical.adapters.statsbomb.sb")
    def test_full_adapter_pipeline(
        self,
        mock_sb: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Exercise fetch -> normalize -> cache -> cache-hit flow.

        The mock data contains 14 rows total:
          - 2 Starting XI  (skipped silently)
          - 2 Half Start   (skipped silently)
          - 3 Pass         (1 incomplete)
          - 2 Carry
          - 1 Shot         (with freeze frame)
          - 1 Pressure
          - 1 Duel
          - 1 Clearance
          - 1 "Wizard Spell" (unknown, skipped with warning)

        Expected normalised output: 9 events.
        """
        fake_df = pd.DataFrame(_INTEGRATION_EVENTS)
        mock_sb.events.return_value = fake_df
        mock_sb.frames.side_effect = requests.exceptions.HTTPError("404")

        adapter = StatsBombAdapter(cache_dir=tmp_path / "match_cache")

        # -- First call: cache miss -> API invoked ----------------------
        events = adapter.load_match_events("999")

        mock_sb.events.assert_called_once_with(
            match_id=999,
            flatten_attrs=True,
        )

        # -- Count & type assertions ------------------------------------
        assert len(events) == 9

        for evt in events:
            assert isinstance(evt, NormalizedEvent)
            assert evt.event_type in CONTROLLED_EVENT_TYPES

        actual_types = [e.event_type for e in events]
        assert actual_types.count("pass") == 3
        assert actual_types.count("carry") == 2
        assert actual_types.count("shot") == 1
        assert actual_types.count("pressure") == 1
        assert actual_types.count("duel") == 1
        assert actual_types.count("clearance") == 1

        # -- Coordinate range check (0-100) -----------------------------
        for evt in events:
            if evt.location is not None:
                x, y = evt.location
                assert 0.0 <= x <= 100.0, f"x={x} out of range"
                assert 0.0 <= y <= 100.0, f"y={y} out of range"
            if evt.end_location is not None:
                x, y = evt.end_location
                assert 0.0 <= x <= 100.0, f"end x={x} out of range"
                assert 0.0 <= y <= 100.0, f"end y={y} out of range"

        # -- Freeze frame on the shot event -----------------------------
        shot_events = [e for e in events if e.event_type == "shot"]
        assert len(shot_events) == 1
        shot = shot_events[0]

        assert shot.freeze_frame is not None
        assert len(shot.freeze_frame) == 3
        for player in shot.freeze_frame:
            assert isinstance(player, FreezeFramePlayer)
            px, py = player.location
            assert 0.0 <= px <= 100.0
            assert 0.0 <= py <= 100.0

        # -- Temporal sort: (period, timestamp) -------------------------
        for i in range(len(events) - 1):
            cur = events[i]
            nxt = events[i + 1]
            assert (cur.period, cur.timestamp) <= (nxt.period, nxt.timestamp), (
                f"Events not sorted at index {i}: "
                f"({cur.period}, {cur.timestamp}) > "
                f"({nxt.period}, {nxt.timestamp})"
            )

        # -- Incomplete pass outcome ------------------------------------
        incomplete = [e for e in events if e.event_outcome == "incomplete"]
        assert len(incomplete) == 1
        assert incomplete[0].event_type == "pass"

        # -- Cache file was created -------------------------------------
        cache_dir = tmp_path / "match_cache"
        cache_files = list(cache_dir.glob("match_999.pkl"))
        assert len(cache_files) == 1

        # -- Second call: cache hit -> API NOT called again -------------
        mock_sb.events.reset_mock()
        mock_sb.frames.reset_mock()
        events_cached = adapter.load_match_events("999")

        mock_sb.events.assert_not_called()
        assert len(events_cached) == len(events)
        assert [e.event_id for e in events_cached] == [e.event_id for e in events]

    @patch("tactical.adapters.statsbomb.sb")
    def test_full_lineup_pipeline(
        self,
        mock_sb: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Exercise lineup fetch -> normalize -> cache -> cache-hit flow.

        The mock lineup data contains two teams:
          - HomeFC (team 1): 2 starters + 1 unused sub
          - AwayFC (team 2): 2 starters

        Validates schema types, starter detection, position spells,
        cards, caching, and cache-key isolation from events.
        """
        mock_sb.lineups.return_value = _INTEGRATION_LINEUPS

        adapter = StatsBombAdapter(cache_dir=tmp_path / "match_cache")

        # -- First call: cache miss -> API invoked ----------------------
        lineups = adapter.load_match_lineups("999")

        mock_sb.lineups.assert_called_once_with(match_id=999, fmt="dict")

        # -- Two teams returned -----------------------------------------
        assert len(lineups) == 2
        assert "1" in lineups
        assert "2" in lineups

        # -- All values are TeamLineup instances ------------------------
        for team in lineups.values():
            assert isinstance(team, TeamLineup)
            for player in team.players:
                assert isinstance(player, PlayerLineup)
                for pos in player.positions:
                    assert isinstance(pos, PositionSpell)
                for card in player.cards:
                    assert isinstance(card, CardEvent)

        # -- Team names preserved ---------------------------------------
        assert lineups["1"].team_name == "HomeFC"
        assert lineups["2"].team_name == "AwayFC"

        # -- Player counts (starters + subs + unused) -------------------
        assert len(lineups["1"].players) == 3
        assert len(lineups["2"].players) == 2

        # -- Starter detection ------------------------------------------
        home = lineups["1"]
        alice = home.players[0]
        dave = home.players[1]
        eve = home.players[2]
        assert alice.starter is True
        assert dave.starter is True
        assert eve.starter is False
        assert eve.positions == ()

        # -- Player IDs are strings -------------------------------------
        assert alice.player_id == "10"
        assert dave.player_id == "40"

        # -- Position spells: tactical shift ----------------------------
        assert len(dave.positions) == 2
        assert dave.positions[0].position == "Right Wing"
        assert dave.positions[0].end_reason == "Tactical Shift"
        assert dave.positions[1].position == "Center Forward"
        assert dave.positions[1].start_reason == "Tactical Shift"
        assert dave.positions[1].to_time is None

        # -- Cards: single card -----------------------------------------
        assert len(dave.cards) == 1
        assert dave.cards[0].card_type == "Yellow Card"
        assert dave.cards[0].period == 2

        # -- Cards: double yellow ---------------------------------------
        carol = lineups["2"].players[1]
        assert len(carol.cards) == 2
        assert carol.cards[0].card_type == "Yellow Card"
        assert carol.cards[1].card_type == "Second Yellow"

        # -- Lineup cache file created ----------------------------------
        cache_dir = tmp_path / "match_cache"
        lineup_cache = list(cache_dir.glob("match_999_lineups.pkl"))
        assert len(lineup_cache) == 1

        # -- Second call: cache hit -> API NOT called again -------------
        mock_sb.lineups.reset_mock()
        lineups_cached = adapter.load_match_lineups("999")

        mock_sb.lineups.assert_not_called()
        assert len(lineups_cached) == len(lineups)
        for tid in lineups:
            assert len(lineups_cached[tid].players) == len(lineups[tid].players)

    @patch("tactical.adapters.statsbomb.sb")
    def test_events_and_lineups_cached_independently(
        self,
        mock_sb: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Event and lineup caches use distinct keys and don't interfere."""
        fake_df = pd.DataFrame(_INTEGRATION_EVENTS)
        mock_sb.events.return_value = fake_df
        mock_sb.frames.side_effect = requests.exceptions.HTTPError("404")
        mock_sb.lineups.return_value = _INTEGRATION_LINEUPS

        adapter = StatsBombAdapter(cache_dir=tmp_path / "match_cache")

        # Load events only
        adapter.load_match_events("999")
        mock_sb.events.assert_called_once()
        mock_sb.lineups.assert_not_called()

        # Load lineups only
        adapter.load_match_lineups("999")
        mock_sb.lineups.assert_called_once()

        cache_dir = tmp_path / "match_cache"
        event_cache = cache_dir / "match_999.pkl"
        lineup_cache = cache_dir / "match_999_lineups.pkl"
        assert event_cache.is_file()
        assert lineup_cache.is_file()

        # Both cached: neither API called on second pass
        mock_sb.events.reset_mock()
        mock_sb.lineups.reset_mock()

        adapter.load_match_events("999")
        adapter.load_match_lineups("999")

        mock_sb.events.assert_not_called()
        mock_sb.lineups.assert_not_called()
