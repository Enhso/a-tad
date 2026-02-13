"""Integration test for the full adapter pipeline.

Exercises the end-to-end flow: raw StatsBomb-shaped data is fetched
(mocked), normalized into :class:`NormalizedEvent` instances, cached to
disk, and served from the cache on subsequent calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pandas as pd

from tactical.adapters import (
    CONTROLLED_EVENT_TYPES,
    FreezeFramePlayer,
    NormalizedEvent,
    StatsBombAdapter,
)

if TYPE_CHECKING:
    from pathlib import Path

# ------------------------------------------------------------------
# Fixture data
# ------------------------------------------------------------------

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
        events_cached = adapter.load_match_events("999")

        mock_sb.events.assert_not_called()
        assert len(events_cached) == len(events)
        assert [e.event_id for e in events_cached] == [e.event_id for e in events]
