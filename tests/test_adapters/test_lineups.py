"""Tests for lineup loading and normalization.

Validates the :class:`CardEvent`, :class:`PositionSpell`,
:class:`PlayerLineup`, and :class:`TeamLineup` schemas, the
:meth:`StatsBombAdapter.load_match_lineups` cache-first loading, the
``_normalize_lineups`` helper, and edge cases (unused subs, empty
cards/positions).

All tests mock ``statsbombpy.sb`` to avoid real API calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from tactical.adapters.schemas import (
    CardEvent,
    PlayerLineup,
    PositionSpell,
    TeamLineup,
)
from tactical.adapters.statsbomb import (
    StatsBombAdapter,
    _build_card_events,
    _build_position_spells,
)

if TYPE_CHECKING:
    from pathlib import Path


# ------------------------------------------------------------------
# Fixture data
# ------------------------------------------------------------------

_RAW_LINEUPS: dict[str, dict[str, Any]] = {
    "100": {
        "team_id": 100,
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
                "player_id": 11,
                "player_name": "Bob",
                "player_nickname": "Bobby",
                "jersey_number": 4,
                "country": {"id": 1, "name": "Testland"},
                "cards": [
                    {
                        "time": "32:15",
                        "card_type": "Yellow Card",
                        "reason": "Foul Committed",
                        "period": 1,
                    },
                ],
                "positions": [
                    {
                        "position_id": 3,
                        "position": "Right Center Back",
                        "from": "00:00",
                        "to": "65:00",
                        "from_period": 1,
                        "to_period": 2,
                        "start_reason": "Starting XI",
                        "end_reason": "Substitution - Off (Tactical)",
                    },
                ],
            },
            {
                "player_id": 12,
                "player_name": "Carol",
                "player_nickname": None,
                "jersey_number": 17,
                "country": {"id": 2, "name": "Otherland"},
                "cards": [],
                "positions": [
                    {
                        "position_id": 3,
                        "position": "Right Center Back",
                        "from": "65:00",
                        "to": None,
                        "from_period": 2,
                        "to_period": None,
                        "start_reason": "Substitution - On (Tactical)",
                        "end_reason": "Final Whistle",
                    },
                ],
            },
            {
                "player_id": 13,
                "player_name": "Dave",
                "player_nickname": None,
                "jersey_number": 22,
                "country": {"id": 1, "name": "Testland"},
                "cards": [],
                "positions": [],
            },
        ],
    },
    "200": {
        "team_id": 200,
        "team_name": "AwayFC",
        "lineup": [
            {
                "player_id": 20,
                "player_name": "Eve",
                "player_nickname": None,
                "jersey_number": 9,
                "country": {"id": 3, "name": "Farland"},
                "cards": [
                    {
                        "time": "55:00",
                        "card_type": "Yellow Card",
                        "reason": "Foul Committed",
                        "period": 2,
                    },
                    {
                        "time": "78:30",
                        "card_type": "Second Yellow",
                        "reason": "Foul Committed",
                        "period": 2,
                    },
                ],
                "positions": [
                    {
                        "position_id": 23,
                        "position": "Center Forward",
                        "from": "00:00",
                        "to": "72:00",
                        "from_period": 1,
                        "to_period": 2,
                        "start_reason": "Starting XI",
                        "end_reason": "Tactical Shift",
                    },
                    {
                        "position_id": 22,
                        "position": "Right Center Forward",
                        "from": "72:00",
                        "to": None,
                        "from_period": 2,
                        "to_period": None,
                        "start_reason": "Tactical Shift",
                        "end_reason": "Final Whistle",
                    },
                ],
            },
        ],
    },
}


@pytest.fixture()
def raw_lineups() -> dict[str, dict[str, Any]]:
    """Return fake raw lineup data shaped like ``statsbombpy`` output."""
    return _RAW_LINEUPS


@pytest.fixture()
def adapter(tmp_path: Path) -> StatsBombAdapter:
    """Return a StatsBombAdapter backed by a temporary cache directory."""
    return StatsBombAdapter(cache_dir=tmp_path / "cache")


# ------------------------------------------------------------------
# Schema immutability
# ------------------------------------------------------------------


class TestSchemaImmutability:
    """Verify that lineup schemas are frozen."""

    def test_card_event_frozen(self) -> None:
        """CardEvent instances cannot be mutated."""
        card = CardEvent(time="10:00", card_type="Yellow Card", reason="Foul", period=1)
        with pytest.raises(AttributeError):
            card.time = "20:00"  # type: ignore[misc]

    def test_position_spell_frozen(self) -> None:
        """PositionSpell instances cannot be mutated."""
        spell = PositionSpell(
            position_id=1,
            position="Goalkeeper",
            from_time="00:00",
            to_time=None,
            from_period=1,
            to_period=None,
            start_reason="Starting XI",
            end_reason="Final Whistle",
        )
        with pytest.raises(AttributeError):
            spell.position = "Striker"  # type: ignore[misc]

    def test_player_lineup_frozen(self) -> None:
        """PlayerLineup instances cannot be mutated."""
        player = PlayerLineup(
            player_id="1",
            player_name="Test",
            jersey_number=10,
            starter=True,
            positions=(),
            cards=(),
        )
        with pytest.raises(AttributeError):
            player.starter = False  # type: ignore[misc]

    def test_team_lineup_frozen(self) -> None:
        """TeamLineup instances cannot be mutated."""
        team = TeamLineup(team_id="1", team_name="FC Test", players=())
        with pytest.raises(AttributeError):
            team.team_name = "New Name"  # type: ignore[misc]


# ------------------------------------------------------------------
# Schema creation
# ------------------------------------------------------------------


class TestSchemaCreation:
    """Verify all fields are accessible on lineup schema instances."""

    def test_card_event_fields(self) -> None:
        """All CardEvent fields are stored correctly."""
        card = CardEvent(
            time="45:30", card_type="Red Card", reason="Violent Conduct", period=1
        )
        assert card.time == "45:30"
        assert card.card_type == "Red Card"
        assert card.reason == "Violent Conduct"
        assert card.period == 1

    def test_position_spell_fields(self) -> None:
        """All PositionSpell fields are stored correctly."""
        spell = PositionSpell(
            position_id=5,
            position="Left Center Back",
            from_time="00:00",
            to_time="70:00",
            from_period=1,
            to_period=2,
            start_reason="Starting XI",
            end_reason="Substitution - Off (Tactical)",
        )
        assert spell.position_id == 5
        assert spell.position == "Left Center Back"
        assert spell.from_time == "00:00"
        assert spell.to_time == "70:00"
        assert spell.from_period == 1
        assert spell.to_period == 2
        assert spell.start_reason == "Starting XI"
        assert spell.end_reason == "Substitution - Off (Tactical)"

    def test_position_spell_none_fields(self) -> None:
        """PositionSpell accepts None for time and period boundaries."""
        spell = PositionSpell(
            position_id=1,
            position="Goalkeeper",
            from_time=None,
            to_time=None,
            from_period=None,
            to_period=None,
            start_reason="Starting XI",
            end_reason="Final Whistle",
        )
        assert spell.from_time is None
        assert spell.to_time is None
        assert spell.from_period is None
        assert spell.to_period is None

    def test_player_lineup_fields(self) -> None:
        """All PlayerLineup fields are stored correctly."""
        card = CardEvent(time="10:00", card_type="Yellow Card", reason="Foul", period=1)
        spell = PositionSpell(
            position_id=9,
            position="Right Defensive Midfield",
            from_time="00:00",
            to_time=None,
            from_period=1,
            to_period=None,
            start_reason="Starting XI",
            end_reason="Final Whistle",
        )
        player = PlayerLineup(
            player_id="42",
            player_name="Test Player",
            jersey_number=8,
            starter=True,
            positions=(spell,),
            cards=(card,),
        )
        assert player.player_id == "42"
        assert player.player_name == "Test Player"
        assert player.jersey_number == 8
        assert player.starter is True
        assert len(player.positions) == 1
        assert len(player.cards) == 1

    def test_team_lineup_fields(self) -> None:
        """All TeamLineup fields are stored correctly."""
        player = PlayerLineup(
            player_id="1",
            player_name="Solo",
            jersey_number=1,
            starter=True,
            positions=(),
            cards=(),
        )
        team = TeamLineup(team_id="99", team_name="FC Test", players=(player,))
        assert team.team_id == "99"
        assert team.team_name == "FC Test"
        assert len(team.players) == 1
        assert team.players[0].player_name == "Solo"


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------


class TestBuildPositionSpells:
    """Verify _build_position_spells produces correct tuples."""

    def test_single_position(self) -> None:
        """A single position entry yields a one-element tuple."""
        raw = [
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
        ]
        result = _build_position_spells(raw)
        assert len(result) == 1
        assert isinstance(result, tuple)
        assert result[0].position == "Goalkeeper"
        assert result[0].start_reason == "Starting XI"

    def test_multiple_positions(self) -> None:
        """Multiple position entries preserve order."""
        raw = [
            {
                "position_id": 23,
                "position": "Center Forward",
                "from": "00:00",
                "to": "60:00",
                "from_period": 1,
                "to_period": 2,
                "start_reason": "Starting XI",
                "end_reason": "Tactical Shift",
            },
            {
                "position_id": 21,
                "position": "Left Wing",
                "from": "60:00",
                "to": None,
                "from_period": 2,
                "to_period": None,
                "start_reason": "Tactical Shift",
                "end_reason": "Final Whistle",
            },
        ]
        result = _build_position_spells(raw)
        assert len(result) == 2
        assert result[0].position == "Center Forward"
        assert result[1].position == "Left Wing"

    def test_empty_positions(self) -> None:
        """An empty list yields an empty tuple (unused substitute)."""
        result = _build_position_spells([])
        assert result == ()

    def test_missing_start_reason_defaults(self) -> None:
        """A missing start_reason defaults to 'unknown'."""
        raw = [
            {
                "position_id": 2,
                "position": "Right Back",
                "from": "00:00",
                "to": None,
                "from_period": 1,
                "to_period": None,
            },
        ]
        result = _build_position_spells(raw)
        assert result[0].start_reason == "unknown"
        assert result[0].end_reason == "unknown"


class TestBuildCardEvents:
    """Verify _build_card_events produces correct tuples."""

    def test_single_card(self) -> None:
        """A single card entry yields a one-element tuple."""
        raw = [
            {
                "time": "44:00",
                "card_type": "Yellow Card",
                "reason": "Foul Committed",
                "period": 1,
            },
        ]
        result = _build_card_events(raw)
        assert len(result) == 1
        assert isinstance(result, tuple)
        assert result[0].card_type == "Yellow Card"
        assert result[0].time == "44:00"

    def test_multiple_cards(self) -> None:
        """Multiple cards preserve order."""
        raw = [
            {
                "time": "30:00",
                "card_type": "Yellow Card",
                "reason": "Foul Committed",
                "period": 1,
            },
            {
                "time": "88:00",
                "card_type": "Second Yellow",
                "reason": "Foul Committed",
                "period": 2,
            },
        ]
        result = _build_card_events(raw)
        assert len(result) == 2
        assert result[0].card_type == "Yellow Card"
        assert result[1].card_type == "Second Yellow"

    def test_empty_cards(self) -> None:
        """An empty list yields an empty tuple."""
        result = _build_card_events([])
        assert result == ()

    def test_missing_reason_defaults(self) -> None:
        """A missing reason defaults to 'unknown'."""
        raw = [
            {
                "time": "10:00",
                "card_type": "Red Card",
                "period": 1,
            },
        ]
        result = _build_card_events(raw)
        assert result[0].reason == "unknown"


# ------------------------------------------------------------------
# Normalization
# ------------------------------------------------------------------


class TestNormalizeLineups:
    """Verify _normalize_lineups produces the correct schema objects."""

    def test_team_count(self, raw_lineups: dict[str, dict[str, Any]]) -> None:
        """Two teams in the raw data produce two TeamLineup entries."""
        result = StatsBombAdapter._normalize_lineups(raw_lineups)
        assert len(result) == 2

    def test_team_ids_are_strings(self, raw_lineups: dict[str, dict[str, Any]]) -> None:
        """Team IDs in the result dict are strings."""
        result = StatsBombAdapter._normalize_lineups(raw_lineups)
        assert "100" in result
        assert "200" in result

    def test_team_name(self, raw_lineups: dict[str, dict[str, Any]]) -> None:
        """Team names are preserved in the TeamLineup."""
        result = StatsBombAdapter._normalize_lineups(raw_lineups)
        assert result["100"].team_name == "HomeFC"
        assert result["200"].team_name == "AwayFC"

    def test_player_count(self, raw_lineups: dict[str, dict[str, Any]]) -> None:
        """All players (starters + subs + unused) are included."""
        result = StatsBombAdapter._normalize_lineups(raw_lineups)
        assert len(result["100"].players) == 4
        assert len(result["200"].players) == 1

    def test_player_id_is_string(self, raw_lineups: dict[str, dict[str, Any]]) -> None:
        """Player IDs are converted to strings."""
        result = StatsBombAdapter._normalize_lineups(raw_lineups)
        ids = [p.player_id for p in result["100"].players]
        assert ids == ["10", "11", "12", "13"]

    def test_player_instances(self, raw_lineups: dict[str, dict[str, Any]]) -> None:
        """Every player is a PlayerLineup instance."""
        result = StatsBombAdapter._normalize_lineups(raw_lineups)
        for team in result.values():
            assert isinstance(team, TeamLineup)
            for player in team.players:
                assert isinstance(player, PlayerLineup)


# ------------------------------------------------------------------
# Starter detection
# ------------------------------------------------------------------


class TestStarterDetection:
    """Verify the starter flag is derived from position start_reason."""

    def test_starting_xi_flagged(self, raw_lineups: dict[str, dict[str, Any]]) -> None:
        """Players whose first position has start_reason 'Starting XI' are starters."""
        result = StatsBombAdapter._normalize_lineups(raw_lineups)
        home = result["100"]
        alice = home.players[0]
        bob = home.players[1]
        assert alice.starter is True
        assert bob.starter is True

    def test_substitute_not_starter(
        self, raw_lineups: dict[str, dict[str, Any]]
    ) -> None:
        """A player who came on as a sub is not a starter."""
        result = StatsBombAdapter._normalize_lineups(raw_lineups)
        home = result["100"]
        carol = home.players[2]
        assert carol.starter is False

    def test_unused_substitute_not_starter(
        self, raw_lineups: dict[str, dict[str, Any]]
    ) -> None:
        """A player with no positions (unused sub) is not a starter."""
        result = StatsBombAdapter._normalize_lineups(raw_lineups)
        home = result["100"]
        dave = home.players[3]
        assert dave.starter is False
        assert dave.positions == ()

    def test_tactical_shift_starter(
        self, raw_lineups: dict[str, dict[str, Any]]
    ) -> None:
        """A starter who later shifted position retains starter=True."""
        result = StatsBombAdapter._normalize_lineups(raw_lineups)
        away = result["200"]
        eve = away.players[0]
        assert eve.starter is True
        assert len(eve.positions) == 2


# ------------------------------------------------------------------
# Card normalization
# ------------------------------------------------------------------


class TestCardNormalization:
    """Verify cards are correctly attached to their players."""

    def test_player_with_no_cards(self, raw_lineups: dict[str, dict[str, Any]]) -> None:
        """A player with no cards has an empty cards tuple."""
        result = StatsBombAdapter._normalize_lineups(raw_lineups)
        alice = result["100"].players[0]
        assert alice.cards == ()

    def test_player_with_one_card(self, raw_lineups: dict[str, dict[str, Any]]) -> None:
        """Bob received one yellow card."""
        result = StatsBombAdapter._normalize_lineups(raw_lineups)
        bob = result["100"].players[1]
        assert len(bob.cards) == 1
        assert isinstance(bob.cards[0], CardEvent)
        assert bob.cards[0].card_type == "Yellow Card"
        assert bob.cards[0].time == "32:15"
        assert bob.cards[0].reason == "Foul Committed"
        assert bob.cards[0].period == 1

    def test_player_with_multiple_cards(
        self, raw_lineups: dict[str, dict[str, Any]]
    ) -> None:
        """Eve received a yellow followed by a second yellow."""
        result = StatsBombAdapter._normalize_lineups(raw_lineups)
        eve = result["200"].players[0]
        assert len(eve.cards) == 2
        assert eve.cards[0].card_type == "Yellow Card"
        assert eve.cards[1].card_type == "Second Yellow"


# ------------------------------------------------------------------
# Position spell normalization
# ------------------------------------------------------------------


class TestPositionSpellNormalization:
    """Verify position spells are correctly extracted."""

    def test_single_spell_full_match(
        self, raw_lineups: dict[str, dict[str, Any]]
    ) -> None:
        """Alice played Goalkeeper the entire match."""
        result = StatsBombAdapter._normalize_lineups(raw_lineups)
        alice = result["100"].players[0]
        assert len(alice.positions) == 1
        spell = alice.positions[0]
        assert isinstance(spell, PositionSpell)
        assert spell.position == "Goalkeeper"
        assert spell.from_time == "00:00"
        assert spell.to_time is None
        assert spell.from_period == 1
        assert spell.to_period is None
        assert spell.start_reason == "Starting XI"
        assert spell.end_reason == "Final Whistle"

    def test_spell_with_substitution_boundary(
        self, raw_lineups: dict[str, dict[str, Any]]
    ) -> None:
        """Bob's spell ended at 65:00 due to substitution."""
        result = StatsBombAdapter._normalize_lineups(raw_lineups)
        bob = result["100"].players[1]
        assert bob.positions[0].to_time == "65:00"
        assert bob.positions[0].end_reason == "Substitution - Off (Tactical)"

    def test_tactical_shift_spells(
        self, raw_lineups: dict[str, dict[str, Any]]
    ) -> None:
        """Eve had two spells due to a tactical shift."""
        result = StatsBombAdapter._normalize_lineups(raw_lineups)
        eve = result["200"].players[0]
        assert len(eve.positions) == 2
        assert eve.positions[0].end_reason == "Tactical Shift"
        assert eve.positions[1].start_reason == "Tactical Shift"
        assert eve.positions[1].position == "Right Center Forward"


# ------------------------------------------------------------------
# Cache integration
# ------------------------------------------------------------------


class TestCacheUsage:
    """Verify load_match_lineups uses the cache before calling the API."""

    @patch("tactical.adapters.statsbomb.sb")
    def test_load_match_lineups_caches(
        self,
        mock_sb: MagicMock,
        adapter: StatsBombAdapter,
        raw_lineups: dict[str, dict[str, Any]],
    ) -> None:
        """On cache miss the API is called; on hit it is not."""
        mock_sb.lineups.return_value = raw_lineups

        # First call: cache miss -> API called
        result_1 = adapter.load_match_lineups("999")
        mock_sb.lineups.assert_called_once_with(match_id=999, fmt="dict")
        assert len(result_1) == 2

        # Second call: cache hit -> API NOT called again
        mock_sb.lineups.reset_mock()
        result_2 = adapter.load_match_lineups("999")
        mock_sb.lineups.assert_not_called()
        assert len(result_2) == len(result_1)

    @patch("tactical.adapters.statsbomb.sb")
    def test_lineup_cache_key_separate_from_events(
        self,
        mock_sb: MagicMock,
        adapter: StatsBombAdapter,
        raw_lineups: dict[str, dict[str, Any]],
    ) -> None:
        """Lineup cache uses a distinct key from event cache."""
        mock_sb.lineups.return_value = raw_lineups

        adapter.load_match_lineups("500")

        cache_path = adapter._cache.cache_path("500_lineups")
        events_path = adapter._cache.cache_path("500")
        assert cache_path.is_file()
        assert not events_path.is_file()


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    """Verify behaviour for unusual or minimal lineup data."""

    def test_empty_lineup_list(self) -> None:
        """A team with an empty lineup list produces zero players."""
        raw: dict[str, dict[str, Any]] = {
            "1": {
                "team_id": 1,
                "team_name": "EmptyFC",
                "lineup": [],
            },
        }
        result = StatsBombAdapter._normalize_lineups(raw)
        assert len(result["1"].players) == 0

    def test_player_missing_jersey_number_defaults(self) -> None:
        """A player without a jersey_number field gets 0."""
        raw: dict[str, dict[str, Any]] = {
            "1": {
                "team_id": 1,
                "team_name": "FC",
                "lineup": [
                    {
                        "player_id": 99,
                        "player_name": "Ghost",
                        "positions": [],
                        "cards": [],
                    },
                ],
            },
        }
        result = StatsBombAdapter._normalize_lineups(raw)
        assert result["1"].players[0].jersey_number == 0

    def test_single_team_only(self) -> None:
        """A lineup dict with one team produces a single-entry result."""
        raw: dict[str, dict[str, Any]] = {
            "42": {
                "team_id": 42,
                "team_name": "Solo United",
                "lineup": [
                    {
                        "player_id": 7,
                        "player_name": "Lone Player",
                        "jersey_number": 7,
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
                        "cards": [],
                    },
                ],
            },
        }
        result = StatsBombAdapter._normalize_lineups(raw)
        assert len(result) == 1
        assert "42" in result
        assert result["42"].players[0].starter is True

    def test_all_instances_frozen(self) -> None:
        """Verify nested structures are all frozen dataclass instances."""
        raw: dict[str, dict[str, Any]] = {
            "1": {
                "team_id": 1,
                "team_name": "FC",
                "lineup": [
                    {
                        "player_id": 1,
                        "player_name": "Test",
                        "jersey_number": 10,
                        "positions": [
                            {
                                "position_id": 1,
                                "position": "GK",
                                "from": "00:00",
                                "to": None,
                                "from_period": 1,
                                "to_period": None,
                                "start_reason": "Starting XI",
                                "end_reason": "Final Whistle",
                            },
                        ],
                        "cards": [
                            {
                                "time": "10:00",
                                "card_type": "Yellow Card",
                                "reason": "Foul",
                                "period": 1,
                            },
                        ],
                    },
                ],
            },
        }
        result = StatsBombAdapter._normalize_lineups(raw)
        team = result["1"]
        player = team.players[0]

        with pytest.raises(AttributeError):
            team.players = ()  # type: ignore[misc]
        with pytest.raises(AttributeError):
            player.positions = ()  # type: ignore[misc]
        with pytest.raises(AttributeError):
            player.cards[0].time = "99:99"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            player.positions[0].position = "Striker"  # type: ignore[misc]
