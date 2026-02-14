"""Data adapter layer for the Tactical State Discovery Engine.

Re-exports the canonical schemas, the adapter protocol, the StatsBomb
concrete adapter, and the disk-based match cache so that downstream
code can import everything from :mod:`tactical.adapters`.
"""

from tactical.adapters.base import DataAdapter
from tactical.adapters.cache import MatchCache
from tactical.adapters.schemas import (
    CONTROLLED_EVENT_TYPES,
    CardEvent,
    FreezeFramePlayer,
    MatchContext,
    MatchInfo,
    NormalizedEvent,
    PlayerLineup,
    PositionSpell,
    TeamLineup,
)
from tactical.adapters.statsbomb import StatsBombAdapter

__all__ = [
    "CONTROLLED_EVENT_TYPES",
    "CardEvent",
    "DataAdapter",
    "FreezeFramePlayer",
    "MatchCache",
    "MatchContext",
    "MatchInfo",
    "NormalizedEvent",
    "PlayerLineup",
    "PositionSpell",
    "StatsBombAdapter",
    "TeamLineup",
]
