"""Abstract adapter protocol for data providers.

Defines the :class:`DataAdapter` structural interface that every
concrete data-provider adapter must satisfy.  Adapters are responsible
for:

* **Coordinate normalization** -- mapping provider-specific pitch
  coordinates to a common 0-100 range on both axes.
* **Event type mapping** -- translating provider event labels into the
  canonical set defined by :data:`~tactical.adapters.schemas.CONTROLLED_EVENT_TYPES`.
* **Directional normalization** -- ensuring all events for a given team
  are expressed in a consistent attacking direction.
* **Freeze frame extraction** -- converting provider 360 data into
  :class:`~tactical.adapters.schemas.FreezeFramePlayer` tuples when
  available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from tactical.adapters.schemas import MatchInfo, NormalizedEvent


@runtime_checkable
class DataAdapter(Protocol):
    """Structural interface for match-data providers.

    Any class that implements the three methods below is a valid
    ``DataAdapter`` without needing to inherit from this class.
    """

    def list_matches(self, competition_id: int, season_id: int) -> list[MatchInfo]:
        """Return metadata for every match in a competition season.

        Args:
            competition_id: Provider-specific competition identifier.
            season_id: Provider-specific season identifier.

        Returns:
            List of :class:`MatchInfo` instances, one per match.
        """
        ...

    def load_match_events(self, match_id: str) -> list[NormalizedEvent]:
        """Load and normalize all events for a single match.

        The implementation must perform coordinate normalization, event
        type mapping, directional normalization, and freeze frame
        extraction before returning.

        Args:
            match_id: Unique match identifier as used by the provider.

        Returns:
            List of :class:`NormalizedEvent` instances sorted by
            ``(period, timestamp)``.
        """
        ...

    def supports_360(self) -> bool:
        """Indicate whether this adapter can supply 360 freeze frames.

        Returns:
            ``True`` if the provider offers 360 data, ``False``
            otherwise.
        """
        ...
