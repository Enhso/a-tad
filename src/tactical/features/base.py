"""Feature extractor protocol for the Tactical State Discovery Engine.

Defines the :class:`FeatureExtractor` structural interface that every
concrete feature extractor must satisfy.  Extractors declare their
tier (data requirements) and the names of features they produce.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from tactical.adapters.schemas import MatchContext, NormalizedEvent


@runtime_checkable
class FeatureExtractor(Protocol):
    """Protocol for feature extractors.

    Each extractor computes a group of related features from a
    sequence of events.  Extractors declare their tier (data
    requirements) and the names of features they produce.
    """

    @property
    def tier(self) -> int:
        """Feature tier: 1 (event-only), 2 (event-derived), 3 (360)."""
        ...

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names of features this extractor produces."""
        ...

    def extract(
        self,
        events: tuple[NormalizedEvent, ...],
        context: MatchContext,
    ) -> dict[str, float | None]:
        """Extract features from a sequence of events.

        Args:
            events: Tuple of normalized events in the segment.
            context: Per-team match context (IDs, home/away, 360 flag).

        Returns:
            Dictionary mapping feature names to values.
            ``None`` indicates the feature could not be computed.
        """
        ...
