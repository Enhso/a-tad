"""Feature extractor registry for the Tactical State Discovery Engine.

Provides :class:`FeatureRegistry`, the central coordinator that stores
:class:`~tactical.features.base.FeatureExtractor` instances, filters
them by tier, and orchestrates batch extraction with duplicate-name
detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tactical.adapters.schemas import MatchContext, NormalizedEvent
    from tactical.features.base import FeatureExtractor


class FeatureRegistry:
    """Manages feature extractors and coordinates extraction.

    Extractors are registered via :meth:`register`.  The registry
    filters by tier and runs all active extractors through
    :meth:`extract_all`.

    Example::

        registry = FeatureRegistry()
        registry.register(SpatialExtractor())
        registry.register(PressingExtractor())
        features = registry.extract_all(events, ctx, max_tier=2)
    """

    __slots__ = ("_extractors",)

    def __init__(self) -> None:
        self._extractors: list[FeatureExtractor] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, extractor: FeatureExtractor) -> None:
        """Register a feature extractor.

        Args:
            extractor: An object satisfying the
                :class:`~tactical.features.base.FeatureExtractor`
                protocol.
        """
        self._extractors.append(extractor)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_active_extractors(
        self,
        max_tier: int,
    ) -> list[FeatureExtractor]:
        """Return extractors up to and including the specified tier.

        Args:
            max_tier: Maximum tier to include (1, 2, or 3).

        Returns:
            List of extractors whose ``tier`` is ``<= max_tier``.
        """
        return [e for e in self._extractors if e.tier <= max_tier]

    def get_all_feature_names(
        self,
        max_tier: int,
    ) -> tuple[str, ...]:
        """Return all feature names for active tiers.

        Names are returned in registration order; each extractor's
        names appear in the order declared by its ``feature_names``
        property.

        Args:
            max_tier: Maximum tier to include.

        Returns:
            Combined tuple of feature name strings.
        """
        names: list[str] = []
        for extractor in self.get_active_extractors(max_tier):
            names.extend(extractor.feature_names)
        return tuple(names)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_all(
        self,
        events: tuple[NormalizedEvent, ...],
        context: MatchContext,
        max_tier: int = 2,
    ) -> dict[str, float | None]:
        """Run all active extractors and merge results.

        Args:
            events: Tuple of normalised events in the segment.
            context: Per-team match context.
            max_tier: Maximum tier to include (default ``2``).

        Returns:
            Dictionary mapping every feature name to its computed value.
            ``None`` indicates the feature could not be computed.

        Raises:
            ValueError: If two extractors produce the same feature name.
        """
        merged: dict[str, float | None] = {}
        for extractor in self.get_active_extractors(max_tier):
            result = extractor.extract(events, context)
            for name in result:
                if name in merged:
                    msg = f"Duplicate feature name {name!r}"
                    raise ValueError(msg)
            merged.update(result)
        return merged
