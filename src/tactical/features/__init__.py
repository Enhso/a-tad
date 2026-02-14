"""Feature extraction framework for the Tactical State Discovery Engine.

Public API
----------
.. class:: FeatureExtractor

    Runtime-checkable protocol that all feature extractors must satisfy.

.. class:: FeatureRegistry

    Central registry that stores extractors, filters by tier, and
    coordinates batch extraction.

.. function:: create_default_registry

    Factory that returns a registry pre-loaded with all standard
    Tier 1 / 2 / 3 extractors.

.. function:: extract_match_features

    End-to-end pipeline: segments -> features -> polars DataFrame.
"""

from tactical.features.base import FeatureExtractor
from tactical.features.pipeline import extract_match_features
from tactical.features.registry import FeatureRegistry, create_default_registry
from tactical.features.tier1 import (
    CarryingFeatureExtractor,
    ContextFeatureExtractor,
    DefendingFeatureExtractor,
    PassingFeatureExtractor,
    ShootingFeatureExtractor,
    SpatialFeatureExtractor,
    TemporalFeatureExtractor,
)
from tactical.features.tier2 import (
    PressingFeatureExtractor,
    TeamShapeFeatureExtractor,
    TransitionFeatureExtractor,
    ZonalFeatureExtractor,
)
from tactical.features.tier3 import (
    FormationFeatureExtractor,
    RelationalFeatureExtractor,
)

__all__ = [
    "CarryingFeatureExtractor",
    "ContextFeatureExtractor",
    "DefendingFeatureExtractor",
    "FeatureExtractor",
    "FeatureRegistry",
    "FormationFeatureExtractor",
    "PassingFeatureExtractor",
    "PressingFeatureExtractor",
    "RelationalFeatureExtractor",
    "ShootingFeatureExtractor",
    "SpatialFeatureExtractor",
    "TeamShapeFeatureExtractor",
    "TemporalFeatureExtractor",
    "TransitionFeatureExtractor",
    "ZonalFeatureExtractor",
    "create_default_registry",
    "extract_match_features",
]
