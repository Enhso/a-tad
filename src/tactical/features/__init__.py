"""Feature extraction framework for the Tactical State Discovery Engine.

Public API
----------
.. class:: FeatureExtractor

    Runtime-checkable protocol that all feature extractors must satisfy.

.. class:: FeatureRegistry

    Central registry that stores extractors, filters by tier, and
    coordinates batch extraction.
"""

from tactical.features.base import FeatureExtractor
from tactical.features.registry import FeatureRegistry

__all__ = ["FeatureExtractor", "FeatureRegistry"]
