"""Model layer for tactical state discovery.

Provides preprocessing utilities and model backends for clustering
and sequence modelling of tactical features.
"""

from tactical.models.base import TacticalModel
from tactical.models.discovery import DatasetDiscoveryResult, run_gmm_discovery
from tactical.models.gmm import GMMConfig, GMMModel
from tactical.models.preprocessing import PreprocessingPipeline
from tactical.models.selection import ModelSelectionResult, select_gmm_k

__all__ = [
    "DatasetDiscoveryResult",
    "GMMConfig",
    "GMMModel",
    "ModelSelectionResult",
    "PreprocessingPipeline",
    "TacticalModel",
    "run_gmm_discovery",
    "select_gmm_k",
]
