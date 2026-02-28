"""Model layer for tactical state discovery.

Provides preprocessing utilities and model backends for clustering
and sequence modelling of tactical features.
"""

from tactical.models.base import TacticalModel
from tactical.models.changepoint import (
    Changepoint,
    ChangepointResult,
    detect_changepoints,
    detect_changepoints_per_match,
)
from tactical.models.discovery import DatasetDiscoveryResult, run_gmm_discovery
from tactical.models.gmm import GMMConfig, GMMModel
from tactical.models.hmm import HMMModel
from tactical.models.hmm_discovery import (
    HMMDiscoveryResult,
    plot_transition_matrix,
    run_hmm_discovery,
)
from tactical.models.preprocessing import PreprocessingPipeline
from tactical.models.selection import ModelSelectionResult, select_gmm_k
from tactical.models.vae import VAEModel
from tactical.models.vae_discovery import (
    ComparisonRow,
    ComparisonTable,
    VAEGMMDiscoveryResult,
    run_vae_gmm_discovery,
)

__all__ = [
    "Changepoint",
    "ChangepointResult",
    "ComparisonRow",
    "ComparisonTable",
    "DatasetDiscoveryResult",
    "GMMConfig",
    "GMMModel",
    "HMMDiscoveryResult",
    "HMMModel",
    "ModelSelectionResult",
    "PreprocessingPipeline",
    "TacticalModel",
    "VAEGMMDiscoveryResult",
    "VAEModel",
    "detect_changepoints",
    "detect_changepoints_per_match",
    "plot_transition_matrix",
    "run_gmm_discovery",
    "run_hmm_discovery",
    "run_vae_gmm_discovery",
    "select_gmm_k",
]
