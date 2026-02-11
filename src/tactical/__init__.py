"""Tactical State Discovery Engine.

A modular analytics engine for discovering latent tactical states
in football match data.
"""

from tactical.config import PipelineConfig
from tactical.exceptions import TacticalError

__version__ = "0.1.0"

__all__ = ["PipelineConfig", "TacticalError", "__version__"]
