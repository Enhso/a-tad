"""Protocol definition for tactical state discovery models.

All model backends (GMM, HMM, VAE, etc.) must satisfy the
:class:`TacticalModel` protocol so they can be used interchangeably
in the pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


@runtime_checkable
class TacticalModel(Protocol):
    """Protocol for tactical state discovery models.

    Defines the interface that all model backends must implement.
    Models accept numpy feature arrays and produce integer state
    labels or per-state probability matrices.

    The optional ``sequence_lengths`` parameter supports models
    that require sequential structure (e.g., HMM). Non-sequential
    models (e.g., GMM) may ignore it.
    """

    def fit(
        self,
        features: np.ndarray,
        sequence_lengths: list[int] | None = None,
    ) -> None:
        """Fit the model to feature data.

        Args:
            features: Array of shape ``(n_samples, n_features)``.
            sequence_lengths: Per-match sequence lengths for models
                that require sequential structure (HMM). ``None``
                for non-sequential models (GMM).
        """
        ...

    def predict(
        self,
        features: np.ndarray,
        sequence_lengths: list[int] | None = None,
    ) -> np.ndarray:
        """Predict state labels for each sample.

        Args:
            features: Array of shape ``(n_samples, n_features)``.
            sequence_lengths: Per-match sequence lengths for
                sequential models. ``None`` for non-sequential.

        Returns:
            Array of shape ``(n_samples,)`` with integer state labels.
        """
        ...

    def predict_proba(
        self,
        features: np.ndarray,
        sequence_lengths: list[int] | None = None,
    ) -> np.ndarray:
        """Predict state probabilities for each sample.

        Args:
            features: Array of shape ``(n_samples, n_features)``.
            sequence_lengths: Per-match sequence lengths for
                sequential models. ``None`` for non-sequential.

        Returns:
            Array of shape ``(n_samples, n_states)`` where each row
            sums to 1.0.
        """
        ...

    @property
    def n_states(self) -> int:
        """Number of discovered states."""
        ...

    def save(self, path: Path) -> None:
        """Persist model to disk.

        Args:
            path: Destination file path.
        """
        ...

    @classmethod
    def load(cls, path: Path) -> TacticalModel:
        """Load model from disk.

        Args:
            path: Path to a previously saved model.

        Returns:
            The deserialized model instance.
        """
        ...
