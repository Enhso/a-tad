"""Gaussian Mixture Model backend for tactical state discovery.

Wraps :class:`sklearn.mixture.GaussianMixture` (and optionally
:class:`sklearn.mixture.BayesianGaussianMixture`) with additional
functionality for temporal smoothing and persistence.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.mixture import (  # type: ignore[import-untyped]
    BayesianGaussianMixture,
    GaussianMixture,
)

from tactical.exceptions import ModelFitError

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GMMConfig:
    """Configuration for Gaussian Mixture Model.

    Attributes:
        k_min: Minimum number of clusters for model selection.
        k_max: Maximum number of clusters for model selection.
        covariance_type: Covariance parameterisation passed to
            sklearn. One of ``"full"``, ``"tied"``, ``"diag"``,
            or ``"spherical"``.
        n_init: Number of random restarts per fit.
        random_state: Seed for reproducibility.
        use_bayesian: If ``True``, use Dirichlet Process GMM
            (:class:`BayesianGaussianMixture`) instead of vanilla
            :class:`GaussianMixture`.
    """

    k_min: int = 3
    k_max: int = 12
    covariance_type: str = "full"
    n_init: int = 20
    random_state: int = 42
    use_bayesian: bool = False


class GMMModel:
    """Gaussian Mixture Model for tactical state discovery.

    Wraps sklearn ``GaussianMixture`` with additional functionality
    for model selection, temporal smoothing, and persistence.
    Satisfies the :class:`~tactical.models.base.TacticalModel`
    protocol.

    Args:
        config: GMM configuration dataclass.
    """

    def __init__(self, config: GMMConfig) -> None:
        self._config = config
        self._model: GaussianMixture | BayesianGaussianMixture | None = None
        self._n_states: int = 0

    # ------------------------------------------------------------------
    # TacticalModel interface
    # ------------------------------------------------------------------

    def fit(
        self,
        features: np.ndarray,
        sequence_lengths: list[int] | None = None,
    ) -> None:
        """Fit GMM to features.

        ``sequence_lengths`` is accepted for protocol compatibility
        but ignored because GMM does not model sequential structure.

        Args:
            features: Array of shape ``(n_samples, n_features)``.
            sequence_lengths: Ignored.

        Raises:
            ModelFitError: If sklearn fitting fails.
        """
        n_components = self._config.k_max
        try:
            self._model = self._build_mixture(n_components)
            self._model.fit(features)
            self._n_states = n_components
        except Exception as exc:
            msg = f"GMM fitting failed with k={n_components}: {exc}"
            raise ModelFitError(msg) from exc

    def predict(
        self,
        features: np.ndarray,
        sequence_lengths: list[int] | None = None,
    ) -> np.ndarray:
        """Predict state labels for each sample.

        Args:
            features: Array of shape ``(n_samples, n_features)``.
            sequence_lengths: Ignored.

        Returns:
            Array of shape ``(n_samples,)`` with integer labels.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        self._check_fitted()
        assert self._model is not None  # guarded by _check_fitted
        labels: np.ndarray = np.asarray(self._model.predict(features))
        return labels

    def predict_proba(
        self,
        features: np.ndarray,
        sequence_lengths: list[int] | None = None,
    ) -> np.ndarray:
        """Predict state probabilities for each sample.

        Args:
            features: Array of shape ``(n_samples, n_features)``.
            sequence_lengths: Ignored.

        Returns:
            Array of shape ``(n_samples, n_states)`` where each
            row sums to 1.0.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        self._check_fitted()
        assert self._model is not None
        proba: np.ndarray = np.asarray(
            self._model.predict_proba(features),
        )
        return proba

    @property
    def n_states(self) -> int:
        """Number of discovered states.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        self._check_fitted()
        return self._n_states

    def save(self, path: Path) -> None:
        """Persist fitted model to disk using pickle.

        Args:
            path: Destination file path.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        self._check_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> GMMModel:
        """Load a fitted model from disk.

        Args:
            path: Path to a previously saved model.

        Returns:
            The deserialized :class:`GMMModel`.

        Raises:
            FileNotFoundError: If *path* does not exist.
            TypeError: If the loaded object is not a
                :class:`GMMModel`.
        """
        with path.open("rb") as f:
            obj = pickle.load(f)  # noqa: S301
        if not isinstance(obj, cls):
            msg = f"Expected GMMModel, got {type(obj).__name__}."
            raise TypeError(msg)
        return obj

    # ------------------------------------------------------------------
    # Extra utilities
    # ------------------------------------------------------------------

    @staticmethod
    def smooth_states(
        states: np.ndarray,
        window_size: int = 5,
    ) -> np.ndarray:
        """Apply moving-mode temporal smoothing to a state sequence.

        For each position, the mode of the surrounding window is
        taken. This reduces flickering in state assignments.

        Args:
            states: 1-D integer array of state labels.
            window_size: Size of the smoothing window (must be odd
                and >= 1).

        Returns:
            Smoothed 1-D integer array with the same length.
        """
        n = len(states)
        if n == 0 or window_size <= 1:
            return states.copy()

        # Ensure odd window for symmetric padding
        if window_size % 2 == 0:
            window_size += 1

        half = window_size // 2
        smoothed = np.empty_like(states)

        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            window = states[lo:hi]
            # Fast mode via bincount (labels are non-negative ints)
            counts = np.bincount(window)
            smoothed[i] = counts.argmax()

        return smoothed

    @property
    def config(self) -> GMMConfig:
        """Return the model configuration."""
        return self._config

    @property
    def sklearn_model(
        self,
    ) -> GaussianMixture | BayesianGaussianMixture | None:
        """Return the underlying sklearn mixture model, or ``None``."""
        return self._model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_mixture(
        self,
        n_components: int,
    ) -> GaussianMixture | BayesianGaussianMixture:
        """Construct a fresh sklearn mixture model.

        Args:
            n_components: Number of Gaussian components.

        Returns:
            An unfitted sklearn mixture instance.
        """
        cls_: type[GaussianMixture] | type[BayesianGaussianMixture] = (
            BayesianGaussianMixture if self._config.use_bayesian else GaussianMixture
        )
        return cls_(
            n_components=n_components,
            covariance_type=self._config.covariance_type,
            n_init=self._config.n_init,
            random_state=self._config.random_state,
        )

    def _check_fitted(self) -> None:
        """Raise :class:`RuntimeError` if the model is not fitted."""
        if self._model is None:
            msg = "Model has not been fitted. Call fit() before predict/save."
            raise RuntimeError(msg)

    def _fit_k(self, features: np.ndarray, k: int) -> None:
        """Fit the model with a specific number of components.

        Intended for use by the model selection module.

        Args:
            features: Array of shape ``(n_samples, n_features)``.
            k: Number of components.

        Raises:
            ModelFitError: If sklearn fitting fails.
        """
        try:
            self._model = self._build_mixture(k)
            self._model.fit(features)
            self._n_states = k
        except Exception as exc:
            msg = f"GMM fitting failed with k={k}: {exc}"
            raise ModelFitError(msg) from exc
