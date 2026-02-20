"""Hidden Markov Model backend for tactical state discovery.

Wraps :class:`hmmlearn.hmm.GaussianHMM` with support for GMM-based
initialization, per-match sequence handling, and transition matrix
extraction.
"""

from __future__ import annotations

import logging
import pickle
from typing import TYPE_CHECKING

import numpy as np
from hmmlearn.hmm import GaussianHMM  # type: ignore[import-untyped]

from tactical.exceptions import ModelFitError

if TYPE_CHECKING:
    from pathlib import Path

    from tactical.config import HMMConfig
    from tactical.models.gmm import GMMModel

logger = logging.getLogger(__name__)


class HMMModel:
    """Hidden Markov Model for temporal tactical state discovery.

    Wraps hmmlearn GaussianHMM with support for GMM-based
    initialization, per-match sequence handling, and transition
    matrix extraction.

    Satisfies the :class:`~tactical.models.base.TacticalModel`
    protocol.

    Args:
        config: HMM configuration dataclass.
    """

    def __init__(self, config: HMMConfig) -> None:
        self._config = config
        self._model: GaussianHMM | None = None

    # ------------------------------------------------------------------
    # TacticalModel interface
    # ------------------------------------------------------------------

    def fit(
        self,
        features: np.ndarray,
        sequence_lengths: list[int] | None = None,
    ) -> None:
        """Fit HMM to feature sequences.

        Args:
            features: Concatenated feature arrays from all matches,
                shape ``(total_samples, n_features)``.
            sequence_lengths: Length of each match's sequence.
                Required. Sum must equal ``total_samples``.

        Raises:
            ModelFitError: If fitting fails (convergence, numerical).
            ValueError: If ``sequence_lengths`` is ``None`` or doesn't
                sum to ``len(features)``.
        """
        self._validate_sequence_lengths(features, sequence_lengths)
        assert sequence_lengths is not None  # guarded above
        model = self._build_hmm()
        try:
            model.fit(features, lengths=sequence_lengths)
        except Exception as exc:
            msg = f"HMM fitting failed: {exc}"
            raise ModelFitError(msg) from exc
        self._model = model

    def fit_from_gmm(
        self,
        features: np.ndarray,
        sequence_lengths: list[int],
        gmm_model: GMMModel,
    ) -> None:
        """Fit HMM initialized from GMM cluster parameters.

        Initializes HMM means and covariances from the GMM's
        fitted parameters, then runs EM to learn transition
        dynamics.

        Args:
            features: Concatenated feature arrays from all matches,
                shape ``(total_samples, n_features)``.
            sequence_lengths: Length of each match's sequence.
            gmm_model: A fitted :class:`GMMModel` whose parameters
                seed the HMM.

        Raises:
            ModelFitError: If fitting fails.
            ValueError: If ``sequence_lengths`` doesn't sum to
                ``len(features)``.
        """
        self._validate_sequence_lengths(features, sequence_lengths)
        sklearn_gmm = gmm_model.sklearn_model
        if sklearn_gmm is None:
            msg = "GMMModel has not been fitted."
            raise ModelFitError(msg)

        k = self._config.n_states
        model = self._build_hmm(init_params="", params="stmc")

        # Seed from GMM parameters
        model.means_ = sklearn_gmm.means_[:k].copy()
        model.covars_ = sklearn_gmm.covariances_[:k].copy()
        model.startprob_ = np.full(k, 1.0 / k)
        model.transmat_ = np.full((k, k), 1.0 / k)

        try:
            model.fit(features, lengths=sequence_lengths)
        except Exception as exc:
            msg = f"HMM fitting (GMM-init) failed: {exc}"
            raise ModelFitError(msg) from exc
        self._model = model

    def predict(
        self,
        features: np.ndarray,
        sequence_lengths: list[int] | None = None,
    ) -> np.ndarray:
        """Predict most likely state sequence using Viterbi.

        Args:
            features: Array of shape ``(n_samples, n_features)``.
            sequence_lengths: Per-match sequence lengths.

        Returns:
            Array of shape ``(n_samples,)`` with integer state labels.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        self._check_fitted()
        assert self._model is not None
        lengths = sequence_lengths if sequence_lengths is not None else [len(features)]
        labels: np.ndarray = np.asarray(
            self._model.predict(features, lengths=lengths),
        )
        return labels

    def predict_proba(
        self,
        features: np.ndarray,
        sequence_lengths: list[int] | None = None,
    ) -> np.ndarray:
        """Predict state posterior probabilities.

        Args:
            features: Array of shape ``(n_samples, n_features)``.
            sequence_lengths: Per-match sequence lengths.

        Returns:
            Array of shape ``(n_samples, n_states)`` where each
            row sums to 1.0.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        self._check_fitted()
        assert self._model is not None
        lengths = sequence_lengths if sequence_lengths is not None else [len(features)]
        proba: np.ndarray = np.asarray(
            self._model.predict_proba(features, lengths=lengths),
        )
        return proba

    @property
    def n_states(self) -> int:
        """Number of hidden states.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        self._check_fitted()
        return self._config.n_states

    @property
    def transition_matrix(self) -> np.ndarray:
        """Return the state transition matrix (K x K).

        Raises:
            RuntimeError: If model has not been fitted.
        """
        self._check_fitted()
        assert self._model is not None
        mat: np.ndarray = np.asarray(self._model.transmat_)
        return mat

    @property
    def state_means(self) -> np.ndarray:
        """Return the mean feature vector for each state.

        Returns:
            Array of shape ``(n_states, n_features)``.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        self._check_fitted()
        assert self._model is not None
        means: np.ndarray = np.asarray(self._model.means_)
        return means

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
    def load(cls, path: Path) -> HMMModel:
        """Load a fitted model from disk.

        Args:
            path: Path to a previously saved model.

        Returns:
            The deserialized :class:`HMMModel`.

        Raises:
            FileNotFoundError: If *path* does not exist.
            TypeError: If the loaded object is not an
                :class:`HMMModel`.
        """
        with path.open("rb") as f:
            obj = pickle.load(f)  # noqa: S301
        if not isinstance(obj, cls):
            msg = f"Expected HMMModel, got {type(obj).__name__}."
            raise TypeError(msg)
        return obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_hmm(
        self, init_params: str = "stmc", params: str = "stmc"
    ) -> GaussianHMM:
        """Construct a fresh hmmlearn GaussianHMM instance.

        Args:
            init_params: Which parameters to initialise randomly.
            params: Which parameters to update during EM.

        Returns:
            An unfitted :class:`GaussianHMM`.
        """
        return GaussianHMM(
            n_components=self._config.n_states,
            covariance_type=self._config.covariance_type,
            n_iter=self._config.n_iter,
            random_state=self._config.random_state,
            init_params=init_params,
            params=params,
        )

    def _check_fitted(self) -> None:
        """Raise :class:`RuntimeError` if the model is not fitted."""
        if self._model is None:
            msg = "Model has not been fitted. Call fit() before predict/save."
            raise RuntimeError(msg)

    @staticmethod
    def _validate_sequence_lengths(
        features: np.ndarray,
        sequence_lengths: list[int] | None,
    ) -> None:
        """Validate that sequence_lengths is provided and consistent.

        Args:
            features: Feature array.
            sequence_lengths: Per-match sequence lengths.

        Raises:
            ValueError: If ``sequence_lengths`` is ``None`` or its
                sum does not equal ``len(features)``.
        """
        if sequence_lengths is None:
            msg = (
                "sequence_lengths is required for HMM fitting. "
                "Provide the length of each match sequence."
            )
            raise ValueError(msg)
        total = sum(sequence_lengths)
        if total != len(features):
            msg = (
                f"Sum of sequence_lengths ({total}) does not equal "
                f"number of samples ({len(features)})."
            )
            raise ValueError(msg)
