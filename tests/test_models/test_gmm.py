"""Tests for :mod:`tactical.models.gmm` and :mod:`tactical.models.selection`.

Uses synthetic Gaussian blob data with 3 well-separated clusters
(100 samples each) to exercise GMM fitting, prediction, temporal
smoothing, model selection, and persistence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tactical.models.gmm import GMMConfig, GMMModel
from tactical.models.selection import ModelSelectionResult, select_gmm_k

if TYPE_CHECKING:
    from pathlib import Path

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

_RNG = np.random.default_rng(seed=42)
_N_PER_CLUSTER = 100
_N_CLUSTERS = 3
_CENTERS = np.array([[0.0, 0.0], [8.0, 0.0], [4.0, 8.0]])


@pytest.fixture()
def blob_features() -> np.ndarray:
    """300 samples from 3 well-separated 2-D Gaussian blobs."""
    rng = np.random.default_rng(seed=42)
    parts: list[np.ndarray] = []
    for center in _CENTERS:
        pts = rng.normal(loc=center, scale=0.5, size=(_N_PER_CLUSTER, 2))
        parts.append(pts)
    return np.vstack(parts)


@pytest.fixture()
def fitted_model(blob_features: np.ndarray) -> GMMModel:
    """A GMMModel fitted on blob data with K=3."""
    cfg = GMMConfig(k_min=3, k_max=3, n_init=10, random_state=42)
    model = GMMModel(cfg)
    model._fit_k(blob_features, k=3)
    return model


# ------------------------------------------------------------------
# Tests: GMMModel core interface
# ------------------------------------------------------------------


class TestGMMModel:
    """Unit tests for :class:`GMMModel`."""

    def test_gmm_fit_predict(
        self,
        blob_features: np.ndarray,
        fitted_model: GMMModel,
    ) -> None:
        """Fit with K=3, predict returns labels in {0, 1, 2}."""
        labels = fitted_model.predict(blob_features)

        assert labels.shape == (blob_features.shape[0],)
        assert set(np.unique(labels)) == {0, 1, 2}

    def test_gmm_predict_proba_sums_to_one(
        self,
        blob_features: np.ndarray,
        fitted_model: GMMModel,
    ) -> None:
        """Each row of predict_proba sums to ~1.0."""
        proba = fitted_model.predict_proba(blob_features)

        assert proba.shape == (blob_features.shape[0], 3)
        np.testing.assert_allclose(
            proba.sum(axis=1),
            1.0,
            atol=1e-6,
        )

    def test_gmm_n_states(self, fitted_model: GMMModel) -> None:
        """``n_states`` returns the correct K."""
        assert fitted_model.n_states == 3

    def test_gmm_save_load(
        self,
        blob_features: np.ndarray,
        fitted_model: GMMModel,
        tmp_path: Path,
    ) -> None:
        """Roundtrip save/load produces identical predictions."""
        expected = fitted_model.predict(blob_features)

        save_path = tmp_path / "gmm_model.pkl"
        fitted_model.save(save_path)

        loaded = GMMModel.load(save_path)
        result = loaded.predict(blob_features)

        np.testing.assert_array_equal(result, expected)
        assert loaded.n_states == fitted_model.n_states

    def test_transform_before_fit_raises(
        self,
        blob_features: np.ndarray,
    ) -> None:
        """Calling predict on an unfitted model raises RuntimeError."""
        cfg = GMMConfig()
        model = GMMModel(cfg)

        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(blob_features)

    def test_predict_proba_before_fit_raises(
        self,
        blob_features: np.ndarray,
    ) -> None:
        """Calling predict_proba on an unfitted model raises RuntimeError."""
        cfg = GMMConfig()
        model = GMMModel(cfg)

        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_proba(blob_features)


# ------------------------------------------------------------------
# Tests: temporal smoothing
# ------------------------------------------------------------------


class TestSmoothStates:
    """Unit tests for :meth:`GMMModel.smooth_states`."""

    def test_gmm_smooth_states(self) -> None:
        """Sequence [0,1,0,0,0] with window=3 smooths to [0,0,0,0,0]."""
        states = np.array([0, 1, 0, 0, 0])
        result = GMMModel.smooth_states(states, window_size=3)

        np.testing.assert_array_equal(result, np.array([0, 0, 0, 0, 0]))

    def test_smooth_preserves_length(self) -> None:
        """Output has the same length as input."""
        states = np.array([0, 1, 2, 1, 0, 2, 1])
        result = GMMModel.smooth_states(states, window_size=5)

        assert result.shape == states.shape

    def test_smooth_noop_window_1(self) -> None:
        """Window size 1 returns an identical copy."""
        states = np.array([2, 0, 1, 0, 2])
        result = GMMModel.smooth_states(states, window_size=1)

        np.testing.assert_array_equal(result, states)

    def test_smooth_empty(self) -> None:
        """Empty input returns an empty array."""
        states = np.array([], dtype=int)
        result = GMMModel.smooth_states(states, window_size=3)

        assert result.shape == (0,)

    def test_smooth_even_window_rounds_up(self) -> None:
        """Even window sizes are bumped to the next odd number."""
        states = np.array([0, 1, 0, 0, 0])
        # window_size=4 becomes 5 internally
        result = GMMModel.smooth_states(states, window_size=4)

        assert result.shape == states.shape
        # With window=5 the mode around index 1 is 0
        assert result[1] == 0


# ------------------------------------------------------------------
# Tests: model selection
# ------------------------------------------------------------------


class TestModelSelection:
    """Unit tests for :func:`select_gmm_k`."""

    def test_model_selection_finds_k(
        self,
        blob_features: np.ndarray,
    ) -> None:
        """BIC-selected K is 3 (or close) on 3-cluster data."""
        cfg = GMMConfig(
            k_min=2,
            k_max=6,
            n_init=10,
            random_state=42,
        )
        result = select_gmm_k(blob_features, cfg)

        # With well-separated blobs, BIC should find K=3
        assert result.best_k_bic == 3

    def test_model_selection_result_fields(
        self,
        blob_features: np.ndarray,
    ) -> None:
        """All result fields are populated with correct lengths."""
        cfg = GMMConfig(
            k_min=2,
            k_max=5,
            n_init=10,
            random_state=42,
        )
        result = select_gmm_k(blob_features, cfg)

        assert isinstance(result, ModelSelectionResult)

        n_k = 5 - 2 + 1  # k_max - k_min + 1
        assert len(result.k_values) == n_k
        assert len(result.bic_scores) == n_k
        assert len(result.aic_scores) == n_k
        assert len(result.silhouette_scores) == n_k

        assert result.k_values == (2, 3, 4, 5)

        assert result.best_k_bic in result.k_values
        assert result.best_k_silhouette in result.k_values

        # All BIC/AIC scores should be finite
        assert all(np.isfinite(s) for s in result.bic_scores)
        assert all(np.isfinite(s) for s in result.aic_scores)

        # Silhouette scores should be in [-1, 1]
        assert all(-1.0 <= s <= 1.0 for s in result.silhouette_scores)
