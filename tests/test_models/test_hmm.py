"""Tests for :mod:`tactical.models.hmm`.

Uses synthetic sequential data with 3 states and 5 sequences of
length 50 each (250 total samples). State transitions follow a
cyclic pattern: state 0 -> 1 -> 2 -> 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tactical.config import HMMConfig
from tactical.exceptions import ModelFitError
from tactical.models.gmm import GMMConfig, GMMModel
from tactical.models.hmm import HMMModel

if TYPE_CHECKING:
    from pathlib import Path

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_N_STATES = 3
_N_SEQUENCES = 5
_SEQ_LEN = 50
_N_FEATURES = 4
_TOTAL_SAMPLES = _N_SEQUENCES * _SEQ_LEN

# Well-separated cluster centres so models converge reliably
_CENTERS = np.array(
    [
        [0.0, 0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0, 10.0],
        [-10.0, 10.0, -10.0, 10.0],
    ],
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _generate_cyclic_sequences(
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[int]]:
    """Generate synthetic sequential data with cyclic state transitions.

    Each sequence cycles through states 0 -> 1 -> 2 -> 0 in blocks
    of roughly equal length within the sequence.

    Returns:
        Tuple of (features array, sequence_lengths list).
    """
    all_features: list[np.ndarray] = []
    lengths: list[int] = []

    for _ in range(_N_SEQUENCES):
        seq_features = np.empty((_SEQ_LEN, _N_FEATURES))
        block = _SEQ_LEN // _N_STATES
        for t in range(_SEQ_LEN):
            state = (t // block) % _N_STATES
            seq_features[t] = rng.normal(
                loc=_CENTERS[state],
                scale=0.5,
            )
        all_features.append(seq_features)
        lengths.append(_SEQ_LEN)

    return np.vstack(all_features), lengths


@pytest.fixture()
def sequential_data() -> tuple[np.ndarray, list[int]]:
    """250 samples across 5 sequences with 3 cyclic states."""
    rng = np.random.default_rng(seed=42)
    return _generate_cyclic_sequences(rng)


@pytest.fixture()
def hmm_config() -> HMMConfig:
    """HMMConfig with 3 states for testing."""
    return HMMConfig(
        n_states=_N_STATES,
        covariance_type="full",
        n_iter=50,
        random_state=42,
    )


@pytest.fixture()
def fitted_hmm(
    sequential_data: tuple[np.ndarray, list[int]],
    hmm_config: HMMConfig,
) -> HMMModel:
    """An HMMModel fitted on the synthetic sequential data."""
    features, lengths = sequential_data
    model = HMMModel(hmm_config)
    model.fit(features, sequence_lengths=lengths)
    return model


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestHMMModel:
    """Unit tests for :class:`HMMModel`."""

    def test_hmm_fit_predict(
        self,
        sequential_data: tuple[np.ndarray, list[int]],
        fitted_hmm: HMMModel,
    ) -> None:
        """Fit with sequence_lengths, predict returns labels in {0, 1, 2}."""
        features, lengths = sequential_data
        labels = fitted_hmm.predict(features, sequence_lengths=lengths)

        assert labels.shape == (features.shape[0],)
        assert set(np.unique(labels)).issubset({0, 1, 2})

    def test_hmm_requires_sequence_lengths(
        self,
        sequential_data: tuple[np.ndarray, list[int]],
        hmm_config: HMMConfig,
    ) -> None:
        """Fit without sequence_lengths raises ValueError."""
        features, _ = sequential_data
        model = HMMModel(hmm_config)

        with pytest.raises(ValueError, match="sequence_lengths is required"):
            model.fit(features, sequence_lengths=None)

    def test_hmm_sequence_lengths_validation(
        self,
        sequential_data: tuple[np.ndarray, list[int]],
        hmm_config: HMMConfig,
    ) -> None:
        """Sum of sequence_lengths != len(features) raises ValueError."""
        features, _ = sequential_data
        model = HMMModel(hmm_config)
        wrong_lengths = [100, 100]  # sums to 200, not 250

        with pytest.raises(ValueError, match="does not equal"):
            model.fit(features, sequence_lengths=wrong_lengths)

    def test_hmm_transition_matrix_rows_sum_to_one(
        self,
        fitted_hmm: HMMModel,
    ) -> None:
        """Each row of the transition matrix sums to ~1.0."""
        transmat = fitted_hmm.transition_matrix

        np.testing.assert_allclose(
            transmat.sum(axis=1),
            1.0,
            atol=1e-6,
        )

    def test_hmm_transition_matrix_shape(
        self,
        fitted_hmm: HMMModel,
    ) -> None:
        """Transition matrix shape is (n_states, n_states)."""
        transmat = fitted_hmm.transition_matrix

        assert transmat.shape == (_N_STATES, _N_STATES)

    def test_hmm_init_from_gmm(
        self,
        sequential_data: tuple[np.ndarray, list[int]],
        hmm_config: HMMConfig,
    ) -> None:
        """fit_from_gmm converges without ModelFitError."""
        features, lengths = sequential_data

        gmm_cfg = GMMConfig(
            k_min=_N_STATES,
            k_max=_N_STATES,
            covariance_type="full",
            n_init=10,
            random_state=42,
        )
        gmm = GMMModel(gmm_cfg)
        gmm._fit_k(features, k=_N_STATES)

        hmm = HMMModel(hmm_config)
        try:
            hmm.fit_from_gmm(features, lengths, gmm)
        except ModelFitError:
            pytest.fail("fit_from_gmm raised ModelFitError unexpectedly")

        # Model should be usable after GMM-init fitting
        labels = hmm.predict(features, sequence_lengths=lengths)
        assert labels.shape == (features.shape[0],)

    def test_hmm_predict_proba_shape(
        self,
        sequential_data: tuple[np.ndarray, list[int]],
        fitted_hmm: HMMModel,
    ) -> None:
        """predict_proba shape is (n_samples, n_states)."""
        features, lengths = sequential_data
        proba = fitted_hmm.predict_proba(features, sequence_lengths=lengths)

        assert proba.shape == (_TOTAL_SAMPLES, _N_STATES)

    def test_hmm_save_load(
        self,
        sequential_data: tuple[np.ndarray, list[int]],
        fitted_hmm: HMMModel,
        tmp_path: Path,
    ) -> None:
        """Roundtrip save/load produces identical predictions."""
        features, lengths = sequential_data
        expected = fitted_hmm.predict(features, sequence_lengths=lengths)

        save_path = tmp_path / "hmm_model.pkl"
        fitted_hmm.save(save_path)

        loaded = HMMModel.load(save_path)
        result = loaded.predict(features, sequence_lengths=lengths)

        np.testing.assert_array_equal(result, expected)
        assert loaded.n_states == fitted_hmm.n_states

    def test_hmm_state_means_shape(
        self,
        fitted_hmm: HMMModel,
    ) -> None:
        """state_means shape is (n_states, n_features)."""
        means = fitted_hmm.state_means

        assert means.shape == (_N_STATES, _N_FEATURES)
