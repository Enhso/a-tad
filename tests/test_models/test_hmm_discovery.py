"""Tests for :mod:`tactical.models.hmm_discovery`.

Uses a synthetic multi-match polars DataFrame with 3 well-separated
cyclic states across 4 match-team pairs.  Each pair has 30 windows
(120 total).  Features follow state 0 -> 1 -> 2 -> 0 in blocks of 10.

Covers:
- sequence length construction from match-team groups
- HMM fitting via ``run_hmm_discovery`` (random init)
- HMM fitting via ``run_hmm_discovery`` (GMM-initialised)
- transition matrix shape and row normalisation
- transition matrix visualisation (``plot_transition_matrix``)
- state agreement metrics (ARI, NMI) with GMM labels
- flicker rate computation
- result container field types
- HMM probability column shapes
- end-to-end pipeline roundtrip
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from tactical.config import HMMConfig
from tactical.models.gmm import GMMConfig, GMMModel
from tactical.models.hmm_discovery import (
    HMMDiscoveryResult,
    _build_sequences,
    _compute_agreement,
    _compute_flicker_rate,
    plot_transition_matrix,
    run_hmm_discovery,
)
from tactical.models.preprocessing import PreprocessingPipeline

if TYPE_CHECKING:
    from pathlib import Path

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_N_STATES = 3
_BLOCK_LEN = 10
_SEQ_LEN = _N_STATES * _BLOCK_LEN  # 30 per match-team
_N_FEATURES = 4
_MATCH_TEAMS: list[tuple[str, str]] = [
    ("m1", "teamA"),
    ("m1", "teamB"),
    ("m2", "teamA"),
    ("m2", "teamB"),
]
_TOTAL_ROWS = _SEQ_LEN * len(_MATCH_TEAMS)  # 120

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


def _make_synthetic_df(rng: np.random.Generator) -> pl.DataFrame:
    """Build a synthetic window-segment DataFrame.

    Returns:
        A polars DataFrame with metadata and t1_ feature columns.
    """
    rows: list[dict[str, object]] = []
    for match_id, team_id in _MATCH_TEAMS:
        for t in range(_SEQ_LEN):
            state = (t // _BLOCK_LEN) % _N_STATES
            feats = rng.normal(loc=_CENTERS[state], scale=0.3)
            rows.append(
                {
                    "match_id": match_id,
                    "team_id": team_id,
                    "segment_type": "window",
                    "start_time": float(t),
                    "t1_f0": feats[0],
                    "t1_f1": feats[1],
                    "t1_f2": feats[2],
                    "t1_f3": feats[3],
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture()
def synthetic_df() -> pl.DataFrame:
    """120-row DataFrame across 4 match-team pairs with 3 cyclic states."""
    rng = np.random.default_rng(seed=42)
    return _make_synthetic_df(rng)


@pytest.fixture()
def fitted_pipeline(synthetic_df: pl.DataFrame) -> PreprocessingPipeline:
    """A PreprocessingPipeline fitted on the synthetic data (no PCA)."""
    pipeline = PreprocessingPipeline(
        feature_prefix="t1_",
        null_strategy="impute_median",
        pca_variance_threshold=None,
    )
    pipeline.fit(synthetic_df)
    return pipeline


@pytest.fixture()
def fitted_gmm(
    synthetic_df: pl.DataFrame,
    fitted_pipeline: PreprocessingPipeline,
) -> GMMModel:
    """A GMMModel with K=3 fitted on the preprocessed synthetic data."""
    features = fitted_pipeline.transform(synthetic_df)
    cfg = GMMConfig(k_min=3, k_max=3, n_init=10, random_state=42)
    model = GMMModel(cfg)
    model._fit_k(features, k=_N_STATES)
    return model


@pytest.fixture()
def hmm_config() -> HMMConfig:
    """HMMConfig with 3 states for testing."""
    return HMMConfig(
        n_states=_N_STATES,
        covariance_type="full",
        n_iter=50,
        random_state=42,
        init_from_gmm=True,
    )


@pytest.fixture()
def hmm_config_no_gmm() -> HMMConfig:
    """HMMConfig with init_from_gmm disabled."""
    return HMMConfig(
        n_states=_N_STATES,
        covariance_type="full",
        n_iter=50,
        random_state=42,
        init_from_gmm=False,
    )


# ------------------------------------------------------------------
# Tests: _build_sequences
# ------------------------------------------------------------------


class TestBuildSequences:
    """Tests for the sequence-length construction helper."""

    def test_sequence_count(self, synthetic_df: pl.DataFrame) -> None:
        """One sequence per match-team pair."""
        lengths, _ = _build_sequences(synthetic_df)

        assert len(lengths) == len(_MATCH_TEAMS)

    def test_sequence_lengths_sum(self, synthetic_df: pl.DataFrame) -> None:
        """Sum of sequence lengths equals total rows."""
        lengths, df_sorted = _build_sequences(synthetic_df)

        assert sum(lengths) == df_sorted.height
        assert sum(lengths) == _TOTAL_ROWS

    def test_each_sequence_length(self, synthetic_df: pl.DataFrame) -> None:
        """Each match-team pair has SEQ_LEN rows."""
        lengths, _ = _build_sequences(synthetic_df)

        assert all(length == _SEQ_LEN for length in lengths)

    def test_sorted_order(self, synthetic_df: pl.DataFrame) -> None:
        """Returned DataFrame is sorted by match_id, team_id, start_time."""
        _, df_sorted = _build_sequences(synthetic_df)

        match_ids = df_sorted["match_id"].to_list()
        team_ids = df_sorted["team_id"].to_list()
        start_times = df_sorted["start_time"].to_list()

        combined = list(zip(match_ids, team_ids, start_times, strict=False))
        assert combined == sorted(combined)


# ------------------------------------------------------------------
# Tests: _compute_agreement
# ------------------------------------------------------------------


class TestComputeAgreement:
    """Tests for the ARI/NMI agreement helper."""

    def test_perfect_agreement(self) -> None:
        """Identical labels yield ARI=1.0 and NMI=1.0."""
        labels = np.array([0, 0, 1, 1, 2, 2])
        ari, nmi = _compute_agreement(labels, labels.copy())

        assert ari == pytest.approx(1.0)
        assert nmi == pytest.approx(1.0)

    def test_permuted_agreement(self) -> None:
        """Consistent relabelling yields high ARI and NMI."""
        a = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        b = np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])
        ari, nmi = _compute_agreement(a, b)

        assert ari == pytest.approx(1.0)
        assert nmi == pytest.approx(1.0, abs=1e-6)

    def test_random_agreement_low(self) -> None:
        """Random label assignments have low ARI."""
        rng = np.random.default_rng(seed=99)
        a = rng.integers(0, 3, size=300)
        b = rng.integers(0, 3, size=300)
        ari, nmi = _compute_agreement(a, b)

        assert ari < 0.1
        assert nmi < 0.1


# ------------------------------------------------------------------
# Tests: _compute_flicker_rate
# ------------------------------------------------------------------


class TestComputeFlickerRate:
    """Tests for the flicker rate helper."""

    def test_constant_zero_flicker(self) -> None:
        """All same state yields flicker rate 0.0."""
        df = pl.DataFrame(
            {
                "match_id": ["m1"] * 5,
                "team_id": ["t1"] * 5,
                "start_time": [0.0, 1.0, 2.0, 3.0, 4.0],
                "state": [0, 0, 0, 0, 0],
            }
        )
        rate = _compute_flicker_rate(df, "state")

        assert rate == pytest.approx(0.0)

    def test_alternating_full_flicker(self) -> None:
        """Alternating states yield flicker rate 1.0."""
        df = pl.DataFrame(
            {
                "match_id": ["m1"] * 6,
                "team_id": ["t1"] * 6,
                "start_time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "state": [0, 1, 0, 1, 0, 1],
            }
        )
        rate = _compute_flicker_rate(df, "state")

        assert rate == pytest.approx(1.0)

    def test_single_row_zero(self) -> None:
        """Single-row sequences are skipped; result is 0.0."""
        df = pl.DataFrame(
            {
                "match_id": ["m1"],
                "team_id": ["t1"],
                "start_time": [0.0],
                "state": [0],
            }
        )
        rate = _compute_flicker_rate(df, "state")

        assert rate == pytest.approx(0.0)


# ------------------------------------------------------------------
# Tests: plot_transition_matrix
# ------------------------------------------------------------------


class TestPlotTransitionMatrix:
    """Tests for the transition matrix visualisation."""

    def test_returns_figure(self) -> None:
        """Function returns a matplotlib Figure."""
        import matplotlib.figure

        mat = np.eye(3)
        fig = plot_transition_matrix(mat, title="Test")

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_saves_to_disk(self, tmp_path: Path) -> None:
        """Figure is saved when save_path is provided."""
        mat = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        save_path = tmp_path / "test_transmat.png"

        plot_transition_matrix(mat, save_path=save_path)

        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_handles_large_matrix(self) -> None:
        """Visualisation works for larger transition matrices."""
        import matplotlib.figure

        k = 10
        mat = np.full((k, k), 1.0 / k)
        fig = plot_transition_matrix(mat, title="Large")

        assert isinstance(fig, matplotlib.figure.Figure)


# ------------------------------------------------------------------
# Tests: run_hmm_discovery (random init)
# ------------------------------------------------------------------


class TestRunHMMDiscoveryRandomInit:
    """Tests for ``run_hmm_discovery`` without GMM initialisation."""

    def test_returns_correct_types(
        self,
        synthetic_df: pl.DataFrame,
        fitted_pipeline: PreprocessingPipeline,
        hmm_config_no_gmm: HMMConfig,
    ) -> None:
        """Return tuple contains correct types."""
        from tactical.models.hmm import HMMModel

        df_labeled, result, model = run_hmm_discovery(
            df_windows=synthetic_df,
            dataset_label="test_random",
            pipeline=fitted_pipeline,
            hmm_config=hmm_config_no_gmm,
        )

        assert isinstance(df_labeled, pl.DataFrame)
        assert isinstance(result, HMMDiscoveryResult)
        assert isinstance(model, HMMModel)

    def test_labels_in_range(
        self,
        synthetic_df: pl.DataFrame,
        fitted_pipeline: PreprocessingPipeline,
        hmm_config_no_gmm: HMMConfig,
    ) -> None:
        """HMM state labels are in {0, 1, ..., n_states-1}."""
        df_labeled, _, _ = run_hmm_discovery(
            df_windows=synthetic_df,
            dataset_label="test_random",
            pipeline=fitted_pipeline,
            hmm_config=hmm_config_no_gmm,
        )

        labels = df_labeled["hmm_state_label"].to_numpy()
        assert set(np.unique(labels)).issubset(set(range(_N_STATES)))

    def test_no_gmm_agreement_when_omitted(
        self,
        synthetic_df: pl.DataFrame,
        fitted_pipeline: PreprocessingPipeline,
        hmm_config_no_gmm: HMMConfig,
    ) -> None:
        """Agreement metrics are None when no GMM labels are provided."""
        _, result, _ = run_hmm_discovery(
            df_windows=synthetic_df,
            dataset_label="test_random",
            pipeline=fitted_pipeline,
            hmm_config=hmm_config_no_gmm,
        )

        assert result.gmm_agreement_ari is None
        assert result.gmm_agreement_nmi is None


# ------------------------------------------------------------------
# Tests: run_hmm_discovery (GMM init + agreement)
# ------------------------------------------------------------------


class TestRunHMMDiscoveryGMMInit:
    """Tests for ``run_hmm_discovery`` with GMM initialisation."""

    def test_gmm_init_converges(
        self,
        synthetic_df: pl.DataFrame,
        fitted_pipeline: PreprocessingPipeline,
        fitted_gmm: GMMModel,
        hmm_config: HMMConfig,
    ) -> None:
        """GMM-initialised HMM fitting does not raise."""
        features = fitted_pipeline.transform(synthetic_df)
        gmm_labels = fitted_gmm.predict(features)

        df_labeled, result, model = run_hmm_discovery(
            df_windows=synthetic_df,
            dataset_label="test_gmm_init",
            pipeline=fitted_pipeline,
            hmm_config=hmm_config,
            gmm_model=fitted_gmm,
            gmm_labels=gmm_labels,
        )

        assert df_labeled.height == _TOTAL_ROWS
        assert model.n_states == _N_STATES

    def test_gmm_agreement_computed(
        self,
        synthetic_df: pl.DataFrame,
        fitted_pipeline: PreprocessingPipeline,
        fitted_gmm: GMMModel,
        hmm_config: HMMConfig,
    ) -> None:
        """ARI and NMI are computed when GMM labels are provided."""
        features = fitted_pipeline.transform(synthetic_df)
        gmm_labels = fitted_gmm.predict(features)

        _, result, _ = run_hmm_discovery(
            df_windows=synthetic_df,
            dataset_label="test_agreement",
            pipeline=fitted_pipeline,
            hmm_config=hmm_config,
            gmm_model=fitted_gmm,
            gmm_labels=gmm_labels,
        )

        assert result.gmm_agreement_ari is not None
        assert result.gmm_agreement_nmi is not None
        # With well-separated clusters, agreement should be substantial
        assert result.gmm_agreement_ari > 0.3
        assert result.gmm_agreement_nmi > 0.3

    def test_transition_matrix_shape(
        self,
        synthetic_df: pl.DataFrame,
        fitted_pipeline: PreprocessingPipeline,
        fitted_gmm: GMMModel,
        hmm_config: HMMConfig,
    ) -> None:
        """Transition matrix is (n_states, n_states)."""
        _, result, _ = run_hmm_discovery(
            df_windows=synthetic_df,
            dataset_label="test_transmat",
            pipeline=fitted_pipeline,
            hmm_config=hmm_config,
            gmm_model=fitted_gmm,
        )

        assert result.transition_matrix.shape == (_N_STATES, _N_STATES)

    def test_transition_matrix_rows_sum_to_one(
        self,
        synthetic_df: pl.DataFrame,
        fitted_pipeline: PreprocessingPipeline,
        fitted_gmm: GMMModel,
        hmm_config: HMMConfig,
    ) -> None:
        """Each row of the transition matrix sums to ~1.0."""
        _, result, _ = run_hmm_discovery(
            df_windows=synthetic_df,
            dataset_label="test_transmat_sum",
            pipeline=fitted_pipeline,
            hmm_config=hmm_config,
            gmm_model=fitted_gmm,
        )

        np.testing.assert_allclose(
            result.transition_matrix.sum(axis=1),
            1.0,
            atol=1e-6,
        )

    def test_probability_columns_present(
        self,
        synthetic_df: pl.DataFrame,
        fitted_pipeline: PreprocessingPipeline,
        fitted_gmm: GMMModel,
        hmm_config: HMMConfig,
    ) -> None:
        """Per-state probability columns are added to the output."""
        df_labeled, _, _ = run_hmm_discovery(
            df_windows=synthetic_df,
            dataset_label="test_prob_cols",
            pipeline=fitted_pipeline,
            hmm_config=hmm_config,
            gmm_model=fitted_gmm,
        )

        for s in range(_N_STATES):
            col_name = f"hmm_state_prob_{s}"
            assert col_name in df_labeled.columns

    def test_probability_columns_sum_to_one(
        self,
        synthetic_df: pl.DataFrame,
        fitted_pipeline: PreprocessingPipeline,
        fitted_gmm: GMMModel,
        hmm_config: HMMConfig,
    ) -> None:
        """Per-state probability columns sum to ~1.0 per row."""
        df_labeled, _, _ = run_hmm_discovery(
            df_windows=synthetic_df,
            dataset_label="test_prob_sum",
            pipeline=fitted_pipeline,
            hmm_config=hmm_config,
            gmm_model=fitted_gmm,
        )

        prob_cols = [f"hmm_state_prob_{s}" for s in range(_N_STATES)]
        proba = df_labeled.select(prob_cols).to_numpy()

        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_result_fields_populated(
        self,
        synthetic_df: pl.DataFrame,
        fitted_pipeline: PreprocessingPipeline,
        fitted_gmm: GMMModel,
        hmm_config: HMMConfig,
    ) -> None:
        """All result container fields have valid values."""
        features = fitted_pipeline.transform(synthetic_df)
        gmm_labels = fitted_gmm.predict(features)

        _, result, _ = run_hmm_discovery(
            df_windows=synthetic_df,
            dataset_label="test_fields",
            pipeline=fitted_pipeline,
            hmm_config=hmm_config,
            gmm_model=fitted_gmm,
            gmm_labels=gmm_labels,
        )

        assert result.dataset_label == "test_fields"
        assert result.n_states == _N_STATES
        assert result.n_sequences == len(_MATCH_TEAMS)
        assert result.n_windows == _TOTAL_ROWS
        assert result.n_windows_retained == _TOTAL_ROWS
        assert 0.0 <= result.flicker_rate <= 1.0

    def test_flicker_rate_reasonable(
        self,
        synthetic_df: pl.DataFrame,
        fitted_pipeline: PreprocessingPipeline,
        fitted_gmm: GMMModel,
        hmm_config: HMMConfig,
    ) -> None:
        """HMM flicker rate is lower than random (cyclic data is smooth)."""
        _, result, _ = run_hmm_discovery(
            df_windows=synthetic_df,
            dataset_label="test_flicker",
            pipeline=fitted_pipeline,
            hmm_config=hmm_config,
            gmm_model=fitted_gmm,
        )

        # With block-cyclic data the HMM should not flicker excessively
        assert result.flicker_rate < 0.5

    def test_labeled_df_row_count(
        self,
        synthetic_df: pl.DataFrame,
        fitted_pipeline: PreprocessingPipeline,
        fitted_gmm: GMMModel,
        hmm_config: HMMConfig,
    ) -> None:
        """Output DataFrame has same row count as retained input."""
        df_labeled, result, _ = run_hmm_discovery(
            df_windows=synthetic_df,
            dataset_label="test_rows",
            pipeline=fitted_pipeline,
            hmm_config=hmm_config,
            gmm_model=fitted_gmm,
        )

        assert df_labeled.height == result.n_windows_retained

    def test_visualisation_integration(
        self,
        synthetic_df: pl.DataFrame,
        fitted_pipeline: PreprocessingPipeline,
        fitted_gmm: GMMModel,
        hmm_config: HMMConfig,
        tmp_path: Path,
    ) -> None:
        """Transition matrix from discovery can be visualised and saved."""
        _, result, _ = run_hmm_discovery(
            df_windows=synthetic_df,
            dataset_label="test_viz",
            pipeline=fitted_pipeline,
            hmm_config=hmm_config,
            gmm_model=fitted_gmm,
        )

        save_path = tmp_path / "transmat.png"
        fig = plot_transition_matrix(
            result.transition_matrix,
            title="Integration Test",
            save_path=save_path,
        )

        import matplotlib.figure

        assert isinstance(fig, matplotlib.figure.Figure)
        assert save_path.exists()
        assert save_path.stat().st_size > 0
