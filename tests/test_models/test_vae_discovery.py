"""Tests for :mod:`tactical.models.vae_discovery`.

Uses a synthetic multi-match polars DataFrame with 3 well-separated
Gaussian blobs across 4 match-team pairs (120 total rows, 4 features).
A direct GMM and preprocessing pipeline are fitted first; synthetic
latent codes are generated to simulate pre-computed VAE output. Then
:func:`run_vae_gmm_discovery` is exercised end-to-end.

Covers:
- result container field types and sane values
- comparison table structure and polars rendering
- state label assignment to all retained rows
- probability columns shape and normalisation
- silhouette scores in valid range
- BIC values are finite
- agreement metrics (ARI, NMI) in valid range
- helper functions (_safe_silhouette, _compute_agreement)
- latent code row-count mismatch error
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from tactical.models.gmm import GMMConfig, GMMModel
from tactical.models.preprocessing import PreprocessingPipeline
from tactical.models.vae_discovery import (
    ComparisonRow,
    ComparisonTable,
    VAEGMMDiscoveryResult,
    _compute_agreement,
    _safe_silhouette,
    run_vae_gmm_discovery,
)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_N_STATES = 3
_BLOCK_LEN = 10
_SEQ_LEN = _N_STATES * _BLOCK_LEN  # 30 per match-team
_N_FEATURES = 4
_LATENT_DIM = 3
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

_LATENT_CENTERS = np.array(
    [
        [0.0, 0.0, 0.0],
        [8.0, 0.0, 0.0],
        [0.0, 8.0, 0.0],
    ],
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _make_synthetic_df(rng: np.random.Generator) -> pl.DataFrame:
    """Build a synthetic window-segment DataFrame with 3 cyclic states.

    Args:
        rng: Numpy random generator.

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


def _make_synthetic_latent_codes(rng: np.random.Generator) -> np.ndarray:
    """Build synthetic latent codes aligned to the synthetic DataFrame.

    Generates _TOTAL_ROWS latent codes from 3 well-separated clusters
    cycling in blocks of _BLOCK_LEN, matching the state structure
    of :func:`_make_synthetic_df`.

    Args:
        rng: Numpy random generator.

    Returns:
        Array of shape ``(_TOTAL_ROWS, _LATENT_DIM)``.
    """
    parts: list[np.ndarray] = []
    for _ in _MATCH_TEAMS:
        for t in range(_SEQ_LEN):
            state = (t // _BLOCK_LEN) % _N_STATES
            point = rng.normal(loc=_LATENT_CENTERS[state], scale=0.3)
            parts.append(point)
    return np.vstack(parts)


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
def preprocessed_features(
    synthetic_df: pl.DataFrame,
    fitted_pipeline: PreprocessingPipeline,
) -> np.ndarray:
    """Preprocessed feature array from the fitted pipeline."""
    return fitted_pipeline.transform(synthetic_df)


@pytest.fixture()
def fitted_gmm(preprocessed_features: np.ndarray) -> GMMModel:
    """A GMMModel with K=3 fitted on the preprocessed synthetic data."""
    cfg = GMMConfig(k_min=3, k_max=3, n_init=10, random_state=42)
    model = GMMModel(cfg)
    model._fit_k(preprocessed_features, k=_N_STATES)
    return model


@pytest.fixture()
def gmm_labels(
    preprocessed_features: np.ndarray,
    fitted_gmm: GMMModel,
) -> np.ndarray:
    """Direct-GMM labels for the preprocessed data."""
    return fitted_gmm.predict(preprocessed_features)


@pytest.fixture()
def synthetic_latent_codes() -> np.ndarray:
    """Pre-computed synthetic latent codes aligned to synthetic_df."""
    rng = np.random.default_rng(seed=99)
    return _make_synthetic_latent_codes(rng)


@pytest.fixture()
def discovery_result(
    synthetic_df: pl.DataFrame,
    fitted_pipeline: PreprocessingPipeline,
    synthetic_latent_codes: np.ndarray,
    fitted_gmm: GMMModel,
    gmm_labels: np.ndarray,
) -> tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel]:
    """Full run_vae_gmm_discovery result tuple."""
    return run_vae_gmm_discovery(
        df_windows=synthetic_df,
        dataset_label="test_dataset",
        pipeline=fitted_pipeline,
        latent_codes=synthetic_latent_codes,
        gmm_model=fitted_gmm,
        gmm_labels=gmm_labels,
    )


# ------------------------------------------------------------------
# Tests: helper functions
# ------------------------------------------------------------------


class TestSafeSilhouette:
    """Tests for :func:`_safe_silhouette`."""

    def test_returns_valid_score(self) -> None:
        """Normal case returns a score in [-1, 1]."""
        rng = np.random.default_rng(0)
        features = rng.normal(size=(60, 4))
        labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
        score = _safe_silhouette(features, labels)
        assert -1.0 <= score <= 1.0

    def test_single_cluster_returns_negative_one(self) -> None:
        """Only one unique label yields -1.0."""
        features = np.random.default_rng(0).normal(size=(20, 4))
        labels = np.zeros(20, dtype=int)
        assert _safe_silhouette(features, labels) == -1.0

    def test_all_unique_labels_returns_negative_one(self) -> None:
        """One label per sample yields -1.0 (n_unique >= n_samples)."""
        features = np.random.default_rng(0).normal(size=(5, 2))
        labels = np.arange(5)
        assert _safe_silhouette(features, labels) == -1.0


class TestComputeAgreement:
    """Tests for :func:`_compute_agreement`."""

    def test_perfect_agreement(self) -> None:
        """Identical labels produce ARI=1, NMI=1."""
        labels = np.array([0, 0, 1, 1, 2, 2])
        ari, nmi = _compute_agreement(labels, labels)
        assert ari == pytest.approx(1.0)
        assert nmi == pytest.approx(1.0)

    def test_permuted_labels_high_agreement(self) -> None:
        """Relabelled clusters still show high ARI/NMI."""
        a = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        b = np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])
        ari, nmi = _compute_agreement(a, b)
        assert ari == pytest.approx(1.0)
        assert nmi == pytest.approx(1.0)

    def test_random_labels_low_agreement(self) -> None:
        """Random labels produce low ARI."""
        rng = np.random.default_rng(42)
        a = rng.integers(0, 3, size=200)
        b = rng.integers(0, 3, size=200)
        ari, nmi = _compute_agreement(a, b)
        assert ari < 0.2
        assert 0.0 <= nmi <= 1.0


# ------------------------------------------------------------------
# Tests: ComparisonRow / ComparisonTable
# ------------------------------------------------------------------


class TestComparisonTable:
    """Tests for :class:`ComparisonTable` and :class:`ComparisonRow`."""

    def test_to_polars_columns(self) -> None:
        """to_polars returns expected column set."""
        row_a = ComparisonRow(
            model_name="A",
            n_states=3,
            bic=100.0,
            silhouette=0.5,
            silhouette_latent=None,
        )
        row_b = ComparisonRow(
            model_name="B",
            n_states=3,
            bic=90.0,
            silhouette=0.6,
            silhouette_latent=0.7,
        )
        table = ComparisonTable(
            rows=(row_a, row_b),
            agreement_ari=0.8,
            agreement_nmi=0.9,
        )
        df = table.to_polars()
        expected_cols = {
            "model",
            "n_states",
            "bic",
            "silhouette",
            "silhouette_latent",
            "agreement_ari",
            "agreement_nmi",
        }
        assert set(df.columns) == expected_cols

    def test_to_polars_row_count(self) -> None:
        """to_polars returns one row per model."""
        row_a = ComparisonRow(
            model_name="A",
            n_states=3,
            bic=100.0,
            silhouette=0.5,
            silhouette_latent=None,
        )
        table = ComparisonTable(
            rows=(row_a,),
            agreement_ari=0.5,
            agreement_nmi=0.5,
        )
        df = table.to_polars()
        assert df.height == 1

    def test_to_polars_model_names(self) -> None:
        """Model names appear in the correct order."""
        row_a = ComparisonRow("Direct-GMM", 3, 100.0, 0.5, None)
        row_b = ComparisonRow("VAE-GMM", 3, 90.0, 0.6, 0.7)
        table = ComparisonTable(
            rows=(row_a, row_b),
            agreement_ari=0.8,
            agreement_nmi=0.9,
        )
        df = table.to_polars()
        assert df["model"].to_list() == ["Direct-GMM", "VAE-GMM"]


# ------------------------------------------------------------------
# Tests: run_vae_gmm_discovery -- return types
# ------------------------------------------------------------------


class TestRunVaeGmmDiscoveryTypes:
    """Type and structural checks on :func:`run_vae_gmm_discovery`."""

    def test_returns_correct_types(
        self,
        discovery_result: tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel],
    ) -> None:
        """Return tuple contains the expected types."""
        df_labeled, result, latent_gmm = discovery_result
        assert isinstance(df_labeled, pl.DataFrame)
        assert isinstance(result, VAEGMMDiscoveryResult)
        assert isinstance(latent_gmm, GMMModel)

    def test_comparison_table_type(
        self,
        discovery_result: tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel],
    ) -> None:
        """Result contains a ComparisonTable."""
        _, result, _ = discovery_result
        assert isinstance(result.comparison, ComparisonTable)
        assert len(result.comparison.rows) == 2


# ------------------------------------------------------------------
# Tests: run_vae_gmm_discovery -- labels
# ------------------------------------------------------------------


class TestRunVaeGmmDiscoveryLabels:
    """Label assignment tests for :func:`run_vae_gmm_discovery`."""

    def test_labels_assigned_to_all_rows(
        self,
        discovery_result: tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel],
    ) -> None:
        """Every retained row has a ``vae_gmm_state_label``."""
        df_labeled, result, _ = discovery_result
        assert "vae_gmm_state_label" in df_labeled.columns
        assert df_labeled["vae_gmm_state_label"].null_count() == 0
        assert df_labeled.height == result.n_windows_retained

    def test_labels_in_range(
        self,
        discovery_result: tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel],
    ) -> None:
        """Labels are in {0, 1, ..., n_states-1}."""
        df_labeled, result, _ = discovery_result
        labels = df_labeled["vae_gmm_state_label"].to_numpy()
        assert np.all(labels >= 0)
        assert np.all(labels < result.n_states)

    def test_multiple_states_assigned(
        self,
        discovery_result: tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel],
    ) -> None:
        """At least 2 distinct states are assigned (data has 3 clusters)."""
        df_labeled, _, _ = discovery_result
        n_unique = df_labeled["vae_gmm_state_label"].n_unique()
        assert n_unique >= 2


# ------------------------------------------------------------------
# Tests: run_vae_gmm_discovery -- probability columns
# ------------------------------------------------------------------


class TestRunVaeGmmDiscoveryProba:
    """Probability column tests for :func:`run_vae_gmm_discovery`."""

    def test_probability_columns_present(
        self,
        discovery_result: tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel],
    ) -> None:
        """Per-state probability columns exist."""
        df_labeled, result, _ = discovery_result
        prob_cols = [
            c for c in df_labeled.columns if c.startswith("vae_gmm_state_prob_")
        ]
        assert len(prob_cols) == result.n_states

    def test_probability_columns_sum_to_one(
        self,
        discovery_result: tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel],
    ) -> None:
        """Per-state probabilities sum to ~1.0 for each row."""
        df_labeled, _, _ = discovery_result
        prob_cols = [
            c for c in df_labeled.columns if c.startswith("vae_gmm_state_prob_")
        ]
        row_sums = df_labeled.select(prob_cols).sum_horizontal()
        np.testing.assert_allclose(
            row_sums.to_numpy(),
            1.0,
            atol=1e-5,
        )

    def test_probabilities_non_negative(
        self,
        discovery_result: tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel],
    ) -> None:
        """All probability values are non-negative."""
        df_labeled, _, _ = discovery_result
        prob_cols = [
            c for c in df_labeled.columns if c.startswith("vae_gmm_state_prob_")
        ]
        for col in prob_cols:
            assert (df_labeled[col].to_numpy() >= 0.0).all()


# ------------------------------------------------------------------
# Tests: run_vae_gmm_discovery -- metrics
# ------------------------------------------------------------------


class TestRunVaeGmmDiscoveryMetrics:
    """Metric validation for :func:`run_vae_gmm_discovery`."""

    def test_silhouette_direct_in_range(
        self,
        discovery_result: tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel],
    ) -> None:
        """Direct-GMM silhouette is in [-1, 1]."""
        _, result, _ = discovery_result
        assert -1.0 <= result.silhouette_direct <= 1.0

    def test_silhouette_vae_original_in_range(
        self,
        discovery_result: tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel],
    ) -> None:
        """VAE-GMM silhouette on original features is in [-1, 1]."""
        _, result, _ = discovery_result
        assert -1.0 <= result.silhouette_vae_original <= 1.0

    def test_silhouette_vae_latent_in_range(
        self,
        discovery_result: tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel],
    ) -> None:
        """VAE-GMM silhouette on latent features is in [-1, 1]."""
        _, result, _ = discovery_result
        assert -1.0 <= result.silhouette_vae_latent <= 1.0

    def test_bic_values_finite(
        self,
        discovery_result: tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel],
    ) -> None:
        """Both BIC values are finite floats."""
        _, result, _ = discovery_result
        assert np.isfinite(result.bic_direct)
        assert np.isfinite(result.bic_latent)

    def test_ari_in_range(
        self,
        discovery_result: tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel],
    ) -> None:
        """Agreement ARI is in [-1, 1]."""
        _, result, _ = discovery_result
        assert -1.0 <= result.agreement_ari <= 1.0

    def test_nmi_in_range(
        self,
        discovery_result: tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel],
    ) -> None:
        """Agreement NMI is in [0, 1]."""
        _, result, _ = discovery_result
        assert 0.0 <= result.agreement_nmi <= 1.0

    def test_well_separated_data_has_positive_silhouette(
        self,
        discovery_result: tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel],
    ) -> None:
        """With 3 well-separated clusters, silhouette should be > 0."""
        _, result, _ = discovery_result
        assert result.silhouette_direct > 0.0


# ------------------------------------------------------------------
# Tests: run_vae_gmm_discovery -- result fields
# ------------------------------------------------------------------


class TestRunVaeGmmDiscoveryResultFields:
    """Field-level checks on :class:`VAEGMMDiscoveryResult`."""

    def test_dataset_label(
        self,
        discovery_result: tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel],
    ) -> None:
        """dataset_label matches what was passed in."""
        _, result, _ = discovery_result
        assert result.dataset_label == "test_dataset"

    def test_n_states_matches_gmm(
        self,
        discovery_result: tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel],
    ) -> None:
        """n_states defaults to the direct GMM's n_states."""
        _, result, _ = discovery_result
        assert result.n_states == _N_STATES

    def test_n_windows(
        self,
        discovery_result: tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel],
    ) -> None:
        """n_windows equals the total input row count."""
        _, result, _ = discovery_result
        assert result.n_windows == _TOTAL_ROWS

    def test_n_windows_retained(
        self,
        discovery_result: tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel],
    ) -> None:
        """n_windows_retained > 0 and <= n_windows."""
        _, result, _ = discovery_result
        assert 0 < result.n_windows_retained <= result.n_windows

    def test_latent_dim(
        self,
        discovery_result: tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel],
    ) -> None:
        """latent_dim matches the synthetic latent codes."""
        _, result, _ = discovery_result
        assert result.latent_dim == _LATENT_DIM

    def test_comparison_agreement_matches_result(
        self,
        discovery_result: tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel],
    ) -> None:
        """Comparison table agreement matches top-level result fields."""
        _, result, _ = discovery_result
        assert result.comparison.agreement_ari == pytest.approx(result.agreement_ari)
        assert result.comparison.agreement_nmi == pytest.approx(result.agreement_nmi)

    def test_comparison_row_names(
        self,
        discovery_result: tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel],
    ) -> None:
        """Comparison table has Direct-GMM and VAE-GMM rows."""
        _, result, _ = discovery_result
        names = [r.model_name for r in result.comparison.rows]
        assert names == ["Direct-GMM", "VAE-GMM"]


# ------------------------------------------------------------------
# Tests: run_vae_gmm_discovery -- comparison table polars output
# ------------------------------------------------------------------


class TestComparisonTablePolarsOutput:
    """Tests for :meth:`ComparisonTable.to_polars` from real runs."""

    def test_to_polars_has_two_rows(
        self,
        discovery_result: tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel],
    ) -> None:
        """Polars comparison table has 2 rows."""
        _, result, _ = discovery_result
        df = result.comparison.to_polars()
        assert df.height == 2

    def test_to_polars_bic_values_match(
        self,
        discovery_result: tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel],
    ) -> None:
        """BIC values in polars table match the result fields."""
        _, result, _ = discovery_result
        df = result.comparison.to_polars()
        bic_values = df["bic"].to_list()
        assert bic_values[0] == pytest.approx(result.bic_direct)
        assert bic_values[1] == pytest.approx(result.bic_latent)


# ------------------------------------------------------------------
# Tests: run_vae_gmm_discovery -- custom n_states
# ------------------------------------------------------------------


class TestRunVaeGmmDiscoveryCustomK:
    """Tests with a custom n_states override."""

    def test_custom_n_states(
        self,
        synthetic_df: pl.DataFrame,
        fitted_pipeline: PreprocessingPipeline,
        synthetic_latent_codes: np.ndarray,
        fitted_gmm: GMMModel,
        gmm_labels: np.ndarray,
    ) -> None:
        """n_states override is respected."""
        custom_k = 2
        df_labeled, result, latent_gmm = run_vae_gmm_discovery(
            df_windows=synthetic_df,
            dataset_label="custom_k_test",
            pipeline=fitted_pipeline,
            latent_codes=synthetic_latent_codes,
            gmm_model=fitted_gmm,
            gmm_labels=gmm_labels,
            n_states=custom_k,
        )
        assert result.n_states == custom_k
        assert latent_gmm.n_states == custom_k
        prob_cols = [
            c for c in df_labeled.columns if c.startswith("vae_gmm_state_prob_")
        ]
        assert len(prob_cols) == custom_k


# ------------------------------------------------------------------
# Tests: run_vae_gmm_discovery -- custom latent GMM config
# ------------------------------------------------------------------


class TestRunVaeGmmDiscoveryCustomGMMConfig:
    """Tests with a custom GMM config for the latent space."""

    def test_custom_gmm_config(
        self,
        synthetic_df: pl.DataFrame,
        fitted_pipeline: PreprocessingPipeline,
        synthetic_latent_codes: np.ndarray,
        fitted_gmm: GMMModel,
        gmm_labels: np.ndarray,
    ) -> None:
        """Custom latent GMM config is used."""
        custom_cfg = GMMConfig(
            k_min=_N_STATES,
            k_max=_N_STATES,
            covariance_type="diag",
            n_init=5,
            random_state=99,
        )
        _, result, _ = run_vae_gmm_discovery(
            df_windows=synthetic_df,
            dataset_label="custom_cfg_test",
            pipeline=fitted_pipeline,
            latent_codes=synthetic_latent_codes,
            gmm_model=fitted_gmm,
            gmm_labels=gmm_labels,
            gmm_latent_config=custom_cfg,
        )
        assert result.n_states == _N_STATES
        assert np.isfinite(result.bic_latent)


# ------------------------------------------------------------------
# Tests: latent code row-count mismatch
# ------------------------------------------------------------------


class TestLatentCodeValidation:
    """Tests for latent code alignment validation."""

    def test_mismatched_latent_rows_raises(
        self,
        synthetic_df: pl.DataFrame,
        fitted_pipeline: PreprocessingPipeline,
        fitted_gmm: GMMModel,
        gmm_labels: np.ndarray,
    ) -> None:
        """ValueError raised when latent code rows != retained rows."""
        wrong_codes = np.random.default_rng(0).normal(
            size=(10, _LATENT_DIM),
        )
        with pytest.raises(ValueError, match="row count"):
            run_vae_gmm_discovery(
                df_windows=synthetic_df,
                dataset_label="bad_rows",
                pipeline=fitted_pipeline,
                latent_codes=wrong_codes,
                gmm_model=fitted_gmm,
                gmm_labels=gmm_labels,
            )


# ------------------------------------------------------------------
# Tests: returned latent GMM is usable
# ------------------------------------------------------------------


class TestReturnedModels:
    """Verify that the returned latent GMM is fitted and usable."""

    def test_latent_gmm_predict_works(
        self,
        discovery_result: tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel],
        synthetic_latent_codes: np.ndarray,
    ) -> None:
        """Returned latent GMM can predict on latent codes."""
        _, _, latent_gmm = discovery_result
        labels = latent_gmm.predict(synthetic_latent_codes)
        assert labels.shape == (synthetic_latent_codes.shape[0],)
        assert np.all(labels >= 0)
        assert np.all(labels < _N_STATES)

    def test_latent_gmm_predict_proba_works(
        self,
        discovery_result: tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel],
        synthetic_latent_codes: np.ndarray,
    ) -> None:
        """Returned latent GMM produces valid probabilities."""
        _, _, latent_gmm = discovery_result
        proba = latent_gmm.predict_proba(synthetic_latent_codes)
        assert proba.shape == (synthetic_latent_codes.shape[0], _N_STATES)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
