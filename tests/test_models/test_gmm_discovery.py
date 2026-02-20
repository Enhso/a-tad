"""Tests for :mod:`tactical.models.discovery`.

Exercises :func:`run_gmm_discovery` end-to-end with a synthetic
multi-match DataFrame whose feature columns contain 3 well-separated
Gaussian blobs.  Verifies that the pipeline selects the correct K,
assigns state labels to every retained row, and that temporal
smoothing reduces the flicker rate.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from tactical.models.discovery import (
    DatasetDiscoveryResult,
    _compute_flicker_rate,
    _drop_tiers_above,
    _smooth_per_match,
    run_gmm_discovery,
)
from tactical.models.gmm import GMMConfig, GMMModel
from tactical.models.preprocessing import PreprocessingPipeline

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_RNG_SEED = 99
_N_MATCHES = 5
_ROWS_PER_MATCH = 60  # per team => 60 rows per (match, team) pair
_N_CLUSTERS = 3
_CENTERS = np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
_T1_COLS = ("t1_a", "t1_b", "t1_c")
_T2_COLS = ("t2_a", "t2_b")
_T3_COLS = ("t3_a",)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _build_synthetic_df(
    n_matches: int = _N_MATCHES,
    rows_per_match: int = _ROWS_PER_MATCH,
    include_t3: bool = True,
) -> pl.DataFrame:
    """Build a synthetic feature DataFrame with clear cluster structure.

    Each (match, team) pair gets *rows_per_match* rows whose t1/t2
    feature values are drawn from one of ``_N_CLUSTERS`` Gaussian
    blobs, cycling through clusters so that every match contains all
    three states.

    Args:
        n_matches: Number of synthetic matches.
        rows_per_match: Rows per (match, team) pair.
        include_t3: Whether to add a Tier 3 column (all null).

    Returns:
        polars DataFrame ready for :func:`run_gmm_discovery`.
    """
    rng = np.random.default_rng(seed=_RNG_SEED)

    match_ids: list[str] = []
    team_ids: list[str] = []
    segment_types: list[str] = []
    start_times: list[float] = []
    end_times: list[float] = []
    periods: list[int] = []
    match_minutes: list[float] = []
    feat_values: dict[str, list[float | None]] = {c: [] for c in (*_T1_COLS, *_T2_COLS)}
    if include_t3:
        feat_values.update({c: [] for c in _T3_COLS})

    for m_idx in range(n_matches):
        for team_side in ("home", "away"):
            for row_idx in range(rows_per_match):
                match_ids.append(f"match_{m_idx:03d}")
                team_ids.append(f"team_{team_side}_{m_idx}")
                segment_types.append("window")
                t = row_idx * 15.0
                start_times.append(t)
                end_times.append(t + 15.0)
                periods.append(1 if t < 2700 else 2)
                match_minutes.append(t / 60.0)

                # Cycle through clusters to ensure all 3 present
                cluster = row_idx % _N_CLUSTERS
                center = _CENTERS[cluster]
                point = rng.normal(loc=center, scale=0.4, size=len(center))

                # Map first 3 dims to t1 cols, remaining to t2 cols
                for i, col in enumerate(_T1_COLS):
                    feat_values[col].append(float(point[i]))
                for col in _T2_COLS:
                    feat_values[col].append(float(rng.normal(0, 0.3)))

                if include_t3:
                    for col in _T3_COLS:
                        feat_values[col].append(None)

    data: dict[str, list[str] | list[float] | list[int] | list[float | None]] = {
        "match_id": match_ids,
        "team_id": team_ids,
        "segment_type": segment_types,
        "start_time": start_times,
        "end_time": end_times,
        "period": periods,
        "match_minute": match_minutes,
    }
    data.update(feat_values)

    return pl.DataFrame(data)


@pytest.fixture()
def synthetic_df() -> pl.DataFrame:
    """Fixture: multi-match DataFrame with 3 clear clusters."""
    return _build_synthetic_df()


@pytest.fixture()
def gmm_config() -> GMMConfig:
    """Narrow K range so the test runs fast."""
    return GMMConfig(
        k_min=2,
        k_max=5,
        covariance_type="full",
        n_init=5,
        random_state=42,
    )


# ------------------------------------------------------------------
# Tests: full pipeline
# ------------------------------------------------------------------


class TestRunGmmDiscovery:
    """End-to-end tests for :func:`run_gmm_discovery`."""

    def test_returns_correct_types(
        self,
        synthetic_df: pl.DataFrame,
        gmm_config: GMMConfig,
    ) -> None:
        """Return tuple contains the expected types."""
        df_labeled, result, pipeline, model = run_gmm_discovery(
            df_windows=synthetic_df,
            dataset_label="test_dataset",
            gmm_config=gmm_config,
            max_feature_tier=2,
        )

        assert isinstance(df_labeled, pl.DataFrame)
        assert isinstance(result, DatasetDiscoveryResult)
        assert isinstance(pipeline, PreprocessingPipeline)
        assert isinstance(model, GMMModel)

    def test_selects_correct_k(
        self,
        synthetic_df: pl.DataFrame,
        gmm_config: GMMConfig,
    ) -> None:
        """BIC-optimal K is 3 on cleanly separated 3-cluster data."""
        _, result, _, _ = run_gmm_discovery(
            df_windows=synthetic_df,
            dataset_label="test_dataset",
            gmm_config=gmm_config,
            max_feature_tier=2,
        )

        assert result.best_k == 3

    def test_labels_assigned_to_all_rows(
        self,
        synthetic_df: pl.DataFrame,
        gmm_config: GMMConfig,
    ) -> None:
        """Every retained row has a ``state_label`` and ``state_label_smoothed``."""
        df_labeled, result, _, _ = run_gmm_discovery(
            df_windows=synthetic_df,
            dataset_label="test_dataset",
            gmm_config=gmm_config,
            max_feature_tier=2,
        )

        assert "state_label" in df_labeled.columns
        assert "state_label_smoothed" in df_labeled.columns
        assert df_labeled["state_label"].null_count() == 0
        assert df_labeled["state_label_smoothed"].null_count() == 0
        assert df_labeled.height == result.n_windows_retained

    def test_probability_columns_present(
        self,
        synthetic_df: pl.DataFrame,
        gmm_config: GMMConfig,
    ) -> None:
        """Per-state probability columns exist and sum to ~1.0."""
        df_labeled, result, _, _ = run_gmm_discovery(
            df_windows=synthetic_df,
            dataset_label="test_dataset",
            gmm_config=gmm_config,
            max_feature_tier=2,
        )

        prob_cols = [c for c in df_labeled.columns if c.startswith("state_prob_")]
        assert len(prob_cols) == result.best_k

        row_sums = df_labeled.select(prob_cols).sum_horizontal()
        np.testing.assert_allclose(
            row_sums.to_numpy(),
            1.0,
            atol=1e-6,
        )

    def test_state_labels_cover_all_k(
        self,
        synthetic_df: pl.DataFrame,
        gmm_config: GMMConfig,
    ) -> None:
        """Assigned labels span {0, 1, ..., K-1}."""
        df_labeled, result, _, _ = run_gmm_discovery(
            df_windows=synthetic_df,
            dataset_label="test_dataset",
            gmm_config=gmm_config,
            max_feature_tier=2,
        )

        unique_labels = set(df_labeled["state_label"].unique().to_list())
        assert unique_labels == set(range(result.best_k))

    def test_smoothing_reduces_flicker(
        self,
        synthetic_df: pl.DataFrame,
        gmm_config: GMMConfig,
    ) -> None:
        """Smoothed flicker rate is at most the raw flicker rate."""
        _, result, _, _ = run_gmm_discovery(
            df_windows=synthetic_df,
            dataset_label="test_dataset",
            gmm_config=gmm_config,
            max_feature_tier=2,
        )

        assert result.flicker_rate_smoothed <= result.flicker_rate_raw

    def test_result_fields_populated(
        self,
        synthetic_df: pl.DataFrame,
        gmm_config: GMMConfig,
    ) -> None:
        """All :class:`DatasetDiscoveryResult` fields have sane values."""
        _, result, _, _ = run_gmm_discovery(
            df_windows=synthetic_df,
            dataset_label="my_dataset",
            gmm_config=gmm_config,
            max_feature_tier=2,
        )

        assert result.dataset_label == "my_dataset"
        assert result.n_windows == synthetic_df.height
        assert result.n_windows_retained > 0
        assert result.n_features_in > 0
        assert result.n_features_out > 0
        assert result.n_features_out <= result.n_features_in
        assert 0.0 <= result.flicker_rate_raw <= 1.0
        assert 0.0 <= result.flicker_rate_smoothed <= 1.0

    def test_impute_preserves_all_rows(
        self,
        gmm_config: GMMConfig,
    ) -> None:
        """With ``impute_median``, all input rows are retained."""
        df = _build_synthetic_df(n_matches=3, rows_per_match=30, include_t3=False)

        df_labeled, result, _, _ = run_gmm_discovery(
            df_windows=df,
            dataset_label="test",
            gmm_config=gmm_config,
            null_strategy="impute_median",
            max_feature_tier=2,
            pca_variance_threshold=None,
        )

        assert df_labeled.height == df.height
        assert result.n_windows_retained == df.height

    def test_no_pca_preserves_feature_count(
        self,
        gmm_config: GMMConfig,
    ) -> None:
        """With ``pca_variance_threshold=None``, feature count unchanged."""
        df = _build_synthetic_df(n_matches=3, rows_per_match=30, include_t3=False)
        n_feat = len(_T1_COLS) + len(_T2_COLS)

        _, result, _, _ = run_gmm_discovery(
            df_windows=df,
            dataset_label="test",
            gmm_config=gmm_config,
            pca_variance_threshold=None,
            max_feature_tier=2,
        )

        assert result.n_features_in == n_feat
        assert result.n_features_out == n_feat


# ------------------------------------------------------------------
# Tests: helper functions
# ------------------------------------------------------------------


class TestDropTiersAbove:
    """Tests for :func:`_drop_tiers_above`."""

    def test_drops_t3_when_max_2(self) -> None:
        """Tier 3 columns removed when max_feature_tier=2."""
        df = pl.DataFrame(
            {"match_id": ["a"], "t1_x": [1.0], "t2_y": [2.0], "t3_z": [3.0]}
        )
        result = _drop_tiers_above(df, max_tier=2)

        assert "t1_x" in result.columns
        assert "t2_y" in result.columns
        assert "t3_z" not in result.columns
        assert "match_id" in result.columns

    def test_keeps_all_when_max_3(self) -> None:
        """All tiers kept when max_feature_tier=3."""
        df = pl.DataFrame({"t1_x": [1.0], "t2_y": [2.0], "t3_z": [3.0]})
        result = _drop_tiers_above(df, max_tier=3)

        assert result.columns == df.columns


class TestComputeFlickerRate:
    """Tests for :func:`_compute_flicker_rate`."""

    def test_constant_sequence_zero_flicker(self) -> None:
        """A constant sequence has flicker rate 0."""
        df = pl.DataFrame(
            {
                "match_id": ["m"] * 5,
                "team_id": ["t"] * 5,
                "start_time": [0.0, 1.0, 2.0, 3.0, 4.0],
                "state_label": [0, 0, 0, 0, 0],
            }
        )
        assert _compute_flicker_rate(df, "state_label") == pytest.approx(0.0)

    def test_alternating_sequence_full_flicker(self) -> None:
        """An alternating sequence has flicker rate 1."""
        df = pl.DataFrame(
            {
                "match_id": ["m"] * 4,
                "team_id": ["t"] * 4,
                "start_time": [0.0, 1.0, 2.0, 3.0],
                "state_label": [0, 1, 0, 1],
            }
        )
        assert _compute_flicker_rate(df, "state_label") == pytest.approx(1.0)

    def test_single_row_returns_zero(self) -> None:
        """A single-row group yields flicker rate 0."""
        df = pl.DataFrame(
            {
                "match_id": ["m"],
                "team_id": ["t"],
                "start_time": [0.0],
                "state_label": [0],
            }
        )
        assert _compute_flicker_rate(df, "state_label") == pytest.approx(0.0)


class TestSmoothPerMatch:
    """Tests for :func:`_smooth_per_match`."""

    def test_removes_isolated_flicker(self) -> None:
        """A single flipped label is corrected by smoothing."""
        df = pl.DataFrame(
            {
                "match_id": ["m"] * 5,
                "team_id": ["t"] * 5,
                "start_time": [0.0, 1.0, 2.0, 3.0, 4.0],
                "state_label": [0, 1, 0, 0, 0],
            }
        )
        result = _smooth_per_match(df, "state_label", "smoothed", window_size=3)

        expected = [0, 0, 0, 0, 0]
        assert result["smoothed"].to_list() == expected

    def test_preserves_row_count(self) -> None:
        """Smoothing does not add or drop rows."""
        df = pl.DataFrame(
            {
                "match_id": ["m"] * 6,
                "team_id": ["t"] * 6,
                "start_time": list(range(6)),
                "state_label": [0, 1, 0, 1, 0, 1],
            }
        )
        result = _smooth_per_match(df, "state_label", "smoothed", window_size=5)

        assert result.height == df.height
