"""Tests for :mod:`tactical.models.preprocessing`.

Uses a synthetic polars DataFrame fixture with 50 rows, 10 feature
columns (t1_a..t1_e, t2_a..t2_e), plus metadata columns. Nulls are
injected into 3 columns to exercise null-handling strategies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from tactical.models.preprocessing import PreprocessingPipeline

if TYPE_CHECKING:
    from pathlib import Path

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

_RNG = np.random.default_rng(seed=123)
_N = 50
_T1_COLS = [f"t1_{c}" for c in ("a", "b", "c", "d", "e")]
_T2_COLS = [f"t2_{c}" for c in ("a", "b", "c", "d", "e")]
_FEATURE_COLS = sorted(_T1_COLS + _T2_COLS)
_NULL_COLS = ("t1_a", "t2_b", "t2_d")
# Indices that will be set to null (deterministic)
_NULL_INDICES: dict[str, list[int]] = {
    "t1_a": [0, 7, 15, 30, 45],
    "t2_b": [3, 12, 28],
    "t2_d": [7, 40],
}


@pytest.fixture()
def feature_df() -> pl.DataFrame:
    """Synthetic DataFrame with 50 rows, 10 features, and some nulls."""
    rng = np.random.default_rng(seed=123)

    data: dict[str, list[float | str | None]] = {
        "match_id": [f"m_{i:03d}" for i in range(_N)],
        "team_id": [f"team_{'a' if i % 2 == 0 else 'b'}" for i in range(_N)],
        "segment_type": ["window"] * _N,
    }

    for col in _FEATURE_COLS:
        values: list[float | None] = rng.standard_normal(_N).tolist()
        if col in _NULL_INDICES:
            for idx in _NULL_INDICES[col]:
                values[idx] = None
        data[col] = values  # type: ignore[assignment]

    return pl.DataFrame(data)


def _count_null_rows(df: pl.DataFrame) -> int:
    """Count rows with at least one null in feature columns present."""
    feat_cols = [c for c in df.columns if c.startswith("t")]
    feat_cols = [c for c in feat_cols if c in _FEATURE_COLS]
    mask = pl.any_horizontal(*(pl.col(c).is_null() for c in feat_cols))
    return df.select(mask).to_series().sum()  # type: ignore[return-value]


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestPreprocessingPipeline:
    """Unit tests for :class:`PreprocessingPipeline`."""

    def test_fit_transform_shape(self, feature_df: pl.DataFrame) -> None:
        """Output is a 2-D numpy array with correct dimensions."""
        pipe = PreprocessingPipeline(
            feature_prefix="t",
            null_strategy="impute_median",
            pca_variance_threshold=None,
        )
        result = pipe.fit_transform(feature_df)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape == (_N, len(_FEATURE_COLS))

    def test_drop_rows_removes_nulls(self, feature_df: pl.DataFrame) -> None:
        """``null_strategy="drop_rows"`` yields fewer rows than input."""
        pipe = PreprocessingPipeline(
            feature_prefix="t",
            null_strategy="drop_rows",
            pca_variance_threshold=None,
        )
        result = pipe.fit_transform(feature_df)

        # There are rows with nulls, so output must be smaller
        assert result.shape[0] < _N

        # Exactly the rows without any null should remain
        null_row_count = _count_null_rows(feature_df)
        assert result.shape[0] == _N - null_row_count

    def test_impute_median_preserves_rows(
        self,
        feature_df: pl.DataFrame,
    ) -> None:
        """``null_strategy="impute_median"`` keeps all rows."""
        pipe = PreprocessingPipeline(
            feature_prefix="t",
            null_strategy="impute_median",
            pca_variance_threshold=None,
        )
        result = pipe.fit_transform(feature_df)

        assert result.shape[0] == _N

    def test_scaling_zero_mean_unit_var(
        self,
        feature_df: pl.DataFrame,
    ) -> None:
        """After transform, columns have approx mean=0 and std=1."""
        pipe = PreprocessingPipeline(
            feature_prefix="t",
            null_strategy="impute_median",
            pca_variance_threshold=None,
        )
        result = pipe.fit_transform(feature_df)

        col_means = result.mean(axis=0)
        col_stds = result.std(axis=0)

        np.testing.assert_allclose(col_means, 0.0, atol=1e-10)
        np.testing.assert_allclose(col_stds, 1.0, atol=0.15)

    def test_pca_reduces_dimensions(self, feature_df: pl.DataFrame) -> None:
        """PCA with ``pca_variance_threshold=0.95`` yields fewer columns."""
        pipe = PreprocessingPipeline(
            feature_prefix="t",
            null_strategy="impute_median",
            pca_variance_threshold=0.95,
        )
        result = pipe.fit_transform(feature_df)

        assert result.shape[1] < len(_FEATURE_COLS)
        assert pipe.pca_components is not None
        assert pipe.pca_components < len(_FEATURE_COLS)

    def test_pca_none_preserves_dimensions(
        self,
        feature_df: pl.DataFrame,
    ) -> None:
        """Without PCA, output has the same number of feature columns."""
        pipe = PreprocessingPipeline(
            feature_prefix="t",
            null_strategy="impute_median",
            pca_variance_threshold=None,
        )
        result = pipe.fit_transform(feature_df)

        assert result.shape[1] == len(_FEATURE_COLS)
        assert pipe.pca_components is None

    def test_prefix_filtering(self, feature_df: pl.DataFrame) -> None:
        """``feature_prefix="t1_"`` selects only Tier 1 columns."""
        pipe = PreprocessingPipeline(
            feature_prefix="t1_",
            null_strategy="impute_median",
            pca_variance_threshold=None,
        )
        result = pipe.fit_transform(feature_df)

        assert result.shape[1] == len(_T1_COLS)
        assert pipe.n_features_in == len(_T1_COLS)

    def test_transform_before_fit_raises(
        self,
        feature_df: pl.DataFrame,
    ) -> None:
        """Calling ``transform`` without ``fit`` raises RuntimeError."""
        pipe = PreprocessingPipeline()

        with pytest.raises(RuntimeError, match="not been fitted"):
            pipe.transform(feature_df)

    def test_save_load_roundtrip(
        self,
        feature_df: pl.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Save then load produces identical transform output."""
        pipe = PreprocessingPipeline(
            feature_prefix="t",
            null_strategy="impute_median",
            pca_variance_threshold=0.95,
        )
        expected = pipe.fit_transform(feature_df)

        save_path = tmp_path / "pipeline.pkl"
        pipe.save(save_path)

        loaded = PreprocessingPipeline.load(save_path)
        result = loaded.transform(feature_df)

        np.testing.assert_array_equal(result, expected)

    def test_retained_row_mask(self, feature_df: pl.DataFrame) -> None:
        """Mask correctly identifies which rows survive null handling."""
        pipe = PreprocessingPipeline(
            feature_prefix="t",
            null_strategy="drop_rows",
            pca_variance_threshold=None,
        )
        pipe.fit(feature_df)
        mask = pipe.get_retained_row_mask(feature_df)

        assert mask.dtype == bool
        assert mask.shape == (_N,)

        # Number of True values should equal transform output row count
        result = pipe.transform(feature_df)
        assert mask.sum() == result.shape[0]

        # Check mask against impute variant (all True)
        pipe_imp = PreprocessingPipeline(
            feature_prefix="t",
            null_strategy="impute_median",
            pca_variance_threshold=None,
        )
        pipe_imp.fit(feature_df)
        mask_imp = pipe_imp.get_retained_row_mask(feature_df)
        assert mask_imp.all()

    def test_n_features_properties(self, feature_df: pl.DataFrame) -> None:
        """``n_features_in`` and ``n_features_out`` are correct."""
        # Without PCA
        pipe_no_pca = PreprocessingPipeline(
            feature_prefix="t",
            null_strategy="impute_median",
            pca_variance_threshold=None,
        )
        pipe_no_pca.fit(feature_df)

        assert pipe_no_pca.n_features_in == len(_FEATURE_COLS)
        assert pipe_no_pca.n_features_out == len(_FEATURE_COLS)
        assert pipe_no_pca.n_features_in == pipe_no_pca.n_features_out

        # With PCA
        pipe_pca = PreprocessingPipeline(
            feature_prefix="t",
            null_strategy="impute_median",
            pca_variance_threshold=0.95,
        )
        pipe_pca.fit(feature_df)

        assert pipe_pca.n_features_in == len(_FEATURE_COLS)
        assert pipe_pca.n_features_out < pipe_pca.n_features_in
        assert pipe_pca.n_features_out == pipe_pca.pca_components
