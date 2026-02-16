"""Preprocessing pipeline for transforming feature DataFrames into model-ready arrays.

Sits between the Feature Engine's polars DataFrame output and the model
layer's numpy input. Handles feature column selection, null handling,
z-score normalization, and optional PCA dimensionality reduction.
"""

from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingPipeline:
    """Transforms a feature DataFrame into model-ready numpy arrays.

    Handles feature column selection, null handling, z-score
    normalization, and optional PCA dimensionality reduction.
    Fitted on training data and applied consistently to new data.

    Attributes:
        feature_prefix: Prefix filter for feature columns
            (e.g., "t1_" for Tier 1 only, or "t" for all tiers).
        null_strategy: How to handle nulls: "drop_rows" removes
            rows with any null, "impute_median" fills with
            column medians from training data.
        pca_variance_threshold: If set, apply PCA and retain
            components explaining this fraction of variance.
            If None, skip PCA.
    """

    feature_prefix: str = "t"
    null_strategy: str = "drop_rows"
    pca_variance_threshold: float | None = 0.95

    # Fitted state (not set at construction)
    _feature_columns: list[str] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _scaler: StandardScaler = field(
        default_factory=StandardScaler,
        init=False,
        repr=False,
    )
    _pca: PCA | None = field(default=None, init=False, repr=False)
    _medians: dict[str, float] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _is_fitted: bool = field(default=False, init=False, repr=False)

    _VALID_NULL_STRATEGIES = frozenset({"drop_rows", "impute_median"})
    _FEATURE_PATTERN: re.Pattern[str] = re.compile(r"^t\d+_")

    def _select_feature_columns(self, df: pl.DataFrame) -> list[str]:
        """Return sorted column names matching the feature prefix.

        Only columns following the ``t{digit}_*`` naming convention are
        considered feature columns. The *feature_prefix* is applied as
        an additional ``startswith`` filter on top of that convention.

        Args:
            df: Source DataFrame.

        Returns:
            Sorted list of matching column names.

        Raises:
            ValueError: If no columns match the prefix.
        """
        cols = sorted(
            c
            for c in df.columns
            if self._FEATURE_PATTERN.match(c) and c.startswith(self.feature_prefix)
        )
        if not cols:
            msg = (
                f"No feature columns found with prefix "
                f"'{self.feature_prefix}' in DataFrame columns."
            )
            raise ValueError(msg)
        return cols

    def _validate_null_strategy(self) -> None:
        """Raise ValueError for an unsupported null strategy."""
        if self.null_strategy not in self._VALID_NULL_STRATEGIES:
            msg = (
                f"Unknown null_strategy '{self.null_strategy}'. "
                f"Must be one of {sorted(self._VALID_NULL_STRATEGIES)}."
            )
            raise ValueError(msg)

    def _handle_nulls(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply the configured null strategy to feature columns.

        Args:
            df: DataFrame limited to feature columns.

        Returns:
            DataFrame with nulls handled.

        Raises:
            ValueError: If all rows are dropped.
        """
        if self.null_strategy == "drop_rows":
            result = df.drop_nulls(subset=self._feature_columns)
        else:
            # impute_median
            fill_exprs = [
                pl.col(c).fill_null(pl.lit(self._medians[c]))
                for c in self._feature_columns
                if c in self._medians
            ]
            result = df.with_columns(fill_exprs) if fill_exprs else df
        if result.height == 0:
            msg = "No rows remain after null handling."
            raise ValueError(msg)
        return result

    def fit(self, df: pl.DataFrame) -> PreprocessingPipeline:
        """Fit the pipeline on training data.

        Args:
            df: polars DataFrame from the feature pipeline.

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If no feature columns found or no rows
                remain after null handling.
        """
        self._validate_null_strategy()
        self._feature_columns = self._select_feature_columns(df)

        # Compute medians for all feature columns (used by impute_median)
        self._medians = {}
        for c in self._feature_columns:
            med = df.get_column(c).median()
            self._medians[c] = float(med) if isinstance(med, (int, float)) else 0.0

        # Handle nulls
        clean = self._handle_nulls(df)

        # Convert to numpy and fit scaler
        arr = clean.select(self._feature_columns).to_numpy(writable=True)
        self._scaler = StandardScaler()
        scaled = self._scaler.fit_transform(arr)

        # Optionally fit PCA
        if self.pca_variance_threshold is not None:
            max_components = min(scaled.shape[0], scaled.shape[1])
            self._pca = PCA(
                n_components=min(
                    max_components,
                    scaled.shape[1],
                ),
            )
            self._pca.fit(scaled)
            # Determine how many components to keep
            cumvar = np.cumsum(self._pca.explained_variance_ratio_)
            n_keep = int(np.searchsorted(cumvar, self.pca_variance_threshold) + 1)
            n_keep = min(n_keep, max_components)
            # Re-fit with the exact number of components
            self._pca = PCA(n_components=n_keep)
            self._pca.fit(scaled)
        else:
            self._pca = None

        self._is_fitted = True
        return self

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        """Transform new data using the fitted pipeline.

        Args:
            df: polars DataFrame with same feature columns.

        Returns:
            numpy array of shape (n_samples, n_features) or
            (n_samples, n_components) if PCA is applied.

        Raises:
            RuntimeError: If pipeline has not been fitted.
            ValueError: If required columns are missing.
        """
        if not self._is_fitted:
            msg = "Pipeline has not been fitted. Call fit() before transform()."
            raise RuntimeError(msg)

        missing = [c for c in self._feature_columns if c not in df.columns]
        if missing:
            msg = f"Missing required feature columns: {missing}"
            raise ValueError(msg)

        clean = self._handle_nulls(df)
        arr = clean.select(self._feature_columns).to_numpy(writable=True)
        scaled: np.ndarray = np.asarray(self._scaler.transform(arr))

        if self._pca is not None:
            return np.asarray(self._pca.transform(scaled))
        return scaled

    def fit_transform(self, df: pl.DataFrame) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            df: polars DataFrame from the feature pipeline.

        Returns:
            Transformed numpy array.
        """
        return self.fit(df).transform(df)

    def get_retained_row_mask(self, df: pl.DataFrame) -> np.ndarray:
        """Return boolean mask of rows retained after null handling.

        Useful for aligning transformed arrays with the original
        DataFrame when using ``null_strategy="drop_rows"``.

        Args:
            df: polars DataFrame with feature columns.

        Returns:
            1-D boolean numpy array of length ``df.height``.

        Raises:
            RuntimeError: If pipeline has not been fitted.
        """
        if not self._is_fitted:
            msg = (
                "Pipeline has not been fitted. Call fit() before "
                "get_retained_row_mask()."
            )
            raise RuntimeError(msg)

        if self.null_strategy == "drop_rows":
            null_expr = pl.any_horizontal(
                *(pl.col(c).is_null() for c in self._feature_columns)
            )
            mask: np.ndarray = ~df.select(null_expr).to_series().to_numpy()
            return mask

        # impute_median keeps all rows
        return np.ones(df.height, dtype=bool)

    @property
    def n_features_in(self) -> int:
        """Number of features before PCA.

        Raises:
            RuntimeError: If pipeline has not been fitted.
        """
        if not self._is_fitted:
            msg = "Pipeline has not been fitted."
            raise RuntimeError(msg)
        return len(self._feature_columns)

    @property
    def n_features_out(self) -> int:
        """Number of features after PCA (or same as in if no PCA).

        Raises:
            RuntimeError: If pipeline has not been fitted.
        """
        if not self._is_fitted:
            msg = "Pipeline has not been fitted."
            raise RuntimeError(msg)
        if self._pca is not None:
            return int(self._pca.n_components_)
        return len(self._feature_columns)

    @property
    def pca_components(self) -> int | None:
        """Number of PCA components retained, or None if no PCA.

        Raises:
            RuntimeError: If pipeline has not been fitted.
        """
        if not self._is_fitted:
            msg = "Pipeline has not been fitted."
            raise RuntimeError(msg)
        if self._pca is not None:
            return int(self._pca.n_components_)
        return None

    def save(self, path: Path) -> None:
        """Persist fitted pipeline to disk using pickle.

        Args:
            path: Destination file path.

        Raises:
            RuntimeError: If pipeline has not been fitted.
        """
        if not self._is_fitted:
            msg = "Cannot save an unfitted pipeline."
            raise RuntimeError(msg)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> PreprocessingPipeline:
        """Load a fitted pipeline from disk.

        Args:
            path: Path to a previously saved pipeline.

        Returns:
            The deserialized :class:`PreprocessingPipeline`.

        Raises:
            FileNotFoundError: If path does not exist.
            TypeError: If the loaded object is not a
                :class:`PreprocessingPipeline`.
        """
        with path.open("rb") as f:
            obj = pickle.load(f)  # noqa: S301
        if not isinstance(obj, cls):
            msg = f"Expected PreprocessingPipeline, got {type(obj).__name__}."
            raise TypeError(msg)
        return obj
