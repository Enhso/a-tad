"""Per-dataset GMM discovery orchestration.

Provides :func:`run_gmm_discovery`, the single entry-point that takes a
polars DataFrame of window segments for **one** dataset and returns:

* a labelled copy of the DataFrame (with ``state_label`` and
  ``state_label_smoothed`` columns),
* a frozen :class:`DatasetDiscoveryResult` summary,
* the fitted :class:`PreprocessingPipeline`, and
* the fitted :class:`GMMModel`.

All heavy lifting is delegated to the already-tested library modules
(:mod:`~tactical.models.preprocessing`,
:mod:`~tactical.models.gmm`,
:mod:`~tactical.models.selection`).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import polars as pl

from tactical.models.gmm import GMMConfig, GMMModel
from tactical.models.preprocessing import PreprocessingPipeline
from tactical.models.selection import ModelSelectionResult, select_gmm_k

logger = logging.getLogger(__name__)

_TIER_PATTERN = re.compile(r"^t(\d+)_")


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DatasetDiscoveryResult:
    """Summary produced by :func:`run_gmm_discovery`.

    Attributes:
        dataset_label: Human-readable identifier for the dataset.
        selection: Full model-selection result across all candidate K.
        best_k: The K value chosen for the final model (BIC-based).
        n_windows: Number of window rows in the input DataFrame.
        n_windows_retained: Rows surviving null handling.
        n_features_in: Feature count before PCA.
        n_features_out: Feature count after PCA (or same if no PCA).
        flicker_rate_raw: Fraction of consecutive state changes
            before smoothing (averaged across match-team sequences).
        flicker_rate_smoothed: Same metric after temporal smoothing.
    """

    dataset_label: str
    selection: ModelSelectionResult
    best_k: int
    n_windows: int
    n_windows_retained: int
    n_features_in: int
    n_features_out: int
    flicker_rate_raw: float
    flicker_rate_smoothed: float


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _drop_tiers_above(
    df: pl.DataFrame,
    max_tier: int,
) -> pl.DataFrame:
    """Remove feature columns whose tier number exceeds *max_tier*.

    Non-feature columns (metadata) are always kept.

    Args:
        df: Input DataFrame.
        max_tier: Maximum feature tier to retain (e.g. 2 keeps
            ``t1_*`` and ``t2_*`` but drops ``t3_*``).

    Returns:
        DataFrame with high-tier feature columns removed.
    """
    cols_to_drop: list[str] = []
    for col in df.columns:
        m = _TIER_PATTERN.match(col)
        if m and int(m.group(1)) > max_tier:
            cols_to_drop.append(col)
    return df.drop(cols_to_drop) if cols_to_drop else df


def _compute_flicker_rate(
    df: pl.DataFrame,
    state_col: str,
) -> float:
    """Fraction of consecutive-pair state changes, averaged over sequences.

    Each unique ``(match_id, team_id)`` group is treated as an
    independent time-ordered sequence.  The flicker rate for a single
    sequence of length *n* is ``(# of i where s[i] != s[i-1]) / (n-1)``.
    The return value is the **mean** flicker rate across all sequences.

    Args:
        df: DataFrame containing *state_col*, ``match_id``,
            ``team_id``, and ``start_time`` columns.
        state_col: Name of the integer state column.

    Returns:
        Mean flicker rate in ``[0.0, 1.0]``.  Returns ``0.0`` when
        there are no sequences longer than 1 row.
    """
    rates: list[float] = []
    partitions = df.sort(["match_id", "team_id", "start_time"]).partition_by(
        ["match_id", "team_id"], maintain_order=True
    )
    for part in partitions:
        states = part[state_col].to_numpy()
        n = len(states)
        if n < 2:
            continue
        changes = int(np.sum(states[1:] != states[:-1]))
        rates.append(changes / (n - 1))
    return float(np.mean(rates)) if rates else 0.0


def _smooth_per_match(
    df: pl.DataFrame,
    raw_col: str,
    smoothed_col: str,
    window_size: int,
) -> pl.DataFrame:
    """Apply :meth:`GMMModel.smooth_states` within each match-team group.

    Rows are sorted by ``start_time`` within each
    ``(match_id, team_id)`` partition.  The original row order of *df*
    is **not** guaranteed to be preserved; callers should sort or
    re-index afterwards if needed.

    Args:
        df: DataFrame that already contains *raw_col*.
        raw_col: Name of the integer state-label column.
        smoothed_col: Name for the new smoothed column.
        window_size: Smoothing window forwarded to
            :meth:`GMMModel.smooth_states`.

    Returns:
        DataFrame with an additional *smoothed_col* column.
    """
    parts: list[pl.DataFrame] = []
    partitions = df.sort(["match_id", "team_id", "start_time"]).partition_by(
        ["match_id", "team_id"], maintain_order=True
    )
    for part in partitions:
        raw = part[raw_col].to_numpy()
        smoothed = GMMModel.smooth_states(raw, window_size=window_size)
        parts.append(
            part.with_columns(
                pl.Series(name=smoothed_col, values=smoothed),
            ),
        )
    return (
        pl.concat(parts)
        if parts
        else df.with_columns(
            pl.lit(None).cast(pl.Int64).alias(smoothed_col),
        )
    )


# ------------------------------------------------------------------
# Main entry-point
# ------------------------------------------------------------------


def run_gmm_discovery(
    df_windows: pl.DataFrame,
    dataset_label: str,
    gmm_config: GMMConfig,
    feature_prefix: str = "t",
    null_strategy: str = "impute_median",
    pca_variance_threshold: float | None = 0.95,
    max_feature_tier: int = 2,
    smooth_window: int = 5,
) -> tuple[pl.DataFrame, DatasetDiscoveryResult, PreprocessingPipeline, GMMModel]:
    """Run the full GMM discovery pipeline for a single dataset.

    Steps performed:

    1. Drop feature columns above *max_feature_tier*.
    2. Fit a :class:`PreprocessingPipeline` (scaling + optional PCA).
    3. Run :func:`select_gmm_k` across the configured K range.
    4. Fit a final :class:`GMMModel` with the BIC-optimal K.
    5. Predict state labels and probabilities for every retained row.
    6. Apply per-match temporal smoothing.
    7. Compute raw and smoothed flicker rates.

    Args:
        df_windows: polars DataFrame of **window** segments for one
            dataset.  Must contain metadata columns (``match_id``,
            ``team_id``, ``start_time``, …) and feature columns.
        dataset_label: Human-readable name for logging / results.
        gmm_config: GMM hyper-parameters (K range, covariance type,
            etc.).
        feature_prefix: Passed to :class:`PreprocessingPipeline`.
        null_strategy: ``"drop_rows"`` or ``"impute_median"``.
        pca_variance_threshold: Cumulative variance for PCA.  Pass
            ``None`` to skip PCA entirely.  Defaults to ``0.95``.
        max_feature_tier: Highest tier to keep (1, 2, or 3).
        smooth_window: Window size for the moving-mode smoother.

    Returns:
        A 4-tuple of:

        * **df_labeled** -- copy of the retained rows with added
          ``state_label``, ``state_label_smoothed``, and per-state
          probability columns (``state_prob_0``, …).
        * **result** -- :class:`DatasetDiscoveryResult` summary.
        * **pipeline** -- the fitted preprocessing pipeline.
        * **model** -- the fitted GMM model.

    Raises:
        ValueError: If *df_windows* is empty or has no feature
            columns after tier filtering.
        tactical.exceptions.ModelFitError: If GMM fitting fails
            for every candidate K.
    """
    logger.info(
        "Starting GMM discovery for '%s' (%d rows)",
        dataset_label,
        df_windows.height,
    )

    # -- 1. Tier filtering ---------------------------------------------
    df_filtered = _drop_tiers_above(df_windows, max_feature_tier)

    # -- 2. Preprocessing (scale + PCA) --------------------------------
    pipeline = PreprocessingPipeline(
        feature_prefix=feature_prefix,
        null_strategy=null_strategy,
        pca_variance_threshold=pca_variance_threshold,
    )
    features = pipeline.fit_transform(df_filtered)

    retained_mask = pipeline.get_retained_row_mask(df_filtered)
    df_retained = df_filtered.filter(pl.Series(retained_mask))

    n_in = pipeline.n_features_in
    n_out = pipeline.n_features_out

    logger.info(
        "  Preprocessing: %d -> %d rows, %d -> %d features (PCA)",
        df_windows.height,
        features.shape[0],
        n_in,
        n_out,
    )

    # -- 3. Model selection --------------------------------------------
    logger.info(
        "  Model selection: K in [%d, %d]",
        gmm_config.k_min,
        gmm_config.k_max,
    )
    selection = select_gmm_k(features, gmm_config)
    best_k = selection.best_k_bic

    logger.info(
        "  Best K (BIC): %d  |  Best K (silhouette): %d",
        best_k,
        selection.best_k_silhouette,
    )

    # -- 4. Fit final model with best K --------------------------------
    model = GMMModel(gmm_config)
    model._fit_k(features, best_k)

    # -- 5. Predict labels + probabilities -----------------------------
    labels = model.predict(features)
    proba = model.predict_proba(features)

    # Add state_label column
    df_labeled = df_retained.with_columns(
        pl.Series(name="state_label", values=labels),
    )

    # Add per-state probability columns
    prob_exprs: list[pl.Series] = [
        pl.Series(name=f"state_prob_{k}", values=proba[:, k]) for k in range(best_k)
    ]
    df_labeled = df_labeled.with_columns(prob_exprs)

    # -- 6. Temporal smoothing per match-team --------------------------
    df_labeled = _smooth_per_match(
        df_labeled,
        raw_col="state_label",
        smoothed_col="state_label_smoothed",
        window_size=smooth_window,
    )

    # -- 7. Flicker rates ----------------------------------------------
    flicker_raw = _compute_flicker_rate(df_labeled, "state_label")
    flicker_smooth = _compute_flicker_rate(df_labeled, "state_label_smoothed")

    logger.info(
        "  Flicker rate: raw=%.3f  smoothed=%.3f (reduction %.0f%%)",
        flicker_raw,
        flicker_smooth,
        100.0 * (1.0 - flicker_smooth / flicker_raw) if flicker_raw > 0 else 0.0,
    )

    # -- Build result --------------------------------------------------
    result = DatasetDiscoveryResult(
        dataset_label=dataset_label,
        selection=selection,
        best_k=best_k,
        n_windows=df_windows.height,
        n_windows_retained=features.shape[0],
        n_features_in=n_in,
        n_features_out=n_out,
        flicker_rate_raw=flicker_raw,
        flicker_rate_smoothed=flicker_smooth,
    )

    return df_labeled, result, pipeline, model
