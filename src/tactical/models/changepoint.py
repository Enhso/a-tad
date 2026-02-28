"""Changepoint detection for tactical state time series.

Wraps the `ruptures <https://centre-borelli.github.io/ruptures-docs/>`_
library to detect structural breaks in latent codes, state probability
sequences, or raw feature time series.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import ruptures as rpt  # type: ignore[import-untyped]
from tqdm import tqdm

if TYPE_CHECKING:
    import polars as pl

    from tactical.config import ChangepointConfig

logger = logging.getLogger(__name__)

# Maximum number of feature deltas to report per changepoint.
_TOP_K_DELTAS = 5

# Number of windows on each side of a changepoint used to compute
# the mean difference (feature delta).
_DELTA_WINDOW = 5


@dataclass(frozen=True, slots=True)
class Changepoint:
    """A detected tactical changepoint within a match.

    Attributes:
        index: Window index where the change occurs.
        match_minute: Approximate match minute of the changepoint.
        feature_deltas: Top feature changes across the boundary,
            keyed by feature name with signed delta values.
    """

    index: int
    match_minute: float
    feature_deltas: dict[str, float]


@dataclass(frozen=True, slots=True)
class ChangepointResult:
    """All changepoints detected in a single match.

    Attributes:
        match_id: Identifier for the match.
        changepoints: Detected changepoints in temporal order.
        n_segments: Number of segments (``len(changepoints) + 1``).
    """

    match_id: str
    changepoints: tuple[Changepoint, ...]
    n_segments: int


def _build_algorithm(
    config: ChangepointConfig,
) -> rpt.Pelt | rpt.Binseg | rpt.Window:
    """Instantiate a ruptures search algorithm from *config*.

    Args:
        config: Changepoint detection configuration.

    Returns:
        An unfitted ruptures algorithm instance.

    Raises:
        ValueError: If ``config.method`` is not recognised.
    """
    supported = {"pelt", "binseg", "window"}
    if config.method not in supported:
        msg = (
            f"Unknown changepoint method '{config.method}'. "
            f"Supported: {sorted(supported)}"
        )
        raise ValueError(msg)

    if config.method == "window":
        return rpt.Window(
            model=config.model,
            min_size=config.min_segment_size,
            width=config.min_segment_size,
        )

    cls: type[rpt.Pelt] | type[rpt.Binseg] = (
        rpt.Pelt if config.method == "pelt" else rpt.Binseg
    )
    return cls(model=config.model, min_size=config.min_segment_size)


def _compute_feature_deltas(
    signal: np.ndarray,
    cp_index: int,
    feature_names: list[str],
) -> dict[str, float]:
    """Compute per-feature mean deltas around a changepoint.

    Takes the mean of ``signal[cp:cp+w]`` minus
    ``signal[cp-w:cp]`` for each feature and returns the top-k
    by absolute magnitude.

    Args:
        signal: Array of shape ``(n_windows, n_features)``.
        cp_index: Index of the changepoint.
        feature_names: Feature names aligned with signal columns.

    Returns:
        Dict mapping feature name to signed delta for the top
        features.
    """
    n = signal.shape[0]
    before_start = max(0, cp_index - _DELTA_WINDOW)
    after_end = min(n, cp_index + _DELTA_WINDOW)

    mean_before = signal[before_start:cp_index].mean(axis=0)
    mean_after = signal[cp_index:after_end].mean(axis=0)
    deltas = mean_after - mean_before

    # Rank by absolute magnitude, keep top-k.
    abs_deltas = np.abs(deltas)
    n_features = len(feature_names)
    k = min(_TOP_K_DELTAS, n_features)
    if k >= n_features:
        # Keep all features; just sort by descending absolute delta.
        top_indices = np.argsort(-abs_deltas)
    else:
        top_indices = np.argpartition(-abs_deltas, k)[:k]
        # Sort the top-k by descending absolute delta for determinism.
        top_indices = top_indices[np.argsort(-abs_deltas[top_indices])]

    return {feature_names[i]: float(deltas[i]) for i in top_indices}


def detect_changepoints(
    signal: np.ndarray,
    match_minutes: np.ndarray,
    match_id: str,
    config: ChangepointConfig,
    feature_names: list[str] | None = None,
) -> ChangepointResult:
    """Detect tactical changepoints in a time-series signal.

    Args:
        signal: Array of shape ``(n_windows, n_features)``. Can be
            latent codes, state probabilities, or raw features.
        match_minutes: Array of shape ``(n_windows,)`` with the
            match minute for each window.
        match_id: Match identifier attached to the result.
        config: Changepoint detection configuration.
        feature_names: Optional names for the features in *signal*,
            used when computing ``feature_deltas``.

    Returns:
        :class:`ChangepointResult` with detected changepoints.

    Raises:
        ValueError: If *signal* is empty or not two-dimensional.
    """
    if signal.ndim != 2:
        msg = f"signal must be 2-D (n_windows, n_features), got shape {signal.shape}"
        raise ValueError(msg)
    if signal.shape[0] == 0:
        msg = "signal is empty (0 windows)"
        raise ValueError(msg)

    algo = _build_algorithm(config)
    algo.fit(signal)
    breakpoints: list[int] = algo.predict(pen=config.penalty)

    # ruptures always appends ``len(signal)`` as the final element.
    if breakpoints and breakpoints[-1] == signal.shape[0]:
        breakpoints = breakpoints[:-1]

    changepoints: list[Changepoint] = []
    for cp_idx in breakpoints:
        minute = float(match_minutes[cp_idx])
        deltas: dict[str, float] = {}
        if feature_names is not None:
            deltas = _compute_feature_deltas(signal, cp_idx, feature_names)
        changepoints.append(
            Changepoint(
                index=cp_idx,
                match_minute=minute,
                feature_deltas=deltas,
            )
        )

    return ChangepointResult(
        match_id=match_id,
        changepoints=tuple(changepoints),
        n_segments=len(changepoints) + 1,
    )


def detect_changepoints_per_match(
    features_df: pl.DataFrame,
    signal_columns: list[str],
    config: ChangepointConfig,
    group_by_columns: list[str] | None = None,
) -> list[ChangepointResult]:
    """Run changepoint detection on each group in a DataFrame.

    Groups by *group_by_columns* (default ``["match_id"]``), sorts
    each group by ``match_minute``, extracts *signal_columns* as
    the signal matrix, and calls :func:`detect_changepoints` per
    group.

    When grouping by multiple columns (e.g.
    ``["match_id", "team_id"]``), the resulting
    :attr:`ChangepointResult.match_id` is the column values joined
    with ``"_"``.

    Args:
        features_df: Polars DataFrame with ``match_minute`` and
            the columns listed in *signal_columns* and
            *group_by_columns*.
        signal_columns: Column names to use as the signal.
        config: Changepoint configuration.
        group_by_columns: Columns to group by. Defaults to
            ``["match_id"]``.

    Returns:
        List of :class:`ChangepointResult`, one per group.
    """
    if group_by_columns is None:
        group_by_columns = ["match_id"]

    results: list[ChangepointResult] = []
    groups = features_df.group_by(group_by_columns, maintain_order=True)

    for key_vals, group_df in tqdm(
        groups,
        desc="Detecting changepoints",
    ):
        match_id = "_".join(str(v) for v in key_vals)
        group_df = group_df.sort("match_minute")
        signal = group_df.select(signal_columns).to_numpy()
        match_minutes = group_df["match_minute"].to_numpy()

        if signal.shape[0] < config.min_segment_size:
            logger.warning(
                "Match %s has only %d windows (< min_segment_size=%d), "
                "skipping changepoint detection",
                match_id,
                signal.shape[0],
                config.min_segment_size,
            )
            results.append(
                ChangepointResult(
                    match_id=match_id,
                    changepoints=(),
                    n_segments=1,
                )
            )
            continue

        result = detect_changepoints(
            signal=signal,
            match_minutes=match_minutes,
            match_id=match_id,
            config=config,
            feature_names=signal_columns,
        )
        results.append(result)

    return results
