"""Per-dataset HMM discovery orchestration.

Provides :func:`run_hmm_discovery`, the single entry-point that takes a
polars DataFrame of window segments for **one** dataset, an already-fitted
:class:`PreprocessingPipeline`, and optionally a fitted :class:`GMMModel`,
then returns:

* a labelled copy of the DataFrame (with ``hmm_state_label`` column),
* a frozen :class:`HMMDiscoveryResult` summary (including transition
  matrix and GMM agreement metrics),
* the fitted :class:`HMMModel`.

The HMM is fitted with per-match-team sequence boundaries so that
temporal structure is respected across match boundaries.  When a fitted
GMM is supplied the HMM emission parameters are seeded from the GMM
cluster centres, avoiding poor local optima.

State agreement between GMM and HMM is quantified with the Adjusted
Rand Index (ARI) and Normalised Mutual Information (NMI).

Transition matrix visualisation is provided by
:func:`plot_transition_matrix`, which renders a heatmap to a
:class:`matplotlib.figure.Figure` and can optionally save it to disk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import (  # type: ignore[import-untyped]
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from tactical.exceptions import ModelFitError
from tactical.models.hmm import HMMModel

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure

    from tactical.config import HMMConfig
    from tactical.models.gmm import GMMModel
    from tactical.models.preprocessing import PreprocessingPipeline

logger = logging.getLogger(__name__)

# Use non-interactive backend so figures can be created without a display
matplotlib.use("Agg")


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HMMDiscoveryResult:
    """Summary produced by :func:`run_hmm_discovery`.

    Attributes:
        dataset_label: Human-readable identifier for the dataset.
        n_states: Number of hidden states in the fitted HMM.
        n_sequences: Number of match-team sequences used for fitting.
        n_windows: Total window rows in the input DataFrame.
        n_windows_retained: Rows surviving preprocessing.
        transition_matrix: Learned state transition matrix (K x K).
        gmm_agreement_ari: Adjusted Rand Index between HMM and GMM
            state labels.  ``None`` when no GMM comparison was done.
        gmm_agreement_nmi: Normalised Mutual Information between HMM
            and GMM state labels.  ``None`` when no GMM comparison.
        flicker_rate: Fraction of consecutive HMM state changes,
            averaged across match-team sequences.
    """

    dataset_label: str
    n_states: int
    n_sequences: int
    n_windows: int
    n_windows_retained: int
    transition_matrix: np.ndarray = field(repr=False)
    gmm_agreement_ari: float | None
    gmm_agreement_nmi: float | None
    flicker_rate: float


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _build_sequences(
    df: pl.DataFrame,
) -> tuple[list[int], pl.DataFrame]:
    """Build per-match-team sequence lengths from sorted DataFrame.

    Rows are sorted by ``(match_id, team_id, start_time)`` and
    grouped by ``(match_id, team_id)`` to produce one contiguous
    sequence per match-team pair.

    Args:
        df: DataFrame with ``match_id``, ``team_id``, and
            ``start_time`` columns.

    Returns:
        Tuple of (sequence_lengths, sorted DataFrame).
    """
    df_sorted = df.sort(["match_id", "team_id", "start_time"])
    groups = df_sorted.group_by(["match_id", "team_id"], maintain_order=True).len()
    lengths: list[int] = groups["len"].to_list()
    return lengths, df_sorted


def _compute_flicker_rate(
    df: pl.DataFrame,
    state_col: str,
) -> float:
    """Fraction of consecutive-pair state changes, averaged over sequences.

    Each unique ``(match_id, team_id)`` group is treated as an
    independent time-ordered sequence.

    Args:
        df: DataFrame containing *state_col*, ``match_id``,
            ``team_id``, and ``start_time`` columns.
        state_col: Name of the integer state column.

    Returns:
        Mean flicker rate in ``[0.0, 1.0]``.
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


def _compute_agreement(
    hmm_labels: np.ndarray,
    gmm_labels: np.ndarray,
) -> tuple[float, float]:
    """Compute ARI and NMI between two label vectors.

    Args:
        hmm_labels: Integer labels from HMM Viterbi decoding.
        gmm_labels: Integer labels from GMM prediction.

    Returns:
        Tuple of (adjusted_rand_index, normalised_mutual_info).
    """
    ari = float(adjusted_rand_score(gmm_labels, hmm_labels))
    nmi = float(normalized_mutual_info_score(gmm_labels, hmm_labels))
    return ari, nmi


# ------------------------------------------------------------------
# Visualisation
# ------------------------------------------------------------------


def plot_transition_matrix(
    transition_matrix: np.ndarray,
    title: str = "HMM State Transition Matrix",
    save_path: Path | None = None,
    figsize: tuple[float, float] = (8.0, 6.5),
) -> Figure:
    """Render a transition matrix as an annotated heatmap.

    Args:
        transition_matrix: Square array of shape ``(K, K)`` where
            ``transition_matrix[i, j]`` is ``P(state_j | state_i)``.
        title: Figure title.
        save_path: If provided, the figure is saved to this path
            (PNG/PDF/SVG depending on extension).
        figsize: Width and height in inches.

    Returns:
        The :class:`matplotlib.figure.Figure` object.
    """
    k = transition_matrix.shape[0]
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(transition_matrix, cmap="YlOrRd", vmin=0.0, vmax=1.0)

    # Annotate each cell with the probability value
    for i in range(k):
        for j in range(k):
            val = transition_matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=max(8, 14 - k),
            )

    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels([f"S{i}" for i in range(k)])
    ax.set_yticklabels([f"S{i}" for i in range(k)])
    ax.set_xlabel("To State")
    ax.set_ylabel("From State")
    ax.set_title(title)

    fig.colorbar(im, ax=ax, label="Transition Probability")
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Transition matrix saved to %s", save_path)

    plt.close(fig)
    return fig


# ------------------------------------------------------------------
# Main entry-point
# ------------------------------------------------------------------


def run_hmm_discovery(
    df_windows: pl.DataFrame,
    dataset_label: str,
    pipeline: PreprocessingPipeline,
    hmm_config: HMMConfig,
    gmm_model: GMMModel | None = None,
    gmm_labels: np.ndarray | None = None,
) -> tuple[pl.DataFrame, HMMDiscoveryResult, HMMModel]:
    """Run the full HMM discovery pipeline for a single dataset.

    Steps performed:

    1. Transform features using the provided (already-fitted)
       preprocessing pipeline.
    2. Build per-match-team sequence lengths.
    3. Fit HMM (optionally initialised from GMM parameters).
    4. Predict HMM state labels via Viterbi decoding.
    5. Predict state posterior probabilities.
    6. If GMM labels are available, compute state agreement
       (ARI, NMI).
    7. Extract transition matrix.
    8. Compute flicker rate.

    Args:
        df_windows: polars DataFrame of **window** segments for one
            dataset.  Must contain metadata columns (``match_id``,
            ``team_id``, ``start_time``) and feature columns.
        dataset_label: Human-readable name for logging / results.
        pipeline: A **fitted** :class:`PreprocessingPipeline`,
            typically loaded from GMM discovery artifacts.
        hmm_config: HMM hyper-parameters.
        gmm_model: Optional fitted GMM whose parameters seed the
            HMM.  When ``hmm_config.init_from_gmm`` is ``True``
            and this is provided, :meth:`HMMModel.fit_from_gmm`
            is used.
        gmm_labels: Optional GMM state labels aligned to the
            retained rows of *df_windows* (after preprocessing).
            Used for computing agreement metrics.

    Returns:
        A 3-tuple of:

        * **df_labeled** -- copy of retained rows with added
          ``hmm_state_label`` and per-state probability columns
          (``hmm_state_prob_0``, ...).
        * **result** -- :class:`HMMDiscoveryResult` summary.
        * **model** -- the fitted HMM model.

    Raises:
        ValueError: If *df_windows* is empty or has no feature
            columns.
        ModelFitError: If HMM fitting fails.
    """
    logger.info(
        "Starting HMM discovery for '%s' (%d rows)",
        dataset_label,
        df_windows.height,
    )

    # -- 1. Transform features -----------------------------------------
    features = pipeline.transform(df_windows)
    retained_mask = pipeline.get_retained_row_mask(df_windows)
    df_retained = df_windows.filter(pl.Series(retained_mask))

    logger.info(
        "  Preprocessing: %d -> %d rows, %d features",
        df_windows.height,
        features.shape[0],
        features.shape[1],
    )

    # -- 2. Build sequence boundaries ----------------------------------
    sequence_lengths, df_sorted = _build_sequences(df_retained)
    n_sequences = len(sequence_lengths)

    logger.info(
        "  Sequences: %d match-team pairs",
        n_sequences,
    )

    # Re-order features to match the sorted DataFrame.
    # Build a sort index from df_retained -> df_sorted.
    # We add a temporary row index, sort, and extract the mapping.
    idx_col = "__row_idx__"
    df_with_idx = df_retained.with_row_index(idx_col)
    df_sorted_idx = df_with_idx.sort(["match_id", "team_id", "start_time"])
    sort_order: np.ndarray = df_sorted_idx[idx_col].to_numpy()
    features_sorted = features[sort_order]

    # -- 3. Fit HMM ----------------------------------------------------
    hmm = HMMModel(hmm_config)

    use_gmm_init = hmm_config.init_from_gmm and gmm_model is not None

    try:
        if use_gmm_init:
            assert gmm_model is not None  # for type narrowing
            logger.info("  Initialising HMM from GMM parameters")
            hmm.fit_from_gmm(features_sorted, sequence_lengths, gmm_model)
        else:
            logger.info("  Fitting HMM from random initialisation")
            hmm.fit(features_sorted, sequence_lengths=sequence_lengths)
    except ModelFitError:
        raise
    except Exception as exc:
        msg = f"HMM discovery failed for '{dataset_label}': {exc}"
        raise ModelFitError(msg) from exc

    # -- 4. Predict state labels (Viterbi) -----------------------------
    hmm_labels_sorted = hmm.predict(features_sorted, sequence_lengths=sequence_lengths)

    # -- 5. Predict posterior probabilities -----------------------------
    hmm_proba_sorted = hmm.predict_proba(
        features_sorted, sequence_lengths=sequence_lengths
    )

    # -- 6. Add columns to sorted DataFrame ----------------------------
    n_states = hmm_config.n_states

    df_labeled = df_sorted.with_columns(
        pl.Series(name="hmm_state_label", values=hmm_labels_sorted),
    )
    prob_series: list[pl.Series] = [
        pl.Series(
            name=f"hmm_state_prob_{s}",
            values=hmm_proba_sorted[:, s],
        )
        for s in range(n_states)
    ]
    df_labeled = df_labeled.with_columns(prob_series)

    # -- 7. State agreement with GMM -----------------------------------
    ari: float | None = None
    nmi: float | None = None

    if gmm_labels is not None:
        # Re-order GMM labels to match the sorted DataFrame
        gmm_labels_sorted = gmm_labels[sort_order]
        ari, nmi = _compute_agreement(hmm_labels_sorted, gmm_labels_sorted)
        logger.info(
            "  GMM agreement: ARI=%.4f  NMI=%.4f",
            ari,
            nmi,
        )

    # -- 8. Transition matrix ------------------------------------------
    transmat = hmm.transition_matrix

    # -- 9. Flicker rate -----------------------------------------------
    flicker = _compute_flicker_rate(df_labeled, "hmm_state_label")
    logger.info("  HMM flicker rate: %.4f", flicker)

    # -- Build result --------------------------------------------------
    result = HMMDiscoveryResult(
        dataset_label=dataset_label,
        n_states=n_states,
        n_sequences=n_sequences,
        n_windows=df_windows.height,
        n_windows_retained=features.shape[0],
        transition_matrix=transmat,
        gmm_agreement_ari=ari,
        gmm_agreement_nmi=nmi,
        flicker_rate=flicker,
    )

    return df_labeled, result, hmm
