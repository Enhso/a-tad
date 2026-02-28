"""VAE-GMM hybrid discovery and comparison with direct GMM.

Provides :func:`run_vae_gmm_discovery`, the single entry-point that
takes pre-computed VAE latent codes, a polars DataFrame of window
segments for **one** dataset, an already-fitted
:class:`PreprocessingPipeline`, and an already-fitted direct
:class:`GMMModel`, then returns:

* a labelled copy of the DataFrame (with ``vae_gmm_state_label``),
* a frozen :class:`VAEGMMDiscoveryResult` summary (including BIC,
  silhouette, and agreement metrics),
* a :class:`ComparisonTable` summarising direct-GMM vs VAE-GMM.

The hybrid approach takes pre-trained VAE latent codes (mu vectors)
and fits a GMM on the latent space.  Comparison metrics (state
agreement via ARI/NMI, BIC, silhouette) are computed between the
direct-GMM and VAE-GMM assignments.

Latent codes are expected to have been generated externally (e.g.
on a GPU-equipped Kaggle instance) and saved as ``.npy`` files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.metrics import (  # type: ignore[import-untyped]
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from tactical.exceptions import ModelFitError
from tactical.models.gmm import GMMConfig, GMMModel

if TYPE_CHECKING:
    from tactical.models.preprocessing import PreprocessingPipeline

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Result containers
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    """A single row in the direct-GMM vs VAE-GMM comparison table.

    Attributes:
        model_name: Identifier for the model variant.
        n_states: Number of clusters / states.
        bic: Bayesian Information Criterion (lower is better).
            ``None`` when BIC is not computable.
        silhouette: Silhouette score in the **original** feature
            space (higher is better, range -1 to 1).
        silhouette_latent: Silhouette score in the **latent**
            feature space. ``None`` for direct-GMM.
    """

    model_name: str
    n_states: int
    bic: float | None
    silhouette: float
    silhouette_latent: float | None


@dataclass(frozen=True, slots=True)
class ComparisonTable:
    """Full comparison between direct-GMM and VAE-GMM.

    Attributes:
        rows: Per-model metric rows.
        agreement_ari: Adjusted Rand Index between direct-GMM and
            VAE-GMM state labels.
        agreement_nmi: Normalised Mutual Information between
            direct-GMM and VAE-GMM state labels.
    """

    rows: tuple[ComparisonRow, ...]
    agreement_ari: float
    agreement_nmi: float

    def to_polars(self) -> pl.DataFrame:
        """Render the comparison as a polars DataFrame.

        Returns:
            DataFrame with one row per model and columns for each
            metric, plus agreement columns.
        """
        records: list[dict[str, object]] = []
        for row in self.rows:
            records.append(
                {
                    "model": row.model_name,
                    "n_states": row.n_states,
                    "bic": row.bic,
                    "silhouette": round(row.silhouette, 4),
                    "silhouette_latent": (
                        round(row.silhouette_latent, 4)
                        if row.silhouette_latent is not None
                        else None
                    ),
                }
            )
        df = pl.DataFrame(records)
        return df.with_columns(
            pl.lit(round(self.agreement_ari, 4)).alias("agreement_ari"),
            pl.lit(round(self.agreement_nmi, 4)).alias("agreement_nmi"),
        )


@dataclass(frozen=True, slots=True)
class VAEGMMDiscoveryResult:
    """Summary produced by :func:`run_vae_gmm_discovery`.

    Attributes:
        dataset_label: Human-readable identifier for the dataset.
        n_states: Number of GMM clusters fitted on latent codes.
        n_windows: Total window rows in the input DataFrame.
        n_windows_retained: Rows surviving preprocessing.
        latent_dim: Dimensionality of the VAE latent space.
        bic_direct: BIC of the direct-GMM on original features.
        bic_latent: BIC of the VAE-GMM on latent features.
        silhouette_direct: Silhouette of direct-GMM labels on
            original features.
        silhouette_vae_original: Silhouette of VAE-GMM labels on
            original features.
        silhouette_vae_latent: Silhouette of VAE-GMM labels on
            latent features.
        agreement_ari: Adjusted Rand Index between the two models.
        agreement_nmi: Normalised Mutual Information.
        comparison: Full :class:`ComparisonTable`.
    """

    dataset_label: str
    n_states: int
    n_windows: int
    n_windows_retained: int
    latent_dim: int
    bic_direct: float
    bic_latent: float
    silhouette_direct: float
    silhouette_vae_original: float
    silhouette_vae_latent: float
    agreement_ari: float
    agreement_nmi: float
    comparison: ComparisonTable


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _safe_silhouette(
    features: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute silhouette score, returning -1.0 for degenerate cases.

    Args:
        features: Array of shape ``(n_samples, n_features)``.
        labels: Integer cluster labels of shape ``(n_samples,)``.

    Returns:
        Silhouette score in ``[-1, 1]``, or ``-1.0`` if fewer than
        2 unique labels exist.
    """
    n_unique = len(np.unique(labels))
    if n_unique < 2 or n_unique >= features.shape[0]:
        return -1.0
    return float(silhouette_score(features, labels))


def _compute_agreement(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> tuple[float, float]:
    """Compute ARI and NMI between two label vectors.

    Args:
        labels_a: First set of integer labels.
        labels_b: Second set of integer labels.

    Returns:
        Tuple of ``(adjusted_rand_index, normalised_mutual_info)``.
    """
    ari = float(adjusted_rand_score(labels_a, labels_b))
    nmi = float(normalized_mutual_info_score(labels_a, labels_b))
    return ari, nmi


# ------------------------------------------------------------------
# Main entry-point
# ------------------------------------------------------------------


def run_vae_gmm_discovery(
    df_windows: pl.DataFrame,
    dataset_label: str,
    pipeline: PreprocessingPipeline,
    latent_codes: np.ndarray,
    gmm_model: GMMModel,
    gmm_labels: np.ndarray,
    n_states: int | None = None,
    gmm_latent_config: GMMConfig | None = None,
) -> tuple[pl.DataFrame, VAEGMMDiscoveryResult, GMMModel]:
    """Run VAE-GMM hybrid discovery and compare with direct GMM.

    Uses **pre-computed** VAE latent codes (generated externally,
    e.g. on a GPU-equipped Kaggle instance) rather than training
    a VAE locally.

    Steps performed:

    1. Transform features using the fitted preprocessing pipeline
       (for silhouette / BIC on the original feature space).
    2. Fit GMM on the pre-computed latent codes.
    3. Predict VAE-GMM labels from latent GMM.
    4. Compute silhouette on both original and latent features.
    5. Compute BIC for both direct and latent GMMs.
    6. Compute state agreement (ARI, NMI) between direct and hybrid.
    7. Build comparison table.

    Args:
        df_windows: polars DataFrame of window segments for one
            dataset.  Must contain metadata columns (``match_id``,
            ``team_id``, ``start_time``) and feature columns.
        dataset_label: Human-readable name for logging / results.
        pipeline: A **fitted** :class:`PreprocessingPipeline`.
        latent_codes: Pre-computed VAE latent codes (mu vectors)
            aligned row-for-row with the retained rows of
            *df_windows* after preprocessing.  Shape
            ``(n_retained, latent_dim)``.
        gmm_model: A **fitted** direct GMM model for comparison.
        gmm_labels: Direct-GMM state labels aligned to the
            retained rows of *df_windows* (after preprocessing).
        n_states: Number of clusters for the latent GMM.  Defaults
            to the direct GMM's ``n_states``.
        gmm_latent_config: Optional GMM config for the latent-space
            GMM.  Defaults to a config matching *n_states* with
            ``k_min == k_max == n_states``.

    Returns:
        A 3-tuple of:

        * **df_labeled** -- copy of retained rows with added
          ``vae_gmm_state_label`` and per-state probability columns
          (``vae_gmm_state_prob_0``, ...).
        * **result** -- :class:`VAEGMMDiscoveryResult` summary.
        * **latent_gmm** -- the fitted GMM on latent codes.

    Raises:
        ValueError: If *df_windows* is empty, no rows remain after
            preprocessing, or *latent_codes* row count does not
            match the retained row count.
        ModelFitError: If latent GMM fitting fails.
    """
    logger.info(
        "Starting VAE-GMM discovery for '%s' (%d rows)",
        dataset_label,
        df_windows.height,
    )

    # -- 1. Transform features -----------------------------------------
    features = pipeline.transform(df_windows)
    retained_mask = pipeline.get_retained_row_mask(df_windows)
    df_retained = df_windows.filter(pl.Series(retained_mask))

    n_retained = features.shape[0]
    logger.info(
        "  Preprocessing: %d -> %d rows, %d features",
        df_windows.height,
        n_retained,
        features.shape[1],
    )

    if n_retained == 0:
        msg = "No rows remain after preprocessing."
        raise ValueError(msg)

    # -- Validate latent codes alignment -------------------------------
    if latent_codes.shape[0] != n_retained:
        msg = (
            f"Latent codes row count ({latent_codes.shape[0]}) does not "
            f"match retained row count ({n_retained})."
        )
        raise ValueError(msg)

    latent_dim = latent_codes.shape[1]
    logger.info(
        "  Latent codes: (%d, %d)",
        latent_codes.shape[0],
        latent_dim,
    )

    # Determine K
    k = n_states if n_states is not None else gmm_model.n_states

    # -- 2. Fit GMM on latent codes ------------------------------------
    if gmm_latent_config is None:
        gmm_latent_config = GMMConfig(
            k_min=k,
            k_max=k,
            covariance_type="full",
            n_init=20,
            random_state=42,
        )

    latent_gmm = GMMModel(gmm_latent_config)
    try:
        latent_gmm._fit_k(latent_codes, k)
    except ModelFitError:
        raise
    except Exception as exc:
        msg = f"Latent GMM fitting failed for '{dataset_label}': {exc}"
        raise ModelFitError(msg) from exc

    # -- 3. Predict VAE-GMM labels -------------------------------------
    vae_gmm_labels = latent_gmm.predict(latent_codes)
    vae_gmm_proba = latent_gmm.predict_proba(latent_codes)

    # -- 4. Silhouette scores ------------------------------------------
    sil_direct = _safe_silhouette(features, gmm_labels)
    sil_vae_orig = _safe_silhouette(features, vae_gmm_labels)
    sil_vae_latent = _safe_silhouette(latent_codes, vae_gmm_labels)

    logger.info(
        "  Silhouette -- direct-GMM: %.4f  VAE-GMM(orig): %.4f  VAE-GMM(latent): %.4f",
        sil_direct,
        sil_vae_orig,
        sil_vae_latent,
    )

    # -- 5. BIC scores -------------------------------------------------
    direct_sklearn = gmm_model.sklearn_model
    assert direct_sklearn is not None
    bic_direct = float(direct_sklearn.bic(features))

    latent_sklearn = latent_gmm.sklearn_model
    assert latent_sklearn is not None
    bic_latent = float(latent_sklearn.bic(latent_codes))

    logger.info(
        "  BIC -- direct-GMM: %.2f  VAE-GMM(latent): %.2f",
        bic_direct,
        bic_latent,
    )

    # -- 6. State agreement --------------------------------------------
    ari, nmi = _compute_agreement(gmm_labels, vae_gmm_labels)

    logger.info(
        "  Agreement -- ARI: %.4f  NMI: %.4f",
        ari,
        nmi,
    )

    # -- 7. Build comparison table -------------------------------------
    direct_row = ComparisonRow(
        model_name="Direct-GMM",
        n_states=gmm_model.n_states,
        bic=bic_direct,
        silhouette=sil_direct,
        silhouette_latent=None,
    )
    vae_row = ComparisonRow(
        model_name="VAE-GMM",
        n_states=k,
        bic=bic_latent,
        silhouette=sil_vae_orig,
        silhouette_latent=sil_vae_latent,
    )
    comparison = ComparisonTable(
        rows=(direct_row, vae_row),
        agreement_ari=ari,
        agreement_nmi=nmi,
    )

    # -- 8. Label DataFrame --------------------------------------------
    df_labeled = df_retained.with_columns(
        pl.Series(name="vae_gmm_state_label", values=vae_gmm_labels),
    )
    prob_series: list[pl.Series] = [
        pl.Series(
            name=f"vae_gmm_state_prob_{s}",
            values=vae_gmm_proba[:, s],
        )
        for s in range(k)
    ]
    df_labeled = df_labeled.with_columns(prob_series)

    # -- Build result --------------------------------------------------
    result = VAEGMMDiscoveryResult(
        dataset_label=dataset_label,
        n_states=k,
        n_windows=df_windows.height,
        n_windows_retained=n_retained,
        latent_dim=latent_dim,
        bic_direct=bic_direct,
        bic_latent=bic_latent,
        silhouette_direct=sil_direct,
        silhouette_vae_original=sil_vae_orig,
        silhouette_vae_latent=sil_vae_latent,
        agreement_ari=ari,
        agreement_nmi=nmi,
        comparison=comparison,
    )

    return df_labeled, result, latent_gmm
