"""Model selection utilities for tactical state discovery.

Provides tools for selecting the optimal number of clusters (*K*) for
Gaussian Mixture Models by evaluating BIC, AIC, and silhouette scores
across a range of candidate *K* values.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import silhouette_score  # type: ignore[import-untyped]
from tqdm import tqdm

from tactical.exceptions import ModelFitError
from tactical.models.gmm import GMMConfig, GMMModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelSelectionResult:
    """Results from model selection over a range of K values.

    Attributes:
        k_values: Candidate K values that were evaluated.
        bic_scores: BIC score for each K (lower is better).
        aic_scores: AIC score for each K (lower is better).
        silhouette_scores: Silhouette score for each K (higher
            is better, range -1 to 1).
        best_k_bic: K that minimises BIC.
        best_k_silhouette: K that maximises silhouette score.
    """

    k_values: tuple[int, ...]
    bic_scores: tuple[float, ...]
    aic_scores: tuple[float, ...]
    silhouette_scores: tuple[float, ...]
    best_k_bic: int
    best_k_silhouette: int


def select_gmm_k(
    features: np.ndarray,
    config: GMMConfig,
) -> ModelSelectionResult:
    """Run GMM model selection over ``k_min`` to ``k_max``.

    For each candidate *K*, a :class:`GMMModel` is fitted and three
    metrics are recorded:

    * **BIC** -- Bayesian Information Criterion (lower is better).
    * **AIC** -- Akaike Information Criterion (lower is better).
    * **Silhouette score** -- cluster separation quality (higher is
      better).

    Progress is tracked with :mod:`tqdm`.

    Args:
        features: Array of shape ``(n_samples, n_features)``.
        config: GMM configuration with ``k_min``, ``k_max``, and
            other hyperparameters.

    Returns:
        :class:`ModelSelectionResult` with per-K scores and the
        best K according to BIC and silhouette.

    Raises:
        ModelFitError: If fitting fails for every candidate K.
        ValueError: If ``k_min > k_max`` or ``features`` has
            fewer samples than ``k_max``.
    """
    if config.k_min > config.k_max:
        msg = f"k_min ({config.k_min}) must be <= k_max ({config.k_max})."
        raise ValueError(msg)

    if features.shape[0] < config.k_max:
        msg = f"Not enough samples ({features.shape[0]}) for k_max={config.k_max}."
        raise ValueError(msg)

    k_values: list[int] = []
    bic_scores: list[float] = []
    aic_scores: list[float] = []
    sil_scores: list[float] = []

    k_range = range(config.k_min, config.k_max + 1)

    for k in tqdm(k_range, desc="GMM model selection"):
        model = GMMModel(config)
        try:
            model._fit_k(features, k)
        except ModelFitError:
            logger.error("GMM fitting failed for k=%d, skipping.", k)
            continue

        sklearn_model = model.sklearn_model
        assert sklearn_model is not None  # guarded by _fit_k success

        bic = float(sklearn_model.bic(features))
        aic = float(sklearn_model.aic(features))

        labels = model.predict(features)
        n_unique = len(np.unique(labels))

        # Silhouette undefined for < 2 clusters
        sil = -1.0 if n_unique < 2 else float(silhouette_score(features, labels))

        k_values.append(k)
        bic_scores.append(bic)
        aic_scores.append(aic)
        sil_scores.append(sil)

    if not k_values:
        msg = "GMM fitting failed for all candidate K values."
        raise ModelFitError(msg)

    best_k_bic = k_values[int(np.argmin(bic_scores))]
    best_k_sil = k_values[int(np.argmax(sil_scores))]

    return ModelSelectionResult(
        k_values=tuple(k_values),
        bic_scores=tuple(bic_scores),
        aic_scores=tuple(aic_scores),
        silhouette_scores=tuple(sil_scores),
        best_k_bic=best_k_bic,
        best_k_silhouette=best_k_sil,
    )
