"""Run VAE-GMM hybrid discovery on each dataset, compare to direct GMM.

Loads the extracted features Parquet, the pre-computed VAE latent codes
(generated on Kaggle), and the per-dataset GMM artifacts (preprocessing
pipeline + GMM model) produced by ``run_gmm_discovery.py``, then for
each dataset:

1. Transforms features using the saved GMM preprocessing pipeline
   (for silhouette / BIC computation on the original feature space).
2. Obtains direct-GMM labels for comparison.
3. Slices the pre-computed latent codes for the current dataset.
4. Fits a GMM on the latent codes.
5. Computes comparison metrics (state agreement ARI/NMI, BIC,
   silhouette) between direct-GMM and VAE-GMM.
6. Persists latent GMM model artifacts and labelled Parquet.
7. Writes the comparison table as both Parquet and JSON.

Finally, a combined labelled Parquet and a comprehensive summary
(including the Direct-GMM vs VAE-GMM comparison table) are written
to ``data/output/vae_discovery/``.

Prerequisites:

    python scripts/run_feature_extraction.py   # features.parquet
    python scripts/run_gmm_discovery.py        # per-dataset GMM artifacts
    (Kaggle notebook)                          # latent_codes.npy

Usage::

    python scripts/run_vae_gmm_discovery.py
"""

from __future__ import annotations

import logging
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import orjson
import polars as pl

# ---------------------------------------------------------------------------
# Ensure the src package is importable when running as a standalone script.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from statsbombpy.api_client import (  # type: ignore[import-untyped]  # noqa: E402
    NoAuthWarning,
)

warnings.filterwarnings("ignore", category=NoAuthWarning)
# Suppress sklearn version mismatch from Kaggle-pickled pipeline
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from tactical.adapters import MatchInfo, StatsBombAdapter  # noqa: E402
from tactical.models.gmm import GMMModel  # noqa: E402
from tactical.models.preprocessing import PreprocessingPipeline  # noqa: E402
from tactical.models.vae_discovery import (  # noqa: E402
    VAEGMMDiscoveryResult,
    run_vae_gmm_discovery,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

CACHE_DIR = _PROJECT_ROOT / "data" / "statsbomb_cache"
FEATURES_PATH = _PROJECT_ROOT / "data" / "output" / "features.parquet"
VAE_DISCOVERY_DIR = _PROJECT_ROOT / "data" / "output" / "vae_discovery"
LATENT_CODES_PATH = VAE_DISCOVERY_DIR / "latent_codes.npy"
VAE_PIPELINE_PATH = VAE_DISCOVERY_DIR / "preprocessing_pipeline.pkl"
OUTPUT_DIR = VAE_DISCOVERY_DIR
MODELS_DIR = _PROJECT_ROOT / "data" / "models"

# Tier pattern used by the VAE notebook to drop Tier 3+ columns
_TIER_PATTERN = re.compile(r"^t(\d+)_")
_MAX_FEATURE_TIER = 2


# ------------------------------------------------------------------
# Dataset definitions (mirrors run_gmm_discovery.py)
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Dataset:
    """A single competition-season to process.

    Attributes:
        label: Human-readable label for log output.
        slug: Filesystem-safe short name for output directories.
        competition_id: StatsBomb competition identifier.
        season_id: StatsBomb season identifier.
    """

    label: str
    slug: str
    competition_id: int
    season_id: int


DATASETS: tuple[Dataset, ...] = (
    Dataset(
        "Arsenal 15-16 Premier League",
        "arsenal_1516",
        competition_id=2,
        season_id=27,
    ),
    Dataset(
        "Leverkusen 23-24 Bundesliga",
        "leverkusen_2324",
        competition_id=9,
        season_id=281,
    ),
    Dataset(
        "Arsenal Invincibles 03-04 Premier League",
        "arsenal_0304",
        competition_id=2,
        season_id=44,
    ),
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _build_match_to_dataset_map(
    adapter: StatsBombAdapter,
    datasets: tuple[Dataset, ...],
) -> dict[str, Dataset]:
    """Map each match_id to its source Dataset via the adapter.

    Args:
        adapter: Configured StatsBomb adapter with populated cache.
        datasets: Competition-seasons to look up.

    Returns:
        Dictionary from match_id (string) to Dataset.
    """
    mapping: dict[str, Dataset] = {}
    for ds in datasets:
        matches: list[MatchInfo] = adapter.list_matches(
            competition_id=ds.competition_id,
            season_id=ds.season_id,
        )
        for m in matches:
            mapping[str(m.match_id)] = ds
    return mapping


def _slugify_path(base: Path, slug: str) -> Path:
    """Return ``base / slug``, creating the directory if needed.

    Args:
        base: Parent directory.
        slug: Subdirectory name.

    Returns:
        Path to the (now-existing) subdirectory.
    """
    p = base / slug
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_gmm_best_k(model_dir: Path) -> int:
    """Read best_k from the GMM selection_result.json.

    Args:
        model_dir: Directory containing ``selection_result.json``.

    Returns:
        The best K value from GMM model selection.

    Raises:
        FileNotFoundError: If the JSON file is missing.
    """
    json_path = model_dir / "selection_result.json"
    data = orjson.loads(json_path.read_bytes())
    best_k: int = data["best_k"]
    return best_k


def _drop_tiers_above(df: pl.DataFrame, max_tier: int) -> pl.DataFrame:
    """Remove feature columns whose tier number exceeds *max_tier*.

    Non-feature columns (metadata) are always kept.

    Args:
        df: Input DataFrame.
        max_tier: Maximum feature tier to retain.

    Returns:
        DataFrame with high-tier feature columns removed.
    """
    cols_to_drop: list[str] = []
    for col in df.columns:
        m = _TIER_PATTERN.match(col)
        if m and int(m.group(1)) > max_tier:
            cols_to_drop.append(col)
    return df.drop(cols_to_drop) if cols_to_drop else df


def _result_to_json(result: VAEGMMDiscoveryResult) -> bytes:
    """Serialise a :class:`VAEGMMDiscoveryResult` to JSON bytes.

    Args:
        result: Discovery result to serialise.

    Returns:
        UTF-8 encoded JSON bytes (pretty-printed).
    """
    comparison_rows: list[dict[str, object]] = []
    for row in result.comparison.rows:
        comparison_rows.append(
            {
                "model_name": row.model_name,
                "n_states": row.n_states,
                "bic": round(row.bic, 2) if row.bic is not None else None,
                "silhouette": round(row.silhouette, 4),
                "silhouette_latent": (
                    round(row.silhouette_latent, 4)
                    if row.silhouette_latent is not None
                    else None
                ),
            }
        )

    payload: dict[str, object] = {
        "dataset_label": result.dataset_label,
        "n_states": result.n_states,
        "n_windows": result.n_windows,
        "n_windows_retained": result.n_windows_retained,
        "latent_dim": result.latent_dim,
        "bic_direct": round(result.bic_direct, 2),
        "bic_latent": round(result.bic_latent, 2),
        "silhouette_direct": round(result.silhouette_direct, 4),
        "silhouette_vae_original": round(result.silhouette_vae_original, 4),
        "silhouette_vae_latent": round(result.silhouette_vae_latent, 4),
        "agreement_ari": round(result.agreement_ari, 4),
        "agreement_nmi": round(result.agreement_nmi, 4),
        "comparison_table": comparison_rows,
    }
    return orjson.dumps(payload, option=orjson.OPT_INDENT_2)


def _format_comparison_table(results: list[VAEGMMDiscoveryResult]) -> str:
    """Build a multi-line comparison summary table string.

    Args:
        results: One result per dataset.

    Returns:
        Formatted table string ready for logging.
    """
    sep = "=" * 130
    header = (
        f"{'Dataset':<45} {'K':>3} {'BIC(direct)':>12} {'BIC(latent)':>12} "
        f"{'Sil(dir)':>9} {'Sil(vae-o)':>10} {'Sil(vae-l)':>10} "
        f"{'ARI':>7} {'NMI':>7}"
    )
    lines = [
        sep,
        "VAE-GMM HYBRID DISCOVERY: Direct-GMM vs VAE-GMM Comparison",
        sep,
        header,
        "-" * 130,
    ]
    for r in results:
        lines.append(
            f"{r.dataset_label:<45} {r.n_states:>3} "
            f"{r.bic_direct:>12.2f} {r.bic_latent:>12.2f} "
            f"{r.silhouette_direct:>9.4f} {r.silhouette_vae_original:>10.4f} "
            f"{r.silhouette_vae_latent:>10.4f} "
            f"{r.agreement_ari:>7.4f} {r.agreement_nmi:>7.4f}"
        )
    lines.append(sep)
    return "\n".join(lines)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    """Run VAE-GMM hybrid discovery on every configured dataset."""
    # -- Validate prerequisites ----------------------------------------
    if not FEATURES_PATH.exists():
        logger.error(
            "Features file not found at %s. "
            "Run `python scripts/run_feature_extraction.py` first.",
            FEATURES_PATH,
        )
        sys.exit(1)

    if not LATENT_CODES_PATH.exists():
        logger.error(
            "Pre-computed latent codes not found at %s. "
            "Run the VAE training notebook on Kaggle first.",
            LATENT_CODES_PATH,
        )
        sys.exit(1)

    if not VAE_PIPELINE_PATH.exists():
        logger.error(
            "VAE preprocessing pipeline not found at %s. "
            "Run the VAE training notebook on Kaggle first.",
            VAE_PIPELINE_PATH,
        )
        sys.exit(1)

    # -- Load features and latent codes --------------------------------
    logger.info("Loading features from %s", FEATURES_PATH)
    df_all = pl.read_parquet(FEATURES_PATH)
    logger.info("  Total rows: %d  |  Columns: %d", df_all.height, df_all.width)

    logger.info("Loading pre-computed latent codes from %s", LATENT_CODES_PATH)
    all_latent_codes = np.load(LATENT_CODES_PATH)
    logger.info(
        "  Latent codes: %d rows x %d dims",
        all_latent_codes.shape[0],
        all_latent_codes.shape[1],
    )

    logger.info("Loading VAE preprocessing pipeline from %s", VAE_PIPELINE_PATH)
    vae_pipeline = PreprocessingPipeline.load(VAE_PIPELINE_PATH)
    logger.info(
        "  VAE pipeline: %d features in, %d features out, pca=%s",
        vae_pipeline.n_features_in,
        vae_pipeline.n_features_out,
        vae_pipeline.pca_components,
    )

    # Map match_ids to datasets
    adapter = StatsBombAdapter(cache_dir=CACHE_DIR)
    match_map = _build_match_to_dataset_map(adapter, DATASETS)
    logger.info("  Mapped %d match_ids to %d datasets", len(match_map), len(DATASETS))

    # Add dataset column via the mapping
    all_match_ids = df_all["match_id"].cast(pl.Utf8).to_list()
    dataset_labels = [
        match_map[mid].label if mid in match_map else "unknown" for mid in all_match_ids
    ]
    dataset_slugs = [
        match_map[mid].slug if mid in match_map else "unknown" for mid in all_match_ids
    ]
    df_all = df_all.with_columns(
        pl.Series(name="dataset", values=dataset_labels),
        pl.Series(name="dataset_slug", values=dataset_slugs),
    )

    # Focus on window segments (the VAE was trained on these)
    df_windows = df_all.filter(pl.col("segment_type") == "window")
    logger.info(
        "  Window segments: %d  |  Possession segments: %d",
        df_windows.height,
        df_all.height - df_windows.height,
    )

    # Drop Tier 3+ feature columns to match the VAE notebook's
    # preprocessing (the VAE pipeline was fitted after this drop)
    df_windows = _drop_tiers_above(df_windows, _MAX_FEATURE_TIER)

    # Validate latent code alignment
    if all_latent_codes.shape[0] != df_windows.height:
        logger.error(
            "Latent code row count (%d) does not match window segment "
            "count (%d). The latent codes may have been generated from "
            "a different features.parquet.",
            all_latent_codes.shape[0],
            df_windows.height,
        )
        sys.exit(1)

    # -- Build per-dataset row index -----------------------------------
    # The latent codes are row-aligned with df_windows (all datasets
    # combined, in the order they appear in features.parquet).  We
    # need to slice out the rows belonging to each dataset.
    df_windows = df_windows.with_row_index("__global_idx__")

    # Per-dataset discovery
    all_labeled: list[pl.DataFrame] = []
    all_results: list[VAEGMMDiscoveryResult] = []

    for ds in DATASETS:
        df_ds = df_windows.filter(pl.col("dataset") == ds.label)
        if df_ds.height == 0:
            logger.warning("No windows found for '%s' -- skipping", ds.label)
            continue

        logger.info("")
        logger.info("=" * 72)
        logger.info("Dataset: %s (%d windows)", ds.label, df_ds.height)
        logger.info("=" * 72)

        # -- Load GMM artifacts ----------------------------------------
        gmm_model_dir = MODELS_DIR / ds.slug
        pipeline_path = gmm_model_dir / "preprocessing_pipeline.pkl"
        gmm_path = gmm_model_dir / "gmm_model.pkl"

        if not pipeline_path.exists() or not gmm_path.exists():
            logger.error(
                "GMM artifacts not found for '%s' at %s. "
                "Run `python scripts/run_gmm_discovery.py` first.",
                ds.label,
                gmm_model_dir,
            )
            continue

        gmm_pipeline = PreprocessingPipeline.load(pipeline_path)
        gmm_model = GMMModel.load(gmm_path)
        best_k = _load_gmm_best_k(gmm_model_dir)
        logger.info("  Loaded GMM artifacts (K=%d) from %s", best_k, gmm_model_dir)

        # -- Slice latent codes for this dataset -----------------------
        global_indices: np.ndarray = df_ds["__global_idx__"].to_numpy()
        ds_latent_codes = all_latent_codes[global_indices]
        logger.info(
            "  Sliced latent codes: %d rows x %d dims",
            ds_latent_codes.shape[0],
            ds_latent_codes.shape[1],
        )

        # Drop bookkeeping columns before passing to discovery
        df_input = df_ds.drop(["dataset", "dataset_slug", "__global_idx__"])

        # -- Obtain GMM labels for comparison --------------------------
        # The GMM pipeline may differ from the VAE pipeline (different
        # feature set, PCA, etc.), so we use the GMM's own pipeline
        # to produce its labels.
        gmm_features = gmm_pipeline.transform(df_input)
        gmm_labels = gmm_model.predict(gmm_features)

        # -- Run VAE-GMM hybrid discovery ------------------------------
        # The VAE pipeline is used inside run_vae_gmm_discovery to
        # produce the original feature space for silhouette/BIC of
        # the direct GMM.  The latent codes come pre-computed.
        df_labeled, result, latent_gmm = run_vae_gmm_discovery(
            df_windows=df_input,
            dataset_label=ds.label,
            pipeline=gmm_pipeline,
            latent_codes=ds_latent_codes,
            gmm_model=gmm_model,
            gmm_labels=gmm_labels,
            n_states=best_k,
        )

        # Re-attach dataset columns for combined output
        df_labeled = df_labeled.with_columns(
            pl.lit(ds.label).alias("dataset"),
            pl.lit(ds.slug).alias("dataset_slug"),
        )

        # -- Save per-dataset model artifacts --------------------------
        model_dir = _slugify_path(MODELS_DIR, ds.slug)
        latent_gmm.save(model_dir / "latent_gmm_model.pkl")
        (model_dir / "vae_gmm_result.json").write_bytes(_result_to_json(result))
        logger.info("  VAE-GMM model artifacts saved to %s", model_dir)

        # -- Save per-dataset latent codes slice -----------------------
        np.save(model_dir / "latent_codes.npy", ds_latent_codes)

        # -- Save per-dataset labelled Parquet -------------------------
        ds_output_dir = _slugify_path(OUTPUT_DIR, ds.slug)
        ds_parquet = ds_output_dir / "vae_gmm_labeled_windows.parquet"
        df_labeled.write_parquet(ds_parquet)
        logger.info("  Labelled Parquet: %s (%d rows)", ds_parquet, df_labeled.height)

        # -- Save comparison table as Parquet --------------------------
        comparison_df = result.comparison.to_polars()
        comparison_path = ds_output_dir / "comparison_table.parquet"
        comparison_df.write_parquet(comparison_path)
        logger.info("  Comparison table: %s", comparison_path)

        all_labeled.append(df_labeled)
        all_results.append(result)

    if not all_labeled:
        logger.error("No datasets produced results -- aborting.")
        sys.exit(1)

    # Combined output (diagonal concat handles varying prob columns)
    df_combined = pl.concat(all_labeled, how="diagonal_relaxed")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined_path = OUTPUT_DIR / "all_vae_gmm_labeled_windows.parquet"
    df_combined.write_parquet(combined_path)
    logger.info("")
    logger.info(
        "Combined VAE-GMM labelled Parquet: %s (%d rows)",
        combined_path,
        df_combined.height,
    )

    # Comparison summary
    summary = _format_comparison_table(all_results)
    for line in summary.split("\n"):
        logger.info(line)

    # Write combined JSON summary
    combined_json: list[dict[str, object]] = []
    for r in all_results:
        combined_json.append(orjson.loads(_result_to_json(r)))
    summary_path = OUTPUT_DIR / "vae_gmm_discovery_summary.json"
    summary_path.write_bytes(orjson.dumps(combined_json, option=orjson.OPT_INDENT_2))
    logger.info("JSON summary: %s", summary_path)


if __name__ == "__main__":
    main()
