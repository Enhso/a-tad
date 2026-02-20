"""Run GMM discovery on each dataset: select optimal K, assign state labels.

Loads the extracted features Parquet, maps matches back to their
source datasets via the StatsBomb adapter cache, then for each
dataset:

1. Filters to window segments and drops Tier 3 feature columns.
2. Runs the full GMM discovery pipeline (preprocessing, model
   selection, fitting, labelling, temporal smoothing).
3. Persists model artifacts (preprocessing pipeline + GMM model).
4. Writes a labelled Parquet with ``state_label`` and
   ``state_label_smoothed`` columns.

Finally, a combined labelled Parquet spanning all datasets is written
and a comprehensive summary is logged.

Usage::

    python scripts/run_gmm_discovery.py
"""

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

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

from tactical.adapters import MatchInfo, StatsBombAdapter  # noqa: E402
from tactical.models.discovery import (  # noqa: E402
    DatasetDiscoveryResult,
    run_gmm_discovery,
)
from tactical.models.gmm import GMMConfig  # noqa: E402

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
OUTPUT_DIR = _PROJECT_ROOT / "data" / "output" / "gmm_discovery"
MODELS_DIR = _PROJECT_ROOT / "data" / "models"


# ------------------------------------------------------------------
# Dataset definitions (mirrors download_data.py)
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
# GMM configuration
# ------------------------------------------------------------------

GMM_CONFIG = GMMConfig(
    k_min=3,
    k_max=12,
    covariance_type="full",
    n_init=20,
    random_state=42,
)

FEATURE_PREFIX = "t"
NULL_STRATEGY = "impute_median"
PCA_VARIANCE = 0.95
MAX_FEATURE_TIER = 2
SMOOTH_WINDOW = 5


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


def _result_to_json(result: DatasetDiscoveryResult) -> bytes:
    """Serialise a :class:`DatasetDiscoveryResult` to JSON bytes.

    Args:
        result: Discovery result to serialise.

    Returns:
        UTF-8 encoded JSON bytes (pretty-printed).
    """
    payload = {
        "dataset_label": result.dataset_label,
        "best_k": result.best_k,
        "n_windows": result.n_windows,
        "n_windows_retained": result.n_windows_retained,
        "n_features_in": result.n_features_in,
        "n_features_out": result.n_features_out,
        "flicker_rate_raw": round(result.flicker_rate_raw, 4),
        "flicker_rate_smoothed": round(result.flicker_rate_smoothed, 4),
        "selection": {
            "k_values": list(result.selection.k_values),
            "bic_scores": [round(s, 2) for s in result.selection.bic_scores],
            "aic_scores": [round(s, 2) for s in result.selection.aic_scores],
            "silhouette_scores": [
                round(s, 4) for s in result.selection.silhouette_scores
            ],
            "best_k_bic": result.selection.best_k_bic,
            "best_k_silhouette": result.selection.best_k_silhouette,
        },
    }
    return orjson.dumps(payload, option=orjson.OPT_INDENT_2)


def _format_summary_table(
    results: list[DatasetDiscoveryResult],
) -> str:
    """Build a multi-line summary table string.

    Args:
        results: One result per dataset.

    Returns:
        Formatted table string ready for logging.
    """
    sep = "=" * 100
    header = (
        f"{'Dataset':<45} {'K':>3} {'Windows':>8} {'Retained':>9} "
        f"{'FeatIn':>7} {'FeatOut':>8} {'FlickRaw':>9} {'FlickSm':>8}"
    )
    lines = [sep, "GMM DISCOVERY SUMMARY", sep, header, "-" * 100]
    for r in results:
        lines.append(
            f"{r.dataset_label:<45} {r.best_k:>3} {r.n_windows:>8} "
            f"{r.n_windows_retained:>9} {r.n_features_in:>7} "
            f"{r.n_features_out:>8} {r.flicker_rate_raw:>9.4f} "
            f"{r.flicker_rate_smoothed:>8.4f}"
        )
    lines.append(sep)
    return "\n".join(lines)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    """Run GMM discovery on every configured dataset."""
    if not FEATURES_PATH.exists():
        logger.error(
            "Features file not found at %s. "
            "Run `python scripts/run_feature_extraction.py` first.",
            FEATURES_PATH,
        )
        sys.exit(1)

    logger.info("Loading features from %s", FEATURES_PATH)
    df_all = pl.read_parquet(FEATURES_PATH)
    logger.info("  Total rows: %d  |  Columns: %d", df_all.height, df_all.width)

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

    # Focus on window segments
    df_windows = df_all.filter(pl.col("segment_type") == "window")
    logger.info(
        "  Window segments: %d  |  Possession segments: %d",
        df_windows.height,
        df_all.height - df_windows.height,
    )

    # Per-dataset discovery
    all_labeled: list[pl.DataFrame] = []
    all_results: list[DatasetDiscoveryResult] = []

    for ds in DATASETS:
        df_ds = df_windows.filter(pl.col("dataset") == ds.label)
        if df_ds.height == 0:
            logger.warning("No windows found for '%s' -- skipping", ds.label)
            continue

        logger.info("")
        logger.info("=" * 72)
        logger.info("Dataset: %s (%d windows)", ds.label, df_ds.height)
        logger.info("=" * 72)

        # Drop dataset bookkeeping columns before discovery
        df_input = df_ds.drop(["dataset", "dataset_slug"])

        df_labeled, result, pipeline, model = run_gmm_discovery(
            df_windows=df_input,
            dataset_label=ds.label,
            gmm_config=GMM_CONFIG,
            feature_prefix=FEATURE_PREFIX,
            null_strategy=NULL_STRATEGY,
            pca_variance_threshold=PCA_VARIANCE,
            max_feature_tier=MAX_FEATURE_TIER,
            smooth_window=SMOOTH_WINDOW,
        )

        # Re-attach dataset columns for combined output
        df_labeled = df_labeled.with_columns(
            pl.lit(ds.label).alias("dataset"),
            pl.lit(ds.slug).alias("dataset_slug"),
        )

        # Save per-dataset artifacts
        model_dir = _slugify_path(MODELS_DIR, ds.slug)
        pipeline.save(model_dir / "preprocessing_pipeline.pkl")
        model.save(model_dir / "gmm_model.pkl")
        (model_dir / "selection_result.json").write_bytes(_result_to_json(result))
        logger.info("  Model artifacts saved to %s", model_dir)

        # Save per-dataset labelled Parquet
        ds_output_dir = _slugify_path(OUTPUT_DIR, ds.slug)
        ds_parquet = ds_output_dir / "labeled_windows.parquet"
        df_labeled.write_parquet(ds_parquet)
        logger.info("  Labelled Parquet: %s (%d rows)", ds_parquet, df_labeled.height)

        all_labeled.append(df_labeled)
        all_results.append(result)

    if not all_labeled:
        logger.error("No datasets produced results -- aborting.")
        sys.exit(1)

    # Combined output (use diagonal concat to handle varying state_prob columns)
    df_combined = pl.concat(all_labeled, how="diagonal_relaxed")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined_path = OUTPUT_DIR / "all_labeled_windows.parquet"
    df_combined.write_parquet(combined_path)
    logger.info("")
    logger.info(
        "Combined labelled Parquet: %s (%d rows)", combined_path, df_combined.height
    )

    # Summary
    summary = _format_summary_table(all_results)
    for line in summary.split("\n"):
        logger.info(line)

    # Write combined JSON summary
    combined_json: list[dict[str, object]] = []
    for r in all_results:
        combined_json.append(orjson.loads(_result_to_json(r)))
    summary_path = OUTPUT_DIR / "discovery_summary.json"
    summary_path.write_bytes(orjson.dumps(combined_json, option=orjson.OPT_INDENT_2))
    logger.info("JSON summary: %s", summary_path)


if __name__ == "__main__":
    main()
