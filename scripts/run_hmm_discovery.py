"""Run HMM discovery on each dataset, compare state assignments vs GMM.

Loads the extracted features Parquet and the per-dataset GMM artifacts
(preprocessing pipeline + GMM model) produced by ``run_gmm_discovery.py``,
then for each dataset:

1. Transforms features using the saved preprocessing pipeline.
2. Builds per-match-team sequence lengths.
3. Fits an HMM initialised from the GMM cluster parameters.
4. Predicts HMM state labels via Viterbi decoding.
5. Computes state agreement with GMM (ARI, NMI).
6. Extracts and visualises the transition matrix as a heatmap.
7. Persists HMM model artifacts and labelled Parquet.

Finally, a combined labelled Parquet and a comprehensive summary
(including agreement metrics and transition matrices) are written.

Usage::

    python scripts/run_hmm_discovery.py
"""

from __future__ import annotations

import logging
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

from tactical.adapters import MatchInfo, StatsBombAdapter  # noqa: E402
from tactical.config import HMMConfig  # noqa: E402
from tactical.models.gmm import GMMModel  # noqa: E402
from tactical.models.hmm_discovery import (  # noqa: E402
    HMMDiscoveryResult,
    plot_transition_matrix,
    run_hmm_discovery,
)
from tactical.models.preprocessing import PreprocessingPipeline  # noqa: E402

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
GMM_OUTPUT_DIR = _PROJECT_ROOT / "data" / "output" / "gmm_discovery"
OUTPUT_DIR = _PROJECT_ROOT / "data" / "output" / "hmm_discovery"
MODELS_DIR = _PROJECT_ROOT / "data" / "models"


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
# HMM configuration
# ------------------------------------------------------------------

# n_states will be overridden per-dataset from the GMM's best_k
BASE_HMM_CONFIG = HMMConfig(
    n_states=6,
    covariance_type="full",
    n_iter=100,
    random_state=42,
    init_from_gmm=True,
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


def _result_to_json(result: HMMDiscoveryResult) -> bytes:
    """Serialise an :class:`HMMDiscoveryResult` to JSON bytes.

    Args:
        result: Discovery result to serialise.

    Returns:
        UTF-8 encoded JSON bytes (pretty-printed).
    """
    payload: dict[str, object] = {
        "dataset_label": result.dataset_label,
        "n_states": result.n_states,
        "n_sequences": result.n_sequences,
        "n_windows": result.n_windows,
        "n_windows_retained": result.n_windows_retained,
        "transition_matrix": [
            [round(float(v), 4) for v in row] for row in result.transition_matrix
        ],
        "gmm_agreement_ari": (
            round(result.gmm_agreement_ari, 4)
            if result.gmm_agreement_ari is not None
            else None
        ),
        "gmm_agreement_nmi": (
            round(result.gmm_agreement_nmi, 4)
            if result.gmm_agreement_nmi is not None
            else None
        ),
        "flicker_rate": round(result.flicker_rate, 4),
    }
    return orjson.dumps(payload, option=orjson.OPT_INDENT_2)


def _format_summary_table(results: list[HMMDiscoveryResult]) -> str:
    """Build a multi-line summary table string.

    Args:
        results: One result per dataset.

    Returns:
        Formatted table string ready for logging.
    """
    sep = "=" * 110
    header = (
        f"{'Dataset':<45} {'K':>3} {'Seqs':>5} {'Windows':>8} "
        f"{'Retained':>9} {'ARI':>7} {'NMI':>7} {'Flicker':>8}"
    )
    lines = [sep, "HMM DISCOVERY SUMMARY", sep, header, "-" * 110]
    for r in results:
        ari_str = (
            f"{r.gmm_agreement_ari:.4f}" if r.gmm_agreement_ari is not None else "N/A"
        )
        nmi_str = (
            f"{r.gmm_agreement_nmi:.4f}" if r.gmm_agreement_nmi is not None else "N/A"
        )
        lines.append(
            f"{r.dataset_label:<45} {r.n_states:>3} {r.n_sequences:>5} "
            f"{r.n_windows:>8} {r.n_windows_retained:>9} "
            f"{ari_str:>7} {nmi_str:>7} {r.flicker_rate:>8.4f}"
        )
    lines.append(sep)
    return "\n".join(lines)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    """Run HMM discovery on every configured dataset."""
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
    all_results: list[HMMDiscoveryResult] = []

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

        pipeline = PreprocessingPipeline.load(pipeline_path)
        gmm_model = GMMModel.load(gmm_path)
        best_k = _load_gmm_best_k(gmm_model_dir)
        logger.info("  Loaded GMM artifacts (K=%d) from %s", best_k, gmm_model_dir)

        # -- Build HMM config with GMM's best K -----------------------
        hmm_config = HMMConfig(
            n_states=best_k,
            covariance_type=BASE_HMM_CONFIG.covariance_type,
            n_iter=BASE_HMM_CONFIG.n_iter,
            random_state=BASE_HMM_CONFIG.random_state,
            init_from_gmm=BASE_HMM_CONFIG.init_from_gmm,
        )

        # Drop dataset bookkeeping columns before discovery
        df_input = df_ds.drop(["dataset", "dataset_slug"])

        # -- Obtain GMM labels for comparison --------------------------
        features = pipeline.transform(df_input)
        gmm_labels = gmm_model.predict(features)

        # -- Run HMM discovery -----------------------------------------
        df_labeled, result, hmm_model = run_hmm_discovery(
            df_windows=df_input,
            dataset_label=ds.label,
            pipeline=pipeline,
            hmm_config=hmm_config,
            gmm_model=gmm_model,
            gmm_labels=gmm_labels,
        )

        # Re-attach dataset columns for combined output
        df_labeled = df_labeled.with_columns(
            pl.lit(ds.label).alias("dataset"),
            pl.lit(ds.slug).alias("dataset_slug"),
        )

        # -- Save per-dataset artifacts --------------------------------
        hmm_model_dir = _slugify_path(MODELS_DIR, ds.slug)
        hmm_model.save(hmm_model_dir / "hmm_model.pkl")
        (hmm_model_dir / "hmm_result.json").write_bytes(_result_to_json(result))
        logger.info("  HMM model artifacts saved to %s", hmm_model_dir)

        # Transition matrix as npy for programmatic access
        np.save(hmm_model_dir / "transition_matrix.npy", result.transition_matrix)

        # -- Visualise transition matrix -------------------------------
        ds_output_dir = _slugify_path(OUTPUT_DIR, ds.slug)
        fig_path = ds_output_dir / "transition_matrix.png"
        plot_transition_matrix(
            result.transition_matrix,
            title=f"HMM Transition Matrix -- {ds.label} (K={best_k})",
            save_path=fig_path,
        )

        # -- Save labelled Parquet -------------------------------------
        ds_parquet = ds_output_dir / "hmm_labeled_windows.parquet"
        df_labeled.write_parquet(ds_parquet)
        logger.info("  Labelled Parquet: %s (%d rows)", ds_parquet, df_labeled.height)

        all_labeled.append(df_labeled)
        all_results.append(result)

    if not all_labeled:
        logger.error("No datasets produced results -- aborting.")
        sys.exit(1)

    # Combined output (diagonal concat handles varying prob columns)
    df_combined = pl.concat(all_labeled, how="diagonal_relaxed")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined_path = OUTPUT_DIR / "all_hmm_labeled_windows.parquet"
    df_combined.write_parquet(combined_path)
    logger.info("")
    logger.info(
        "Combined HMM labelled Parquet: %s (%d rows)",
        combined_path,
        df_combined.height,
    )

    # Summary
    summary = _format_summary_table(all_results)
    for line in summary.split("\n"):
        logger.info(line)

    # Write combined JSON summary
    combined_json: list[dict[str, object]] = []
    for r in all_results:
        combined_json.append(orjson.loads(_result_to_json(r)))
    summary_path = OUTPUT_DIR / "hmm_discovery_summary.json"
    summary_path.write_bytes(orjson.dumps(combined_json, option=orjson.OPT_INDENT_2))
    logger.info("JSON summary: %s", summary_path)


if __name__ == "__main__":
    main()
