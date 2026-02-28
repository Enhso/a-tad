"""Run changepoint detection on state sequences and feature time series.

Loads labelled window outputs from GMM, HMM, and VAE discovery stages
plus the VAE latent codes, then runs PELT changepoint detection on four
signal types per (match_id, team_id) group:

1. **GMM state probabilities** (12-D soft assignments)
2. **HMM state probabilities** (12-D soft assignments)
3. **VAE latent codes** (16-D z_t time series) -- preferred per SPEC
4. **Selected raw features** (key tactical features, median-imputed)

For each signal type the script:

- Groups by (match_id, team_id), sorts windows by match_minute.
- Runs :func:`detect_changepoints` with configurable penalty.
- Collects :class:`ChangepointResult` objects with timestamps and
  per-changepoint feature deltas.
- Persists results to Parquet (one row per changepoint) and a
  compact JSON summary.

Usage::

    python scripts/run_changepoint_detection.py
"""

from __future__ import annotations

import logging
import sys
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

from tactical.config import ChangepointConfig  # noqa: E402
from tactical.models.changepoint import (  # noqa: E402
    ChangepointResult,
    detect_changepoints,
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

GMM_PATH = (
    _PROJECT_ROOT / "data" / "output" / "gmm_discovery" / "all_labeled_windows.parquet"
)
HMM_PATH = (
    _PROJECT_ROOT
    / "data"
    / "output"
    / "hmm_discovery"
    / "all_hmm_labeled_windows.parquet"
)
VAE_PATH = (
    _PROJECT_ROOT
    / "data"
    / "output"
    / "vae_discovery"
    / "all_vae_gmm_labeled_windows.parquet"
)
LATENT_PATH = _PROJECT_ROOT / "data" / "output" / "vae_discovery" / "latent_codes.npy"
OUTPUT_DIR = _PROJECT_ROOT / "data" / "output" / "changepoint_detection"

# ------------------------------------------------------------------
# Changepoint configuration
# ------------------------------------------------------------------

# Per-signal-type penalties calibrated to yield ~3-8 changepoints per
# (match, team) group (~400 windows each).  The RBF kernel cost is
# used throughout; penalty magnitude controls sensitivity.
#
# Calibration (sampled over 50 groups):
#   GMM  state probs  pen=1.5  -> median 4 CPs  (range 2-13)
#   HMM  state probs  pen=1.5  -> similar profile
#   VAE  latent codes  pen=2.0  -> median 5 CPs  (range 1-10)
#   Raw  features      pen=1.5  -> median 4 CPs  (range 1-10)

_BASE_CONFIG = ChangepointConfig(
    method="pelt",
    model="rbf",
    penalty=1.5,
    min_segment_size=10,
)

# VAE latent space is 16-D (vs 12-D state probs) and captures more
# subtle variation, so it needs a slightly higher penalty to avoid
# over-segmentation.
_VAE_CONFIG = ChangepointConfig(
    method="pelt",
    model="rbf",
    penalty=2.0,
    min_segment_size=10,
)

# Raw tactical features are noisier and lower-dimensional than model
# outputs, producing many spurious changepoints at the base penalty.
# A moderately higher penalty keeps median CPs in the 3-6 range.
_RAW_CONFIG = ChangepointConfig(
    method="pelt",
    model="rbf",
    penalty=2.0,
    min_segment_size=10,
)

GROUP_BY_COLUMNS = ["match_id", "team_id"]

# Key raw features: low-null, tactically meaningful columns from
# Tier 1 and Tier 2 that capture pressing, territorial, and passing
# behaviour without requiring imputation for the vast majority of
# windows.
RAW_FEATURE_COLUMNS: list[str] = [
    "t1_pass_count",
    "t1_carry_count",
    "t1_defend_pressure_count",
    "t1_defend_interception_count",
    "t1_context_score_differential",
    "t1_context_possession_share",
    "t2_shape_team_centroid_x_est",
    "t2_shape_team_spread_est",
    "t2_shape_compactness_proxy",
    "t2_press_intensity",
]


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """Specification for a single signal type to run detection on.

    Attributes:
        name: Human-readable label (used in logs and output).
        slug: Filesystem-safe identifier for output files.
        signal_columns: Column names in the DataFrame to use as
            the signal matrix.
    """

    name: str
    slug: str
    signal_columns: list[str]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_group_key(row: dict[str, object]) -> str:
    """Join group-by column values into a single string key.

    Args:
        row: Dictionary with group-by column values.

    Returns:
        Underscore-separated string of column values.
    """
    return "_".join(str(row[c]) for c in GROUP_BY_COLUMNS)


def _run_detection_on_signal(
    df: pl.DataFrame,
    spec: SignalSpec,
    config: ChangepointConfig,
) -> list[ChangepointResult]:
    """Run changepoint detection per group for one signal type.

    Groups *df* by :data:`GROUP_BY_COLUMNS`, sorts each group by
    ``match_minute``, imputes nulls with column medians, and calls
    :func:`detect_changepoints`.

    Args:
        df: DataFrame containing ``match_minute``, group-by columns,
            and the signal columns from *spec*.
        spec: Signal specification.
        config: Changepoint detection configuration.

    Returns:
        List of :class:`ChangepointResult`, one per group.
    """
    logger.info(
        "  Config: method=%s  model=%s  penalty=%.2f  min_seg=%d",
        config.method,
        config.model,
        config.penalty,
        config.min_segment_size,
    )
    required = set(GROUP_BY_COLUMNS) | {"match_minute"} | set(spec.signal_columns)
    missing = required - set(df.columns)
    if missing:
        logger.error(
            "Signal '%s': missing columns %s -- skipping",
            spec.name,
            sorted(missing),
        )
        return []

    # Subset to only needed columns and median-impute nulls in
    # signal columns so ruptures receives a dense array.
    select_cols = list(GROUP_BY_COLUMNS) + ["match_minute"] + spec.signal_columns
    df_sub = df.select(select_cols)
    for col in spec.signal_columns:
        median_val = df_sub[col].median()
        if median_val is None:
            median_val = 0.0
        df_sub = df_sub.with_columns(pl.col(col).fill_null(median_val))

    groups = df_sub.group_by(GROUP_BY_COLUMNS, maintain_order=True)
    results: list[ChangepointResult] = []

    for key_vals, group_df in groups:
        group_id = "_".join(str(v) for v in key_vals)
        group_df = group_df.sort("match_minute")

        signal = group_df.select(spec.signal_columns).to_numpy().astype(np.float64)
        match_minutes = group_df["match_minute"].to_numpy().astype(np.float64)

        if signal.shape[0] < config.min_segment_size:
            results.append(
                ChangepointResult(
                    match_id=group_id,
                    changepoints=(),
                    n_segments=1,
                )
            )
            continue

        result = detect_changepoints(
            signal=signal,
            match_minutes=match_minutes,
            match_id=group_id,
            config=config,
            feature_names=spec.signal_columns,
        )
        results.append(result)

    return results


def _results_to_parquet(
    results: list[ChangepointResult],
    spec: SignalSpec,
    output_dir: Path,
) -> Path:
    """Persist changepoint results as a Parquet file.

    Creates one row per changepoint with columns: ``group_id``,
    ``match_id``, ``team_id``, ``cp_index``, ``match_minute``,
    ``n_segments``, and one column per feature delta.

    Args:
        results: Changepoint results from one signal type.
        spec: Signal specification (for naming).
        output_dir: Directory for the Parquet file.

    Returns:
        Path to the written Parquet file.
    """
    rows: list[dict[str, object]] = []
    for r in results:
        parts = r.match_id.split("_", maxsplit=1)
        mid = parts[0] if parts else r.match_id
        tid = parts[1] if len(parts) > 1 else ""

        if not r.changepoints:
            rows.append(
                {
                    "group_id": r.match_id,
                    "match_id": mid,
                    "team_id": tid,
                    "cp_index": None,
                    "match_minute": None,
                    "n_segments": r.n_segments,
                    "top_feature_deltas": None,
                }
            )
            continue

        for cp in r.changepoints:
            delta_str = "; ".join(
                f"{k}: {v:+.4f}" for k, v in cp.feature_deltas.items()
            )
            rows.append(
                {
                    "group_id": r.match_id,
                    "match_id": mid,
                    "team_id": tid,
                    "cp_index": cp.index,
                    "match_minute": cp.match_minute,
                    "n_segments": r.n_segments,
                    "top_feature_deltas": delta_str,
                }
            )

    schema = {
        "group_id": pl.Utf8,
        "match_id": pl.Utf8,
        "team_id": pl.Utf8,
        "cp_index": pl.Int64,
        "match_minute": pl.Float64,
        "n_segments": pl.Int64,
        "top_feature_deltas": pl.Utf8,
    }
    df = pl.DataFrame(rows, schema=schema)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{spec.slug}_changepoints.parquet"
    df.write_parquet(path)
    return path


def _results_to_json_summary(
    results: list[ChangepointResult],
    spec: SignalSpec,
    config: ChangepointConfig,
) -> dict[str, object]:
    """Build a compact JSON-serialisable summary for one signal type.

    Args:
        results: Changepoint results.
        spec: Signal specification.

    Returns:
        Dictionary with aggregate statistics and per-group counts.
    """
    total_cps = sum(len(r.changepoints) for r in results)
    groups_with_cps = sum(1 for r in results if r.changepoints)
    cp_counts = [len(r.changepoints) for r in results]
    cp_arr = np.array(cp_counts) if cp_counts else np.array([0])

    return {
        "signal": spec.name,
        "slug": spec.slug,
        "n_groups": len(results),
        "groups_with_changepoints": groups_with_cps,
        "total_changepoints": total_cps,
        "mean_changepoints_per_group": (
            round(total_cps / len(results), 3) if results else 0.0
        ),
        "median_changepoints_per_group": int(np.median(cp_arr)),
        "max_changepoints_in_group": max(cp_counts) if cp_counts else 0,
        "penalty": config.penalty,
        "method": config.method,
        "model": config.model,
        "min_segment_size": config.min_segment_size,
    }


def _log_summary(summary: dict[str, object]) -> None:
    """Log a signal-level summary to the console.

    Args:
        summary: Dictionary from :func:`_results_to_json_summary`.
    """
    logger.info(
        "  %-30s  groups=%4d  with_cps=%4d  total_cps=%5d"
        "  mean=%.2f  median=%d  max=%d",
        summary["signal"],
        summary["n_groups"],
        summary["groups_with_changepoints"],
        summary["total_changepoints"],
        summary["mean_changepoints_per_group"],
        summary["median_changepoints_per_group"],
        summary["max_changepoints_in_group"],
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    """Run changepoint detection on all signal types."""
    # ----------------------------------------------------------
    # 1. Validate input files exist
    # ----------------------------------------------------------
    required_files = {
        "GMM labeled windows": GMM_PATH,
        "HMM labeled windows": HMM_PATH,
        "VAE labeled windows": VAE_PATH,
        "VAE latent codes": LATENT_PATH,
    }
    for label, path in required_files.items():
        if not path.exists():
            logger.error("%s not found at %s", label, path)
            sys.exit(1)

    # ----------------------------------------------------------
    # 2. Load data
    # ----------------------------------------------------------
    logger.info("Loading GMM labeled windows from %s", GMM_PATH)
    df_gmm = pl.read_parquet(GMM_PATH)
    logger.info("  Shape: %d x %d", df_gmm.height, df_gmm.width)

    logger.info("Loading HMM labeled windows from %s", HMM_PATH)
    df_hmm = pl.read_parquet(HMM_PATH)
    logger.info("  Shape: %d x %d", df_hmm.height, df_hmm.width)

    logger.info("Loading VAE labeled windows from %s", VAE_PATH)
    df_vae = pl.read_parquet(VAE_PATH)
    logger.info("  Shape: %d x %d", df_vae.height, df_vae.width)

    logger.info("Loading VAE latent codes from %s", LATENT_PATH)
    latent_codes = np.load(LATENT_PATH)
    logger.info("  Shape: %s", latent_codes.shape)

    # Attach latent codes as columns to the VAE DataFrame so we
    # can group/sort them alongside match metadata.
    latent_dim = latent_codes.shape[1]
    latent_col_names = [f"z{i}" for i in range(latent_dim)]
    latent_df = pl.DataFrame(
        {name: latent_codes[:, i] for i, name in enumerate(latent_col_names)},
    )
    df_vae_latent = pl.concat([df_vae, latent_df], how="horizontal")

    # ----------------------------------------------------------
    # 3. Define signal specifications
    # ----------------------------------------------------------
    gmm_prob_cols = sorted(c for c in df_gmm.columns if c.startswith("state_prob_"))
    hmm_prob_cols = sorted(c for c in df_hmm.columns if c.startswith("hmm_state_prob_"))

    # Filter raw feature columns to those actually present
    raw_cols = [c for c in RAW_FEATURE_COLUMNS if c in df_gmm.columns]
    if len(raw_cols) < len(RAW_FEATURE_COLUMNS):
        dropped = set(RAW_FEATURE_COLUMNS) - set(raw_cols)
        logger.warning(
            "Raw feature columns not found in GMM data (dropped): %s",
            sorted(dropped),
        )

    specs: list[tuple[SignalSpec, pl.DataFrame, ChangepointConfig]] = [
        (
            SignalSpec(
                name="GMM state probabilities",
                slug="gmm_state_probs",
                signal_columns=gmm_prob_cols,
            ),
            df_gmm,
            _BASE_CONFIG,
        ),
        (
            SignalSpec(
                name="HMM state probabilities",
                slug="hmm_state_probs",
                signal_columns=hmm_prob_cols,
            ),
            df_hmm,
            _BASE_CONFIG,
        ),
        (
            SignalSpec(
                name="VAE latent codes",
                slug="vae_latent_codes",
                signal_columns=latent_col_names,
            ),
            df_vae_latent,
            _VAE_CONFIG,
        ),
        (
            SignalSpec(
                name="Raw tactical features",
                slug="raw_features",
                signal_columns=raw_cols,
            ),
            df_gmm,
            _RAW_CONFIG,
        ),
    ]

    # ----------------------------------------------------------
    # 4. Run detection per signal type
    # ----------------------------------------------------------
    logger.info("")
    logger.info("=" * 80)
    logger.info("Running PELT changepoint detection across 4 signal types")
    logger.info("=" * 80)

    all_summaries: list[dict[str, object]] = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for spec, df, cfg in specs:
        logger.info("")
        logger.info("Signal: %s  (%d columns)", spec.name, len(spec.signal_columns))
        results = _run_detection_on_signal(df, spec, cfg)

        if not results:
            logger.warning("  No results for '%s'", spec.name)
            continue

        # Persist Parquet
        pq_path = _results_to_parquet(results, spec, OUTPUT_DIR)
        logger.info("  Parquet written: %s", pq_path)

        # Summary
        summary = _results_to_json_summary(results, spec, cfg)
        _log_summary(summary)
        all_summaries.append(summary)

    # ----------------------------------------------------------
    # 5. Write combined JSON summary
    # ----------------------------------------------------------
    summary_path = OUTPUT_DIR / "changepoint_summary.json"
    summary_path.write_bytes(
        orjson.dumps(all_summaries, option=orjson.OPT_INDENT_2),
    )
    logger.info("")
    logger.info("JSON summary written: %s", summary_path)

    # ----------------------------------------------------------
    # 6. Final log table
    # ----------------------------------------------------------
    logger.info("")
    logger.info("=" * 80)
    logger.info("CHANGEPOINT DETECTION SUMMARY")
    logger.info("=" * 80)
    logger.info(
        "  %-30s  %6s  %8s  %9s  %6s  %6s  %4s  %7s",
        "Signal",
        "Groups",
        "WithCPs",
        "TotalCPs",
        "Mean",
        "Median",
        "Max",
        "Penalty",
    )
    logger.info("-" * 96)
    for s in all_summaries:
        logger.info(
            "  %-30s  %6d  %8d  %9d  %6.2f  %6d  %4d  %7.2f",
            s["signal"],
            s["n_groups"],
            s["groups_with_changepoints"],
            s["total_changepoints"],
            s["mean_changepoints_per_group"],
            s["median_changepoints_per_group"],
            s["max_changepoints_in_group"],
            s["penalty"],
        )
    logger.info("=" * 80)
    logger.info("Done.")


if __name__ == "__main__":
    main()
