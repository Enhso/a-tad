"""Tests for the end-to-end feature extraction and Parquet output pipeline.

Validates that :func:`run_extraction` produces correct DataFrames,
that Parquet round-trips preserve data, that summary logging reports
accurate statistics, and that edge cases are handled gracefully.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pytest

from tactical.adapters.schemas import MatchContext, MatchInfo, NormalizedEvent
from tactical.config import PipelineConfig, PossessionConfig, WindowConfig
from tactical.exceptions import FeatureExtractionError, SegmentationError
from tactical.features.pipeline import extract_match_features
from tactical.features.registry import create_default_registry

if TYPE_CHECKING:
    import types

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_METADATA_COLUMNS: frozenset[str] = frozenset(
    {
        "match_id",
        "team_id",
        "segment_type",
        "start_time",
        "end_time",
        "period",
        "match_minute",
    }
)


def _evt(
    event_id: str,
    *,
    team_id: str = "team_a",
    player_id: str = "player_01",
    period: int = 1,
    timestamp: float,
    match_minute: float,
    location: tuple[float, float] = (50.0, 50.0),
    end_location: tuple[float, float] | None = None,
    event_type: str = "pass",
    event_outcome: str = "complete",
    under_pressure: bool = False,
    body_part: str = "right_foot",
) -> NormalizedEvent:
    """Build a :class:`NormalizedEvent` with sensible defaults."""
    return NormalizedEvent(
        event_id=event_id,
        match_id="match_001",
        team_id=team_id,
        player_id=player_id,
        period=period,
        timestamp=timestamp,
        match_minute=match_minute,
        location=location,
        end_location=end_location,
        event_type=event_type,
        event_outcome=event_outcome,
        under_pressure=under_pressure,
        body_part=body_part,
        freeze_frame=None,
        score_home=0,
        score_away=0,
        team_is_home=team_id == "team_a",
    )


def _make_events() -> list[NormalizedEvent]:
    """Build a realistic stream of 24 events for pipeline testing.

    Contains interleaved team_a and team_b events across two
    possessions so that both window and possession segmentation
    produce non-empty results with default config.
    """
    return [
        _evt(
            "p01",
            timestamp=0.0,
            match_minute=0.0,
            location=(20.0, 40.0),
            end_location=(35.0, 45.0),
        ),
        _evt(
            "p02",
            timestamp=1.5,
            match_minute=0.03,
            location=(35.0, 45.0),
            event_type="ball_receipt",
            player_id="player_02",
        ),
        _evt(
            "p03",
            timestamp=3.0,
            match_minute=0.05,
            location=(35.0, 45.0),
            end_location=(50.0, 50.0),
            event_type="carry",
            player_id="player_02",
        ),
        _evt(
            "p04",
            timestamp=5.0,
            match_minute=0.08,
            location=(50.0, 50.0),
            end_location=(65.0, 42.0),
            player_id="player_02",
        ),
        _evt(
            "p05",
            timestamp=6.5,
            match_minute=0.11,
            location=(65.0, 42.0),
            event_type="ball_receipt",
            player_id="player_03",
        ),
        _evt(
            "p06",
            timestamp=8.0,
            match_minute=0.13,
            location=(65.0, 42.0),
            end_location=(75.0, 38.0),
            player_id="player_03",
        ),
        _evt(
            "p07",
            timestamp=9.5,
            match_minute=0.16,
            location=(75.0, 38.0),
            event_type="ball_receipt",
            player_id="player_04",
        ),
        _evt(
            "p08",
            timestamp=11.0,
            match_minute=0.18,
            location=(75.0, 38.0),
            end_location=(82.0, 46.0),
            player_id="player_04",
        ),
        _evt(
            "p09",
            team_id="team_b",
            player_id="player_11",
            timestamp=12.0,
            match_minute=0.20,
            location=(80.0, 45.0),
            event_type="pressure",
        ),
        _evt(
            "p10",
            timestamp=13.0,
            match_minute=0.22,
            location=(82.0, 46.0),
            end_location=(85.0, 50.0),
            event_outcome="incomplete",
            under_pressure=True,
            player_id="player_04",
        ),
        _evt(
            "p11",
            team_id="team_b",
            player_id="player_12",
            timestamp=14.0,
            match_minute=0.23,
            location=(85.0, 50.0),
            event_type="ball_recovery",
        ),
        _evt(
            "p12",
            team_id="team_b",
            player_id="player_12",
            timestamp=15.5,
            match_minute=0.26,
            location=(85.0, 50.0),
            end_location=(70.0, 55.0),
        ),
        _evt(
            "p13",
            team_id="team_b",
            player_id="player_13",
            timestamp=17.0,
            match_minute=0.28,
            location=(70.0, 55.0),
            event_type="ball_receipt",
        ),
        _evt(
            "p14",
            team_id="team_b",
            player_id="player_13",
            timestamp=18.5,
            match_minute=0.31,
            location=(70.0, 55.0),
            end_location=(55.0, 48.0),
            event_type="carry",
        ),
        _evt(
            "p15",
            team_id="team_b",
            player_id="player_13",
            timestamp=20.0,
            match_minute=0.33,
            location=(55.0, 48.0),
            end_location=(40.0, 42.0),
        ),
        _evt(
            "p16",
            timestamp=21.5,
            match_minute=0.36,
            location=(42.0, 43.0),
            event_type="pressure",
            player_id="player_01",
        ),
        _evt(
            "p17",
            timestamp=22.0,
            match_minute=0.37,
            location=(40.0, 42.0),
            event_type="interception",
            player_id="player_02",
        ),
        _evt(
            "p18",
            timestamp=23.5,
            match_minute=0.39,
            location=(40.0, 42.0),
            end_location=(55.0, 50.0),
            player_id="player_02",
        ),
        _evt(
            "p19",
            timestamp=25.0,
            match_minute=0.42,
            location=(55.0, 50.0),
            event_type="ball_receipt",
            player_id="player_03",
        ),
        _evt(
            "p20",
            timestamp=26.5,
            match_minute=0.44,
            location=(55.0, 50.0),
            end_location=(68.0, 45.0),
            player_id="player_03",
        ),
        _evt(
            "p21",
            timestamp=28.0,
            match_minute=0.47,
            location=(68.0, 45.0),
            event_type="ball_receipt",
            player_id="player_04",
        ),
        _evt(
            "p22",
            timestamp=29.5,
            match_minute=0.49,
            location=(68.0, 45.0),
            end_location=(78.0, 48.0),
            player_id="player_04",
        ),
        _evt(
            "p23",
            timestamp=31.0,
            match_minute=0.52,
            location=(78.0, 48.0),
            event_type="ball_receipt",
            player_id="player_05",
        ),
        _evt(
            "p24",
            timestamp=32.5,
            match_minute=0.54,
            location=(78.0, 48.0),
            end_location=(88.0, 50.0),
            event_type="shot",
            event_outcome="saved",
            player_id="player_05",
        ),
    ]


# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------

_CTX = MatchContext(
    match_id="match_001",
    team_id="team_a",
    opponent_id="team_b",
    team_is_home=True,
    has_360=False,
)

_CTX_360 = MatchContext(
    match_id="match_001",
    team_id="team_a",
    opponent_id="team_b",
    team_is_home=True,
    has_360=True,
)

_SMALL_WINDOW_CFG = WindowConfig(
    window_seconds=15.0,
    overlap_seconds=5.0,
    min_events=3,
)

_SMALL_POSSESSION_CFG = PossessionConfig(min_events=3)

_MATCH_INFO = MatchInfo(
    match_id="match_001",
    competition="Test League",
    season="2024/2025",
    home_team_id="team_a",
    home_team_name="Team A",
    away_team_id="team_b",
    away_team_name="Team B",
    home_score=1,
    away_score=0,
    match_date="2025-01-01",
)


@pytest.fixture()
def events() -> list[NormalizedEvent]:
    """Realistic event stream for pipeline testing."""
    return _make_events()


@pytest.fixture()
def config_t2() -> PipelineConfig:
    """Pipeline config with max_tier=2 and small windows."""
    return PipelineConfig(
        max_feature_tier=2,
        window=_SMALL_WINDOW_CFG,
        possession=_SMALL_POSSESSION_CFG,
    )


@pytest.fixture()
def config_t3() -> PipelineConfig:
    """Pipeline config with max_tier=3 and small windows."""
    return PipelineConfig(
        max_feature_tier=3,
        window=_SMALL_WINDOW_CFG,
        possession=_SMALL_POSSESSION_CFG,
    )


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    """Temporary directory for Parquet output."""
    out = tmp_path / "output"
    out.mkdir()
    return out


# ------------------------------------------------------------------
# Parquet round-trip tests
# ------------------------------------------------------------------


class TestParquetWriteRead:
    """Parquet file should preserve all data through a write/read cycle."""

    def test_parquet_round_trip(
        self,
        events: list[NormalizedEvent],
        config_t2: PipelineConfig,
        output_dir: Path,
    ) -> None:
        """Write DataFrame to Parquet and read it back unchanged."""
        df = extract_match_features(events, _CTX, config_t2)
        parquet_path = output_dir / "features.parquet"

        df.write_parquet(parquet_path)
        assert parquet_path.exists()
        assert parquet_path.stat().st_size > 0

        loaded = pl.read_parquet(parquet_path)
        assert loaded.shape == df.shape
        assert loaded.columns == df.columns

    def test_parquet_preserves_dtypes(
        self,
        events: list[NormalizedEvent],
        config_t2: PipelineConfig,
        output_dir: Path,
    ) -> None:
        """Numeric columns remain float, string columns remain string."""
        df = extract_match_features(events, _CTX, config_t2)
        parquet_path = output_dir / "roundtrip.parquet"

        df.write_parquet(parquet_path)
        loaded = pl.read_parquet(parquet_path)

        for col in ("match_id", "team_id", "segment_type"):
            assert loaded[col].dtype == pl.Utf8

        for col in ("start_time", "end_time", "match_minute"):
            assert loaded[col].dtype == pl.Float64

    def test_parquet_file_size_reasonable(
        self,
        events: list[NormalizedEvent],
        config_t2: PipelineConfig,
        output_dir: Path,
    ) -> None:
        """Output file should be non-trivial but not absurdly large."""
        df = extract_match_features(events, _CTX, config_t2)
        parquet_path = output_dir / "size_check.parquet"

        df.write_parquet(parquet_path)
        size = parquet_path.stat().st_size

        assert size > 100, "Parquet too small -- likely empty"
        assert size < 10 * 1024 * 1024, "Parquet unexpectedly large (>10 MB)"


# ------------------------------------------------------------------
# Metadata tests
# ------------------------------------------------------------------


class TestMetadataInOutput:
    """All metadata columns should be present with correct values."""

    def test_metadata_columns_present(
        self,
        events: list[NormalizedEvent],
        config_t2: PipelineConfig,
    ) -> None:
        """Every metadata column must exist."""
        df = extract_match_features(events, _CTX, config_t2)
        for col in _METADATA_COLUMNS:
            assert col in df.columns, f"Missing metadata column: {col}"

    def test_match_id_consistent(
        self,
        events: list[NormalizedEvent],
        config_t2: PipelineConfig,
    ) -> None:
        """All rows should carry the context's match_id."""
        df = extract_match_features(events, _CTX, config_t2)
        assert df["match_id"].unique().to_list() == ["match_001"]

    def test_team_id_consistent(
        self,
        events: list[NormalizedEvent],
        config_t2: PipelineConfig,
    ) -> None:
        """All rows should carry the context's team_id."""
        df = extract_match_features(events, _CTX, config_t2)
        assert df["team_id"].unique().to_list() == ["team_a"]

    def test_segment_types_valid(
        self,
        events: list[NormalizedEvent],
        config_t2: PipelineConfig,
    ) -> None:
        """segment_type should only contain 'window' or 'possession'."""
        df = extract_match_features(events, _CTX, config_t2)
        seg_types = set(df["segment_type"].unique().to_list())
        assert seg_types <= {"window", "possession"}
        assert len(seg_types) > 0

    def test_period_positive(
        self,
        events: list[NormalizedEvent],
        config_t2: PipelineConfig,
    ) -> None:
        """Period values should all be positive integers."""
        df = extract_match_features(events, _CTX, config_t2)
        periods = df["period"].to_list()
        assert all(isinstance(p, int) and p > 0 for p in periods)

    def test_start_le_end(
        self,
        events: list[NormalizedEvent],
        config_t2: PipelineConfig,
    ) -> None:
        """start_time must be <= end_time for every row."""
        df = extract_match_features(events, _CTX, config_t2)
        starts = df["start_time"].to_list()
        ends = df["end_time"].to_list()
        for i, (s, e) in enumerate(zip(starts, ends, strict=True)):
            assert s <= e, f"Row {i}: start_time={s} > end_time={e}"


# ------------------------------------------------------------------
# Null handling tests
# ------------------------------------------------------------------


class TestNullHandling:
    """Verify null behavior for tier 3 columns without 360 data."""

    def test_tier3_columns_null_without_360(
        self,
        events: list[NormalizedEvent],
        config_t3: PipelineConfig,
    ) -> None:
        """With max_tier=3 and has_360=False, T3 features are all null."""
        registry = create_default_registry()
        all_names = set(registry.get_all_feature_names(max_tier=3))
        t2_names = set(registry.get_all_feature_names(max_tier=2))
        t3_only = all_names - t2_names

        df = extract_match_features(events, _CTX, config_t3)

        for name in t3_only:
            assert name in df.columns, f"T3 column {name} missing"
            non_null = df[name].drop_nulls()
            assert non_null.len() == 0, (
                f"T3 column {name} has non-null values without 360 data"
            )

    def test_tier2_columns_not_all_null(
        self,
        events: list[NormalizedEvent],
        config_t2: PipelineConfig,
    ) -> None:
        """Tier 1+2 features should have at least some non-null values."""
        registry = create_default_registry()
        t2_names = registry.get_all_feature_names(max_tier=2)

        df = extract_match_features(events, _CTX, config_t2)

        some_populated = False
        for name in t2_names:
            if df[name].drop_nulls().len() > 0:
                some_populated = True
                break

        assert some_populated, "All T1+T2 features are null -- unexpected"

    def test_tier3_absent_when_max_tier_2(
        self,
        events: list[NormalizedEvent],
        config_t2: PipelineConfig,
    ) -> None:
        """With max_feature_tier=2, Tier 3 columns should be absent."""
        registry = create_default_registry()
        all_names = set(registry.get_all_feature_names(max_tier=3))
        t2_names = set(registry.get_all_feature_names(max_tier=2))
        t3_only = all_names - t2_names

        df = extract_match_features(events, _CTX, config_t2)

        df_columns = set(df.columns)
        for name in t3_only:
            assert name not in df_columns, (
                f"T3 column {name} should be absent at max_tier=2"
            )


# ------------------------------------------------------------------
# Summary logging tests
# ------------------------------------------------------------------


class TestSummaryLogging:
    """Validate that _log_dataframe_summary produces correct output."""

    def test_log_reports_total_rows(
        self,
        events: list[NormalizedEvent],
        config_t2: PipelineConfig,
        output_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Summary log should contain the total row count."""
        # Import here so the path manipulation in the script doesn't
        # collide with normal test imports.
        import importlib
        import sys

        script_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
        sys.path.insert(0, str(script_dir))

        mod = importlib.import_module("run_feature_extraction")
        sys.path.pop(0)

        df = extract_match_features(events, _CTX, config_t2)
        parquet_path = output_dir / "log_test.parquet"
        df.write_parquet(parquet_path)

        with caplog.at_level(logging.INFO):
            mod._log_dataframe_summary(df, parquet_path)

        combined = "\n".join(caplog.messages)
        assert f"Total rows       : {df.height}" in combined

    def test_log_reports_feature_count(
        self,
        events: list[NormalizedEvent],
        config_t2: PipelineConfig,
        output_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Summary log should report the number of feature columns."""
        import importlib
        import sys

        script_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
        sys.path.insert(0, str(script_dir))

        mod = importlib.import_module("run_feature_extraction")
        sys.path.pop(0)

        df = extract_match_features(events, _CTX, config_t2)
        parquet_path = output_dir / "log_test2.parquet"
        df.write_parquet(parquet_path)

        feature_cols = [c for c in df.columns if c not in _METADATA_COLUMNS]

        with caplog.at_level(logging.INFO):
            mod._log_dataframe_summary(df, parquet_path)

        combined = "\n".join(caplog.messages)
        assert f"Feature columns  : {len(feature_cols)}" in combined

    def test_log_reports_file_size(
        self,
        events: list[NormalizedEvent],
        config_t2: PipelineConfig,
        output_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Summary log should include the file size."""
        import importlib
        import sys

        script_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
        sys.path.insert(0, str(script_dir))

        mod = importlib.import_module("run_feature_extraction")
        sys.path.pop(0)

        df = extract_match_features(events, _CTX, config_t2)
        parquet_path = output_dir / "log_test3.parquet"
        df.write_parquet(parquet_path)

        with caplog.at_level(logging.INFO):
            mod._log_dataframe_summary(df, parquet_path)

        combined = "\n".join(caplog.messages)
        assert "File size" in combined

    def test_log_reports_null_rate(
        self,
        events: list[NormalizedEvent],
        config_t3: PipelineConfig,
        output_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Summary log should include the overall null rate."""
        import importlib
        import sys

        script_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
        sys.path.insert(0, str(script_dir))

        mod = importlib.import_module("run_feature_extraction")
        sys.path.pop(0)

        df = extract_match_features(events, _CTX, config_t3)
        parquet_path = output_dir / "log_test4.parquet"
        df.write_parquet(parquet_path)

        with caplog.at_level(logging.INFO):
            mod._log_dataframe_summary(df, parquet_path)

        combined = "\n".join(caplog.messages)
        assert "Overall null rate" in combined
        assert "feature cells" in combined


# ------------------------------------------------------------------
# Concatenation tests (multi-team simulation)
# ------------------------------------------------------------------


class TestMultiTeamConcatenation:
    """Simulate extracting features from both teams in one match."""

    def test_two_teams_concatenate(
        self,
        events: list[NormalizedEvent],
        config_t2: PipelineConfig,
    ) -> None:
        """Extracting for home + away and concatenating should work."""
        ctx_away = MatchContext(
            match_id="match_001",
            team_id="team_b",
            opponent_id="team_a",
            team_is_home=False,
            has_360=False,
        )

        df_home = extract_match_features(events, _CTX, config_t2)
        df_away = extract_match_features(events, ctx_away, config_t2)

        combined = pl.concat([df_home, df_away], how="diagonal_relaxed")

        assert combined.height == df_home.height + df_away.height
        assert set(combined["team_id"].unique().to_list()) == {"team_a", "team_b"}

    def test_two_teams_same_columns(
        self,
        events: list[NormalizedEvent],
        config_t2: PipelineConfig,
    ) -> None:
        """Both teams should produce identical column sets."""
        ctx_away = MatchContext(
            match_id="match_001",
            team_id="team_b",
            opponent_id="team_a",
            team_is_home=False,
            has_360=False,
        )

        df_home = extract_match_features(events, _CTX, config_t2)
        df_away = extract_match_features(events, ctx_away, config_t2)

        assert set(df_home.columns) == set(df_away.columns)


# ------------------------------------------------------------------
# Build-contexts helper test
# ------------------------------------------------------------------


class TestBuildContexts:
    """Validate _build_contexts produces correct MatchContext pairs."""

    def test_build_contexts(self) -> None:
        """Home and away contexts should have correct IDs and flags."""
        import importlib
        import sys

        script_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
        sys.path.insert(0, str(script_dir))

        mod = importlib.import_module("run_feature_extraction")
        sys.path.pop(0)

        home_ctx, away_ctx = mod._build_contexts(_MATCH_INFO, has_360=True)

        assert home_ctx.match_id == "match_001"
        assert home_ctx.team_id == "team_a"
        assert home_ctx.opponent_id == "team_b"
        assert home_ctx.team_is_home is True
        assert home_ctx.has_360 is True

        assert away_ctx.match_id == "match_001"
        assert away_ctx.team_id == "team_b"
        assert away_ctx.opponent_id == "team_a"
        assert away_ctx.team_is_home is False
        assert away_ctx.has_360 is True

    def test_build_contexts_no_360(self) -> None:
        """has_360=False should propagate to both contexts."""
        import importlib
        import sys

        script_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
        sys.path.insert(0, str(script_dir))

        mod = importlib.import_module("run_feature_extraction")
        sys.path.pop(0)

        home_ctx, away_ctx = mod._build_contexts(_MATCH_INFO, has_360=False)

        assert home_ctx.has_360 is False
        assert away_ctx.has_360 is False


# ------------------------------------------------------------------
# Format file size helper test
# ------------------------------------------------------------------


class TestFormatFileSize:
    """Validate _format_file_size produces readable strings."""

    @pytest.fixture()
    def _mod(self) -> types.ModuleType:
        """Import the script module."""
        import importlib
        import sys

        script_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
        sys.path.insert(0, str(script_dir))

        mod = importlib.import_module("run_feature_extraction")
        sys.path.pop(0)
        return mod

    def test_bytes(self, _mod: types.ModuleType) -> None:
        """Small values should be formatted as bytes."""
        assert _mod._format_file_size(512) == "512 B"

    def test_kilobytes(self, _mod: types.ModuleType) -> None:
        """Values >= 1024 should use KB."""
        result = _mod._format_file_size(2048)
        assert "KB" in result
        assert "2.00" in result

    def test_megabytes(self, _mod: types.ModuleType) -> None:
        """Values >= 1 MB should use MB."""
        result = _mod._format_file_size(5 * 1024 * 1024)
        assert "MB" in result
        assert "5.00" in result


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case behaviour for the pipeline."""

    def test_empty_events_raises(
        self,
        config_t2: PipelineConfig,
    ) -> None:
        """Empty event list should raise an error."""
        with pytest.raises((FeatureExtractionError, SegmentationError)):
            extract_match_features([], _CTX, config_t2)

    def test_parquet_columns_stable_across_tiers(
        self,
        events: list[NormalizedEvent],
        config_t2: PipelineConfig,
        config_t3: PipelineConfig,
        output_dir: Path,
    ) -> None:
        """Tier 3 output should be a strict superset of tier 2 columns."""
        df_t2 = extract_match_features(events, _CTX, config_t2)
        df_t3 = extract_match_features(events, _CTX, config_t3)

        t2_cols = set(df_t2.columns)
        t3_cols = set(df_t3.columns)
        assert t2_cols <= t3_cols, (
            f"Tier 2 columns not a subset of tier 3: {t2_cols - t3_cols}"
        )

    def test_parquet_write_creates_directory(
        self,
        events: list[NormalizedEvent],
        config_t2: PipelineConfig,
        tmp_path: Path,
    ) -> None:
        """Writing to a nested path should succeed after mkdir."""
        df = extract_match_features(events, _CTX, config_t2)
        nested = tmp_path / "deep" / "nested"
        nested.mkdir(parents=True)

        parquet_path = nested / "features.parquet"
        df.write_parquet(parquet_path)
        assert parquet_path.exists()

        loaded = pl.read_parquet(parquet_path)
        assert loaded.height == df.height
