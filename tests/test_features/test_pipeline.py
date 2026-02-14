"""Tests for the end-to-end feature extraction pipeline.

Validates that :func:`extract_match_features` correctly wires
segmentation to feature extraction, producing a polars DataFrame
with the expected metadata columns, feature columns, and segment
types.
"""

from __future__ import annotations

import polars as pl
import pytest

from tactical.adapters.schemas import MatchContext, NormalizedEvent
from tactical.config import PipelineConfig, PossessionConfig, WindowConfig
from tactical.exceptions import FeatureExtractionError, SegmentationError
from tactical.features.pipeline import extract_match_features
from tactical.features.registry import create_default_registry
from tactical.segmentation.possession import create_possession_sequences
from tactical.segmentation.windows import create_time_windows

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_METADATA_COLUMNS: tuple[str, ...] = (
    "match_id",
    "team_id",
    "segment_type",
    "start_time",
    "end_time",
    "period",
    "match_minute",
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
    """Build a realistic stream of 20+ events for pipeline testing.

    Contains interleaved team_a and team_b events across two
    possessions so that both window and possession segmentation
    produce non-empty results with default config.
    """
    return [
        # --- team_a build-up ---
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
        # --- team_b pressure + turnover ---
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
        # --- team_b counter ---
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
        # --- team_a pressing back ---
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
        # --- team_a second possession ---
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


@pytest.fixture()
def events() -> list[NormalizedEvent]:
    """Realistic event stream for pipeline testing."""
    return _make_events()


@pytest.fixture()
def config() -> PipelineConfig:
    """Pipeline config with small windows for testability."""
    return PipelineConfig(
        max_feature_tier=2,
        window=_SMALL_WINDOW_CFG,
        possession=_SMALL_POSSESSION_CFG,
    )


@pytest.fixture()
def config_tier3() -> PipelineConfig:
    """Pipeline config requesting Tier 3 features."""
    return PipelineConfig(
        max_feature_tier=3,
        window=_SMALL_WINDOW_CFG,
        possession=_SMALL_POSSESSION_CFG,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestPipelineProducesDataFrame:
    """Pipeline should return a polars DataFrame."""

    def test_pipeline_produces_dataframe(
        self,
        events: list[NormalizedEvent],
        config: PipelineConfig,
    ) -> None:
        """Result should be a polars DataFrame with rows."""
        df = extract_match_features(events, _CTX, config)
        assert isinstance(df, pl.DataFrame)
        assert df.height > 0


class TestPipelineWindowRows:
    """Window segments should appear in the output."""

    def test_pipeline_window_rows(
        self,
        events: list[NormalizedEvent],
        config: PipelineConfig,
    ) -> None:
        """At least one row should have segment_type='window'."""
        df = extract_match_features(events, _CTX, config)
        window_rows = df.filter(pl.col("segment_type") == "window")
        assert window_rows.height > 0


class TestPipelinePossessionRows:
    """Possession segments should appear in the output."""

    def test_pipeline_possession_rows(
        self,
        events: list[NormalizedEvent],
        config: PipelineConfig,
    ) -> None:
        """At least one row should have segment_type='possession'."""
        df = extract_match_features(events, _CTX, config)
        poss_rows = df.filter(pl.col("segment_type") == "possession")
        assert poss_rows.height > 0


class TestPipelineFeatureColumnsPresent:
    """All Tier 1 and Tier 2 feature columns should be present."""

    def test_pipeline_feature_columns_present(
        self,
        events: list[NormalizedEvent],
        config: PipelineConfig,
    ) -> None:
        """Every feature name from the registry (tier <= 2) must appear."""
        registry = create_default_registry()
        expected_names = registry.get_all_feature_names(max_tier=2)

        df = extract_match_features(events, _CTX, config)

        df_columns = set(df.columns)
        for name in expected_names:
            assert name in df_columns, f"Missing feature column: {name}"


class TestPipelineTier3ColumnsNullWithout360:
    """Tier 3 columns should be null or absent without 360 data."""

    def test_tier3_absent_when_max_tier_2(
        self,
        events: list[NormalizedEvent],
        config: PipelineConfig,
    ) -> None:
        """With max_feature_tier=2, Tier 3 columns should be absent."""
        registry = create_default_registry()
        t3_names = registry.get_all_feature_names(max_tier=3)
        t2_names = set(registry.get_all_feature_names(max_tier=2))
        tier3_only = [n for n in t3_names if n not in t2_names]

        df = extract_match_features(events, _CTX, config)

        df_columns = set(df.columns)
        for name in tier3_only:
            assert name not in df_columns, (
                f"Tier 3 column {name} present with max_tier=2"
            )

    def test_tier3_null_when_no_360(
        self,
        events: list[NormalizedEvent],
        config_tier3: PipelineConfig,
    ) -> None:
        """With max_tier=3 and has_360=False, T3 columns exist but are null."""
        registry = create_default_registry()
        t3_names = registry.get_all_feature_names(max_tier=3)
        t2_names = set(registry.get_all_feature_names(max_tier=2))
        tier3_only = [n for n in t3_names if n not in t2_names]

        # has_360=False but requesting tier 3
        df = extract_match_features(events, _CTX, config_tier3)

        for name in tier3_only:
            assert name in df.columns, f"Tier 3 column {name} missing"
            non_null = df[name].drop_nulls()
            assert non_null.len() == 0, (
                f"Tier 3 column {name} has non-null values without 360 data"
            )


class TestPipelineEmptyEventsRaises:
    """Empty event list should raise FeatureExtractionError."""

    def test_pipeline_empty_events_raises(
        self,
        config: PipelineConfig,
    ) -> None:
        """FeatureExtractionError when no events are supplied."""
        with pytest.raises((FeatureExtractionError, SegmentationError)):
            extract_match_features([], _CTX, config)


class TestPipelineMetadataColumns:
    """All metadata columns should be present and correct."""

    def test_pipeline_metadata_columns(
        self,
        events: list[NormalizedEvent],
        config: PipelineConfig,
    ) -> None:
        """Every metadata column must exist and contain correct values."""
        df = extract_match_features(events, _CTX, config)

        for col in _METADATA_COLUMNS:
            assert col in df.columns, f"Missing metadata column: {col}"

        # match_id should be consistent
        assert df["match_id"].unique().to_list() == ["match_001"]

        # team_id should be consistent
        assert df["team_id"].unique().to_list() == ["team_a"]

        # segment_type should only contain valid values
        seg_types = set(df["segment_type"].unique().to_list())
        assert seg_types <= {"window", "possession"}

        # period should be positive integers
        periods = df["period"].to_list()
        assert all(isinstance(p, int) and p > 0 for p in periods)

        # start_time < end_time for each row
        starts = df["start_time"].to_list()
        ends = df["end_time"].to_list()
        for i, (s, e) in enumerate(zip(starts, ends, strict=True)):
            assert s <= e, f"Row {i}: start_time {s} > end_time {e}"

    def test_window_metadata_matches_segmentation(
        self,
        events: list[NormalizedEvent],
        config: PipelineConfig,
    ) -> None:
        """Window rows should have metadata matching actual time windows."""
        windows = create_time_windows(events, config.window)
        df = extract_match_features(events, _CTX, config, windows=windows)

        win_df = df.filter(pl.col("segment_type") == "window")
        assert win_df.height == len(windows)

        for i, win in enumerate(windows):
            row = win_df.row(i, named=True)
            assert row["start_time"] == pytest.approx(win.start_time, abs=1e-6)
            assert row["end_time"] == pytest.approx(win.end_time, abs=1e-6)
            assert row["period"] == win.period
            assert row["match_minute"] == pytest.approx(
                win.match_minute_start,
                abs=1e-6,
            )

    def test_possession_metadata_matches_segmentation(
        self,
        events: list[NormalizedEvent],
        config: PipelineConfig,
    ) -> None:
        """Possession rows should have metadata matching actual possessions."""
        possessions = create_possession_sequences(events, config.possession)
        df = extract_match_features(
            events,
            _CTX,
            config,
            possessions=possessions,
        )

        poss_df = df.filter(pl.col("segment_type") == "possession")
        assert poss_df.height == len(possessions)

        for i, poss in enumerate(possessions):
            row = poss_df.row(i, named=True)
            assert row["start_time"] == pytest.approx(poss.start_time, abs=1e-6)
            assert row["end_time"] == pytest.approx(poss.end_time, abs=1e-6)
            assert row["period"] == poss.period


class TestPipelinePrecomputedSegments:
    """Pipeline should accept pre-computed segments."""

    def test_precomputed_windows_only(
        self,
        events: list[NormalizedEvent],
        config: PipelineConfig,
    ) -> None:
        """Passing explicit windows should use them directly."""
        windows = create_time_windows(events, config.window)
        possessions = create_possession_sequences(events, config.possession)

        df = extract_match_features(
            events,
            _CTX,
            config,
            windows=windows,
            possessions=possessions,
        )

        expected_rows = len(windows) + len(possessions)
        assert df.height == expected_rows


class TestPipelineRowCount:
    """Total rows should equal number of windows + possessions."""

    def test_row_count(
        self,
        events: list[NormalizedEvent],
        config: PipelineConfig,
    ) -> None:
        """Row count = len(windows) + len(possessions)."""
        windows = create_time_windows(events, config.window)
        possessions = create_possession_sequences(events, config.possession)

        df = extract_match_features(
            events,
            _CTX,
            config,
            windows=windows,
            possessions=possessions,
        )

        win_count = df.filter(pl.col("segment_type") == "window").height
        poss_count = df.filter(pl.col("segment_type") == "possession").height

        assert win_count == len(windows)
        assert poss_count == len(possessions)


class TestCreateDefaultRegistry:
    """Validate the factory function for the default registry."""

    def test_all_tiers_present(self) -> None:
        """Registry should contain extractors from all 3 tiers."""
        registry = create_default_registry()
        tiers = {e.tier for e in registry.get_active_extractors(max_tier=3)}
        assert tiers == {1, 2, 3}

    def test_no_duplicate_feature_names(self) -> None:
        """No two extractors should produce the same feature name."""
        registry = create_default_registry()
        names = registry.get_all_feature_names(max_tier=3)
        assert len(names) == len(set(names)), "Duplicate feature names found"

    def test_tier_filtering(self) -> None:
        """get_active_extractors respects max_tier boundary."""
        registry = create_default_registry()
        t1 = registry.get_active_extractors(max_tier=1)
        t2 = registry.get_active_extractors(max_tier=2)
        t3 = registry.get_active_extractors(max_tier=3)
        assert len(t1) < len(t2) < len(t3)
