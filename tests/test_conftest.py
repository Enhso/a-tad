"""Tests validating the shared conftest fixtures.

Ensures that :func:`sample_event`, :func:`sample_events_window`,
:func:`sample_events_with_360`, and :func:`sample_feature_df` all
satisfy their documented contracts: correct types, counts, sort
order, coordinate ranges, 360 coverage, and DataFrame schema.
"""

from __future__ import annotations

import polars as pl

from tactical.adapters.schemas import (
    CONTROLLED_EVENT_TYPES,
    FreezeFramePlayer,
    NormalizedEvent,
)
from tests.conftest import (
    ALL_FEATURES,
    N_ROWS,
    N_T1,
    N_T2,
    N_T3,
    T1_FEATURES,
    T2_FEATURES,
    T3_FEATURES,
)

# ------------------------------------------------------------------
# sample_event
# ------------------------------------------------------------------


class TestSampleEvent:
    """Validate the single-event fixture."""

    def test_type(self, sample_event: NormalizedEvent) -> None:
        """Fixture returns a NormalizedEvent instance."""
        assert isinstance(sample_event, NormalizedEvent)

    def test_event_type_controlled(self, sample_event: NormalizedEvent) -> None:
        """Event type belongs to the controlled vocabulary."""
        assert sample_event.event_type in CONTROLLED_EVENT_TYPES

    def test_is_pass(self, sample_event: NormalizedEvent) -> None:
        """Default fixture event is a pass."""
        assert sample_event.event_type == "pass"

    def test_location_in_range(self, sample_event: NormalizedEvent) -> None:
        """Location coordinates are within [0, 100]."""
        assert sample_event.location is not None
        x, y = sample_event.location
        assert 0.0 <= x <= 100.0
        assert 0.0 <= y <= 100.0

    def test_end_location_in_range(self, sample_event: NormalizedEvent) -> None:
        """End-location coordinates are within [0, 100]."""
        assert sample_event.end_location is not None
        x, y = sample_event.end_location
        assert 0.0 <= x <= 100.0
        assert 0.0 <= y <= 100.0

    def test_no_freeze_frame(self, sample_event: NormalizedEvent) -> None:
        """Default fixture has no freeze-frame data."""
        assert sample_event.freeze_frame is None

    def test_match_and_team_ids(self, sample_event: NormalizedEvent) -> None:
        """Metadata strings are populated."""
        assert sample_event.match_id
        assert sample_event.team_id
        assert sample_event.player_id

    def test_outcome_complete(self, sample_event: NormalizedEvent) -> None:
        """Default fixture pass is complete."""
        assert sample_event.event_outcome == "complete"


# ------------------------------------------------------------------
# sample_events_window
# ------------------------------------------------------------------


class TestSampleEventsWindow:
    """Validate the 10-event window fixture."""

    def test_type(self, sample_events_window: tuple[NormalizedEvent, ...]) -> None:
        """Fixture returns a tuple."""
        assert isinstance(sample_events_window, tuple)

    def test_count(self, sample_events_window: tuple[NormalizedEvent, ...]) -> None:
        """Window contains exactly 10 events."""
        assert len(sample_events_window) == 10

    def test_all_normalized_events(
        self, sample_events_window: tuple[NormalizedEvent, ...]
    ) -> None:
        """Every element is a NormalizedEvent."""
        for evt in sample_events_window:
            assert isinstance(evt, NormalizedEvent)

    def test_all_types_controlled(
        self, sample_events_window: tuple[NormalizedEvent, ...]
    ) -> None:
        """Every event type is in the controlled vocabulary."""
        for evt in sample_events_window:
            assert evt.event_type in CONTROLLED_EVENT_TYPES

    def test_sorted_by_period_timestamp(
        self, sample_events_window: tuple[NormalizedEvent, ...]
    ) -> None:
        """Events are sorted by (period, timestamp)."""
        for i in range(len(sample_events_window) - 1):
            cur = sample_events_window[i]
            nxt = sample_events_window[i + 1]
            assert (cur.period, cur.timestamp) <= (nxt.period, nxt.timestamp)

    def test_spans_15_seconds(
        self, sample_events_window: tuple[NormalizedEvent, ...]
    ) -> None:
        """Time span from first to last event is at most 15 seconds."""
        first = sample_events_window[0]
        last = sample_events_window[-1]
        assert last.timestamp - first.timestamp <= 15.0

    def test_single_period(
        self, sample_events_window: tuple[NormalizedEvent, ...]
    ) -> None:
        """All events are in the same period."""
        periods = {e.period for e in sample_events_window}
        assert len(periods) == 1

    def test_locations_in_range(
        self, sample_events_window: tuple[NormalizedEvent, ...]
    ) -> None:
        """All locations are within the [0, 100] pitch range."""
        for evt in sample_events_window:
            if evt.location is not None:
                x, y = evt.location
                assert 0.0 <= x <= 100.0
                assert 0.0 <= y <= 100.0
            if evt.end_location is not None:
                x, y = evt.end_location
                assert 0.0 <= x <= 100.0
                assert 0.0 <= y <= 100.0

    def test_mixed_event_types(
        self, sample_events_window: tuple[NormalizedEvent, ...]
    ) -> None:
        """Window contains more than one event type."""
        types = {e.event_type for e in sample_events_window}
        assert len(types) > 1

    def test_mixed_teams(
        self, sample_events_window: tuple[NormalizedEvent, ...]
    ) -> None:
        """Window contains events from both teams."""
        teams = {e.team_id for e in sample_events_window}
        assert len(teams) == 2

    def test_has_incomplete_pass(
        self, sample_events_window: tuple[NormalizedEvent, ...]
    ) -> None:
        """Window contains at least one incomplete pass."""
        incomplete = [
            e
            for e in sample_events_window
            if e.event_type == "pass" and e.event_outcome == "incomplete"
        ]
        assert len(incomplete) >= 1

    def test_no_freeze_frames(
        self, sample_events_window: tuple[NormalizedEvent, ...]
    ) -> None:
        """None of the window events carry freeze-frame data."""
        for evt in sample_events_window:
            assert evt.freeze_frame is None


# ------------------------------------------------------------------
# sample_events_with_360
# ------------------------------------------------------------------


class TestSampleEventsWith360:
    """Validate the 360-enriched event fixture."""

    def test_type(self, sample_events_with_360: tuple[NormalizedEvent, ...]) -> None:
        """Fixture returns a tuple."""
        assert isinstance(sample_events_with_360, tuple)

    def test_count(self, sample_events_with_360: tuple[NormalizedEvent, ...]) -> None:
        """Fixture contains exactly 8 events."""
        assert len(sample_events_with_360) == 8

    def test_all_normalized_events(
        self, sample_events_with_360: tuple[NormalizedEvent, ...]
    ) -> None:
        """Every element is a NormalizedEvent."""
        for evt in sample_events_with_360:
            assert isinstance(evt, NormalizedEvent)

    def test_all_types_controlled(
        self, sample_events_with_360: tuple[NormalizedEvent, ...]
    ) -> None:
        """Every event type is in the controlled vocabulary."""
        for evt in sample_events_with_360:
            assert evt.event_type in CONTROLLED_EVENT_TYPES

    def test_sorted_by_period_timestamp(
        self, sample_events_with_360: tuple[NormalizedEvent, ...]
    ) -> None:
        """Events are sorted by (period, timestamp)."""
        for i in range(len(sample_events_with_360) - 1):
            cur = sample_events_with_360[i]
            nxt = sample_events_with_360[i + 1]
            assert (cur.period, cur.timestamp) <= (nxt.period, nxt.timestamp)

    def test_has_freeze_frames(
        self, sample_events_with_360: tuple[NormalizedEvent, ...]
    ) -> None:
        """At least one event has a non-None freeze_frame."""
        ff_events = [e for e in sample_events_with_360 if e.freeze_frame is not None]
        assert len(ff_events) >= 1

    def test_freeze_frame_count(
        self, sample_events_with_360: tuple[NormalizedEvent, ...]
    ) -> None:
        """Exactly 5 of 8 events carry freeze-frame data."""
        ff_count = sum(1 for e in sample_events_with_360 if e.freeze_frame is not None)
        assert ff_count == 5

    def test_some_without_freeze_frame(
        self, sample_events_with_360: tuple[NormalizedEvent, ...]
    ) -> None:
        """At least one event has no freeze-frame data."""
        no_ff = [e for e in sample_events_with_360 if e.freeze_frame is None]
        assert len(no_ff) >= 1

    def test_freeze_frame_player_instances(
        self, sample_events_with_360: tuple[NormalizedEvent, ...]
    ) -> None:
        """All freeze-frame entries are FreezeFramePlayer instances."""
        for evt in sample_events_with_360:
            if evt.freeze_frame is not None:
                assert len(evt.freeze_frame) > 0
                for player in evt.freeze_frame:
                    assert isinstance(player, FreezeFramePlayer)

    def test_freeze_frame_locations_in_range(
        self, sample_events_with_360: tuple[NormalizedEvent, ...]
    ) -> None:
        """All freeze-frame player locations are within [0, 100]."""
        for evt in sample_events_with_360:
            if evt.freeze_frame is not None:
                for player in evt.freeze_frame:
                    px, py = player.location
                    assert 0.0 <= px <= 100.0
                    assert 0.0 <= py <= 100.0

    def test_freeze_frame_has_teammates_and_opponents(
        self, sample_events_with_360: tuple[NormalizedEvent, ...]
    ) -> None:
        """At least one freeze frame contains both teammates and opponents."""
        found_mixed = False
        for evt in sample_events_with_360:
            if evt.freeze_frame is not None:
                has_teammate = any(p.teammate for p in evt.freeze_frame)
                has_opponent = any(not p.teammate for p in evt.freeze_frame)
                if has_teammate and has_opponent:
                    found_mixed = True
                    break
        assert found_mixed

    def test_contains_shot_with_freeze_frame(
        self, sample_events_with_360: tuple[NormalizedEvent, ...]
    ) -> None:
        """A shot event exists and carries a freeze frame."""
        shots = [
            e
            for e in sample_events_with_360
            if e.event_type == "shot" and e.freeze_frame is not None
        ]
        assert len(shots) == 1

    def test_locations_in_range(
        self, sample_events_with_360: tuple[NormalizedEvent, ...]
    ) -> None:
        """All event locations are within the [0, 100] pitch range."""
        for evt in sample_events_with_360:
            if evt.location is not None:
                x, y = evt.location
                assert 0.0 <= x <= 100.0
                assert 0.0 <= y <= 100.0
            if evt.end_location is not None:
                x, y = evt.end_location
                assert 0.0 <= x <= 100.0
                assert 0.0 <= y <= 100.0

    def test_mixed_teams(
        self, sample_events_with_360: tuple[NormalizedEvent, ...]
    ) -> None:
        """Events come from both teams."""
        teams = {e.team_id for e in sample_events_with_360}
        assert len(teams) == 2


# ------------------------------------------------------------------
# sample_feature_df
# ------------------------------------------------------------------


class TestSampleFeatureDf:
    """Validate the synthetic feature DataFrame fixture."""

    def test_type(self, sample_feature_df: pl.DataFrame) -> None:
        """Fixture returns a polars DataFrame."""
        assert isinstance(sample_feature_df, pl.DataFrame)

    def test_row_count(self, sample_feature_df: pl.DataFrame) -> None:
        """DataFrame has exactly 100 rows."""
        assert sample_feature_df.height == N_ROWS

    def test_metadata_columns_present(self, sample_feature_df: pl.DataFrame) -> None:
        """All 7 metadata columns are present."""
        meta = {
            "match_id",
            "team_id",
            "segment_type",
            "start_time",
            "end_time",
            "period",
            "match_minute",
        }
        assert meta.issubset(set(sample_feature_df.columns))

    def test_tier1_columns_present(self, sample_feature_df: pl.DataFrame) -> None:
        """All 55 Tier 1 feature columns are present."""
        cols = set(sample_feature_df.columns)
        for feat in T1_FEATURES:
            assert feat in cols, f"Missing T1 feature: {feat}"

    def test_tier2_columns_present(self, sample_feature_df: pl.DataFrame) -> None:
        """All 24 Tier 2 feature columns are present."""
        cols = set(sample_feature_df.columns)
        for feat in T2_FEATURES:
            assert feat in cols, f"Missing T2 feature: {feat}"

    def test_tier3_columns_present(self, sample_feature_df: pl.DataFrame) -> None:
        """All 19 Tier 3 feature columns are present."""
        cols = set(sample_feature_df.columns)
        for feat in T3_FEATURES:
            assert feat in cols, f"Missing T3 feature: {feat}"

    def test_total_feature_count(self, sample_feature_df: pl.DataFrame) -> None:
        """Feature column counts match the spec (55 + 24 + 19 = 98)."""
        assert N_T1 == 55
        assert N_T2 == 24
        assert N_T3 == 19
        assert len(ALL_FEATURES) == 98

    def test_total_column_count(self, sample_feature_df: pl.DataFrame) -> None:
        """DataFrame has 7 metadata + 98 feature = 105 columns."""
        assert sample_feature_df.width == 7 + len(ALL_FEATURES)

    def test_tier1_no_nulls(self, sample_feature_df: pl.DataFrame) -> None:
        """Tier 1 features have no null values."""
        for feat in T1_FEATURES:
            null_count = sample_feature_df[feat].null_count()
            assert null_count == 0, f"T1 feature {feat} has {null_count} nulls"

    def test_tier2_no_nulls(self, sample_feature_df: pl.DataFrame) -> None:
        """Tier 2 features have no null values."""
        for feat in T2_FEATURES:
            null_count = sample_feature_df[feat].null_count()
            assert null_count == 0, f"T2 feature {feat} has {null_count} nulls"

    def test_tier3_has_nulls(self, sample_feature_df: pl.DataFrame) -> None:
        """Tier 3 features contain nulls (matches without 360 data)."""
        total_nulls = sum(sample_feature_df[feat].null_count() for feat in T3_FEATURES)
        assert total_nulls > 0

    def test_tier3_not_all_null(self, sample_feature_df: pl.DataFrame) -> None:
        """Tier 3 features are populated for some rows."""
        for feat in T3_FEATURES:
            non_null = N_ROWS - sample_feature_df[feat].null_count()
            assert non_null > 0, f"T3 feature {feat} is entirely null"

    def test_tier3_null_pattern_consistent(
        self, sample_feature_df: pl.DataFrame
    ) -> None:
        """All Tier 3 features are null/non-null on the same rows."""
        first_null_mask = sample_feature_df[T3_FEATURES[0]].is_null()
        for feat in T3_FEATURES[1:]:
            mask = sample_feature_df[feat].is_null()
            assert (first_null_mask == mask).all(), (
                f"T3 null pattern mismatch: {T3_FEATURES[0]} vs {feat}"
            )

    def test_segment_type_values(self, sample_feature_df: pl.DataFrame) -> None:
        """All segment_type values are 'window'."""
        unique = sample_feature_df["segment_type"].unique().to_list()
        assert unique == ["window"]

    def test_period_values(self, sample_feature_df: pl.DataFrame) -> None:
        """Period column contains only 1 or 2."""
        periods = set(sample_feature_df["period"].to_list())
        assert periods.issubset({1, 2})

    def test_start_before_end(self, sample_feature_df: pl.DataFrame) -> None:
        """start_time is strictly less than end_time for every row."""
        starts = sample_feature_df["start_time"]
        ends = sample_feature_df["end_time"]
        assert (starts < ends).all()

    def test_match_minute_non_negative(self, sample_feature_df: pl.DataFrame) -> None:
        """match_minute is non-negative for all rows."""
        assert (sample_feature_df["match_minute"] >= 0.0).all()

    def test_deterministic(self, sample_feature_df: pl.DataFrame) -> None:
        """Fixture produces identical results across invocations.

        Rebuilds the DataFrame from scratch using the same seed and
        compares a handful of cells to confirm reproducibility.
        """
        import numpy as np

        from tests.conftest import (
            _BINARY_FEATURES,
            _FEATURE_RANGES,
            _INTEGER_FEATURES,
            _T3_AVAILABILITY,
            ALL_FEATURES,
            N_ROWS,
        )

        rng = np.random.default_rng(seed=42)
        n = N_ROWS

        # Burn the same draws the fixture makes for metadata columns.
        np.linspace(0.0, 85.0 * 60, n)  # start_times

        has_360 = rng.random(n) < _T3_AVAILABILITY

        rebuilt: dict[str, list[float | None]] = {}
        for feat in ALL_FEATURES:
            low, high = _FEATURE_RANGES.get(feat, (0.0, 1.0))
            values = rng.uniform(low, high, size=n)
            if feat in _BINARY_FEATURES:
                values = rng.integers(0, 2, size=n).astype(float)
            elif feat in _INTEGER_FEATURES:
                values = np.round(values).clip(low, high)
            is_t3 = feat.startswith("t3_")
            col: list[float | None] = []
            for i in range(n):
                if is_t3 and not has_360[i]:
                    col.append(None)
                else:
                    col.append(float(values[i]))
            rebuilt[feat] = col

        for feat in (T1_FEATURES[0], T2_FEATURES[0], T3_FEATURES[0]):
            for row_idx in (0, 50, 99):
                expected = rebuilt[feat][row_idx]
                actual = sample_feature_df[feat][row_idx]
                assert actual == expected, (
                    f"Mismatch at [{row_idx}][{feat}]: {actual} != {expected}"
                )

    def test_feature_naming_convention(self, sample_feature_df: pl.DataFrame) -> None:
        """All feature columns follow the t{tier}_{name} convention."""
        meta = {
            "match_id",
            "team_id",
            "segment_type",
            "start_time",
            "end_time",
            "period",
            "match_minute",
        }
        for col in sample_feature_df.columns:
            if col in meta:
                continue
            assert col.startswith(("t1_", "t2_", "t3_")), (
                f"Feature column {col!r} does not match naming convention"
            )
