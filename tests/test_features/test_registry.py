"""Tests for the feature extractor registry.

Validates registration, tier-based filtering, feature-name collection,
merged extraction, and duplicate-name detection.
"""

from __future__ import annotations

import pytest

from tactical.adapters.schemas import MatchContext, NormalizedEvent
from tactical.features.base import FeatureExtractor
from tactical.features.registry import FeatureRegistry

# ------------------------------------------------------------------
# Mock extractors
# ------------------------------------------------------------------


class _MockTier1Extractor:
    """Tier 1 extractor producing two fixed features."""

    @property
    def tier(self) -> int:
        return 1

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ("t1_pass_count", "t1_shot_count")

    def extract(
        self,
        events: tuple[NormalizedEvent, ...],
        context: MatchContext,
    ) -> dict[str, float | None]:
        return {"t1_pass_count": 5.0, "t1_shot_count": 2.0}


class _MockTier2Extractor:
    """Tier 2 extractor producing two fixed features."""

    @property
    def tier(self) -> int:
        return 2

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ("t2_pressing_intensity", "t2_ppda")

    def extract(
        self,
        events: tuple[NormalizedEvent, ...],
        context: MatchContext,
    ) -> dict[str, float | None]:
        return {"t2_pressing_intensity": 0.75, "t2_ppda": 12.3}


class _MockTier3Extractor:
    """Tier 3 extractor producing three fixed features."""

    @property
    def tier(self) -> int:
        return 3

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ("t3_convex_hull_area", "t3_team_width", "t3_defensive_line")

    def extract(
        self,
        events: tuple[NormalizedEvent, ...],
        context: MatchContext,
    ) -> dict[str, float | None]:
        return {
            "t3_convex_hull_area": 450.0,
            "t3_team_width": 55.2,
            "t3_defensive_line": 35.0,
        }


class _MockDuplicateExtractor:
    """Tier 1 extractor that collides with :class:`_MockTier1Extractor`."""

    @property
    def tier(self) -> int:
        return 1

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ("t1_pass_count",)

    def extract(
        self,
        events: tuple[NormalizedEvent, ...],
        context: MatchContext,
    ) -> dict[str, float | None]:
        return {"t1_pass_count": 99.0}


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

_CONTEXT = MatchContext(
    match_id="match_001",
    team_id="team_a",
    opponent_id="team_b",
    team_is_home=True,
    has_360=False,
)

_EVENTS: tuple[NormalizedEvent, ...] = (
    NormalizedEvent(
        event_id="evt_001",
        match_id="match_001",
        team_id="team_a",
        player_id="player_01",
        period=1,
        timestamp=10.0,
        match_minute=0.2,
        location=(30.0, 40.0),
        end_location=(50.0, 45.0),
        event_type="pass",
        event_outcome="complete",
        under_pressure=False,
        body_part="right_foot",
        freeze_frame=None,
        score_home=0,
        score_away=0,
        team_is_home=True,
    ),
)


@pytest.fixture()
def registry() -> FeatureRegistry:
    """Return an empty :class:`FeatureRegistry`."""
    return FeatureRegistry()


# ------------------------------------------------------------------
# Protocol conformance sanity check
# ------------------------------------------------------------------


def test_mock_satisfies_protocol() -> None:
    """Mock extractors satisfy the runtime-checkable Protocol."""
    assert isinstance(_MockTier1Extractor(), FeatureExtractor)
    assert isinstance(_MockTier2Extractor(), FeatureExtractor)
    assert isinstance(_MockTier3Extractor(), FeatureExtractor)


# ------------------------------------------------------------------
# Registration & retrieval
# ------------------------------------------------------------------


def test_register_and_retrieve(registry: FeatureRegistry) -> None:
    """Registered extractor is returned by :meth:`get_active_extractors`."""
    ext = _MockTier1Extractor()
    registry.register(ext)

    active = registry.get_active_extractors(max_tier=1)
    assert len(active) == 1
    assert active[0] is ext


# ------------------------------------------------------------------
# Tier filtering
# ------------------------------------------------------------------


def test_tier_filtering(registry: FeatureRegistry) -> None:
    """Only extractors at or below *max_tier* are returned."""
    t1 = _MockTier1Extractor()
    t3 = _MockTier3Extractor()
    registry.register(t1)
    registry.register(t3)

    active = registry.get_active_extractors(max_tier=2)
    assert active == [t1]


def test_tier_filtering_includes_boundary(registry: FeatureRegistry) -> None:
    """Tier equal to *max_tier* is included."""
    t1 = _MockTier1Extractor()
    t2 = _MockTier2Extractor()
    t3 = _MockTier3Extractor()
    registry.register(t1)
    registry.register(t2)
    registry.register(t3)

    active = registry.get_active_extractors(max_tier=3)
    assert len(active) == 3


# ------------------------------------------------------------------
# Feature name collection
# ------------------------------------------------------------------


def test_get_all_feature_names(registry: FeatureRegistry) -> None:
    """Combined feature names from active tiers are returned."""
    registry.register(_MockTier1Extractor())
    registry.register(_MockTier2Extractor())
    registry.register(_MockTier3Extractor())

    names = registry.get_all_feature_names(max_tier=2)
    assert names == (
        "t1_pass_count",
        "t1_shot_count",
        "t2_pressing_intensity",
        "t2_ppda",
    )


def test_get_all_feature_names_empty(registry: FeatureRegistry) -> None:
    """No extractors yields an empty tuple."""
    assert registry.get_all_feature_names(max_tier=3) == ()


# ------------------------------------------------------------------
# Merged extraction
# ------------------------------------------------------------------


def test_extract_all_merges(registry: FeatureRegistry) -> None:
    """Results from multiple extractors are merged into a single dict."""
    registry.register(_MockTier1Extractor())
    registry.register(_MockTier2Extractor())

    result = registry.extract_all(_EVENTS, _CONTEXT, max_tier=2)
    assert result == {
        "t1_pass_count": 5.0,
        "t1_shot_count": 2.0,
        "t2_pressing_intensity": 0.75,
        "t2_ppda": 12.3,
    }


def test_extract_all_respects_tier(registry: FeatureRegistry) -> None:
    """Tier 3 extractor is skipped when *max_tier* is 2."""
    registry.register(_MockTier1Extractor())
    registry.register(_MockTier3Extractor())

    result = registry.extract_all(_EVENTS, _CONTEXT, max_tier=2)
    assert "t3_convex_hull_area" not in result
    assert result == {"t1_pass_count": 5.0, "t1_shot_count": 2.0}


# ------------------------------------------------------------------
# Duplicate feature name detection
# ------------------------------------------------------------------


def test_duplicate_feature_name_raises(registry: FeatureRegistry) -> None:
    """Two extractors producing the same feature name raise ValueError."""
    registry.register(_MockTier1Extractor())
    registry.register(_MockDuplicateExtractor())

    with pytest.raises(ValueError, match="t1_pass_count"):
        registry.extract_all(_EVENTS, _CONTEXT, max_tier=1)
