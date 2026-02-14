"""Tests for Tier 3 freeze-frame / 360 feature extractors.

Validates formation features (centroid, spread, convex hull, defensive /
midfield / attacking lines, eccentricity) and relational features
(nearest-opponent distances, opponent centroid, opponent defensive line),
including edge cases for missing freeze frames and insufficient players.
"""

from __future__ import annotations

import pytest

from tactical.adapters.schemas import (
    FreezeFramePlayer,
    MatchContext,
    NormalizedEvent,
)
from tactical.features.tier3 import (
    FormationFeatureExtractor,
    RelationalFeatureExtractor,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _evt(
    event_id: str,
    *,
    team_id: str = "team_a",
    timestamp: float,
    match_minute: float,
    location: tuple[float, float] = (50.0, 50.0),
    end_location: tuple[float, float] | None = None,
    period: int = 1,
    event_type: str = "pass",
    event_outcome: str = "complete",
    under_pressure: bool = False,
    body_part: str = "right_foot",
    freeze_frame: tuple[FreezeFramePlayer, ...] | None = None,
    score_home: int = 0,
    score_away: int = 0,
    team_is_home: bool = True,
) -> NormalizedEvent:
    """Build a :class:`NormalizedEvent` with sensible defaults."""
    return NormalizedEvent(
        event_id=event_id,
        match_id="match_001",
        team_id=team_id,
        player_id="player_01",
        period=period,
        timestamp=timestamp,
        match_minute=match_minute,
        location=location,
        end_location=end_location,
        event_type=event_type,
        event_outcome=event_outcome,
        under_pressure=under_pressure,
        body_part=body_part,
        freeze_frame=freeze_frame,
        score_home=score_home,
        score_away=score_away,
        team_is_home=team_is_home,
    )


# ------------------------------------------------------------------
# Shared context
# ------------------------------------------------------------------

_CONTEXT = MatchContext(
    match_id="match_001",
    team_id="team_a",
    opponent_id="team_b",
    team_is_home=True,
    has_360=True,
)

# ------------------------------------------------------------------
# Known positions for deterministic assertions
# ------------------------------------------------------------------

# 6 teammates at controlled positions
_TEAMMATES: tuple[FreezeFramePlayer, ...] = (
    FreezeFramePlayer(
        player_id="t1",
        teammate=True,
        location=(20.0, 30.0),
        position="Center Back",
    ),
    FreezeFramePlayer(
        player_id="t2",
        teammate=True,
        location=(25.0, 60.0),
        position="Center Back",
    ),
    FreezeFramePlayer(
        player_id="t3",
        teammate=True,
        location=(40.0, 20.0),
        position="Left Back",
    ),
    FreezeFramePlayer(
        player_id="t4",
        teammate=True,
        location=(45.0, 50.0),
        position="Central Midfield",
    ),
    FreezeFramePlayer(
        player_id="t5",
        teammate=True,
        location=(70.0, 40.0),
        position="Right Wing",
    ),
    FreezeFramePlayer(
        player_id="t6",
        teammate=True,
        location=(80.0, 55.0),
        position="Center Forward",
    ),
)

# 5 opponents at controlled positions
_OPPONENTS: tuple[FreezeFramePlayer, ...] = (
    FreezeFramePlayer(
        player_id="o1",
        teammate=False,
        location=(30.0, 45.0),
        position="Center Back",
    ),
    FreezeFramePlayer(
        player_id="o2",
        teammate=False,
        location=(40.0, 55.0),
        position="Center Back",
    ),
    FreezeFramePlayer(
        player_id="o3",
        teammate=False,
        location=(55.0, 35.0),
        position="Central Midfield",
    ),
    FreezeFramePlayer(
        player_id="o4",
        teammate=False,
        location=(65.0, 50.0),
        position="Left Wing",
    ),
    FreezeFramePlayer(
        player_id="o5",
        teammate=False,
        location=(75.0, 60.0),
        position="Center Forward",
    ),
)

_FREEZE_FRAME: tuple[FreezeFramePlayer, ...] = (*_TEAMMATES, *_OPPONENTS)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def formation_ext() -> FormationFeatureExtractor:
    """Instantiate a :class:`FormationFeatureExtractor`."""
    return FormationFeatureExtractor()


@pytest.fixture()
def relational_ext() -> RelationalFeatureExtractor:
    """Instantiate a :class:`RelationalFeatureExtractor`."""
    return RelationalFeatureExtractor()


@pytest.fixture()
def events_with_ff() -> tuple[NormalizedEvent, ...]:
    """Events where the last event with a freeze frame is the third.

    The fixture ensures the extractor picks the *last* event with a
    non-``None`` freeze frame.
    """
    return (
        _evt(
            "e1",
            timestamp=100.0,
            match_minute=1.0,
            freeze_frame=None,
        ),
        _evt(
            "e2",
            timestamp=102.0,
            match_minute=1.5,
            freeze_frame=_FREEZE_FRAME,
        ),
        _evt(
            "e3",
            timestamp=104.0,
            match_minute=2.0,
            freeze_frame=_FREEZE_FRAME,
        ),
        _evt(
            "e4",
            timestamp=106.0,
            match_minute=2.5,
            freeze_frame=None,
        ),
    )


@pytest.fixture()
def events_no_ff() -> tuple[NormalizedEvent, ...]:
    """Events with no freeze-frame data at all."""
    return (
        _evt("n1", timestamp=200.0, match_minute=3.0),
        _evt("n2", timestamp=202.0, match_minute=3.5),
    )


# ------------------------------------------------------------------
# Formation tests
# ------------------------------------------------------------------


class TestFormationCentroid:
    """Verify team centroid computation from known positions."""

    def test_formation_centroid(
        self,
        formation_ext: FormationFeatureExtractor,
        events_with_ff: tuple[NormalizedEvent, ...],
    ) -> None:
        """Centroid should be the mean of all 6 teammate positions."""
        result = formation_ext.extract(events_with_ff, _CONTEXT)

        # Teammate x: 20, 25, 40, 45, 70, 80  -> mean = 280/6
        expected_x = (20.0 + 25.0 + 40.0 + 45.0 + 70.0 + 80.0) / 6.0
        # Teammate y: 30, 60, 20, 50, 40, 55  -> mean = 255/6
        expected_y = (30.0 + 60.0 + 20.0 + 50.0 + 40.0 + 55.0) / 6.0

        assert result["t3_formation_team_centroid_x"] == pytest.approx(
            expected_x,
            abs=1e-6,
        )
        assert result["t3_formation_team_centroid_y"] == pytest.approx(
            expected_y,
            abs=1e-6,
        )


class TestFormationConvexHull:
    """Verify convex hull area is positive and reasonable."""

    def test_formation_convex_hull(
        self,
        formation_ext: FormationFeatureExtractor,
        events_with_ff: tuple[NormalizedEvent, ...],
    ) -> None:
        """Hull area should be positive; perimeter should be positive."""
        result = formation_ext.extract(events_with_ff, _CONTEXT)

        area = result["t3_formation_convex_hull_area"]
        perimeter = result["t3_formation_convex_hull_perimeter"]

        assert area is not None
        assert perimeter is not None
        assert area > 0.0
        assert perimeter > 0.0
        # Pitch is 100x100 => max area = 10000. Our positions span
        # roughly 60 x 40, so hull area should be well below 5000.
        assert area < 5000.0

    def test_convex_hull_collinear(
        self,
        formation_ext: FormationFeatureExtractor,
    ) -> None:
        """Collinear teammates should yield None for hull features."""
        collinear_tm: tuple[FreezeFramePlayer, ...] = tuple(
            FreezeFramePlayer(
                player_id=f"c{i}",
                teammate=True,
                location=(float(10 * i), 50.0),
                position="Center Back",
            )
            for i in range(4)
        )
        ff = (*collinear_tm, *_OPPONENTS)
        events = (_evt("col1", timestamp=300.0, match_minute=5.0, freeze_frame=ff),)
        result = formation_ext.extract(events, _CONTEXT)

        assert result["t3_formation_convex_hull_area"] is None
        assert result["t3_formation_convex_hull_perimeter"] is None
        # Other formation features should still be computed
        assert result["t3_formation_team_centroid_x"] is not None


class TestFormationDefensiveLine:
    """Verify defensive, midfield, and attacking line computation."""

    def test_formation_defensive_line(
        self,
        formation_ext: FormationFeatureExtractor,
        events_with_ff: tuple[NormalizedEvent, ...],
    ) -> None:
        """Defensive line = mean x of 3 lowest-x teammates."""
        result = formation_ext.extract(events_with_ff, _CONTEXT)

        # Sorted x: 20, 25, 40, 45, 70, 80
        # 3 lowest: 20, 25, 40 -> mean = 85/3
        expected_def = (20.0 + 25.0 + 40.0) / 3.0
        assert result["t3_formation_defensive_line"] == pytest.approx(
            expected_def,
            abs=1e-6,
        )

    def test_formation_midfield_line(
        self,
        formation_ext: FormationFeatureExtractor,
        events_with_ff: tuple[NormalizedEvent, ...],
    ) -> None:
        """Midfield line = median x of all teammates."""
        result = formation_ext.extract(events_with_ff, _CONTEXT)

        # Sorted x: 20, 25, 40, 45, 70, 80 -> median = (40+45)/2 = 42.5
        assert result["t3_formation_midfield_line"] == pytest.approx(42.5, abs=1e-6)

    def test_formation_attacking_line(
        self,
        formation_ext: FormationFeatureExtractor,
        events_with_ff: tuple[NormalizedEvent, ...],
    ) -> None:
        """Attacking line = mean x of 3 highest-x teammates."""
        result = formation_ext.extract(events_with_ff, _CONTEXT)

        # 3 highest: 45, 70, 80 -> mean = 195/3 = 65.0
        expected_att = (45.0 + 70.0 + 80.0) / 3.0
        assert result["t3_formation_attacking_line"] == pytest.approx(
            expected_att,
            abs=1e-6,
        )

    def test_formation_line_gaps(
        self,
        formation_ext: FormationFeatureExtractor,
        events_with_ff: tuple[NormalizedEvent, ...],
    ) -> None:
        """def_mid_gap and mid_att_gap should be consistent."""
        result = formation_ext.extract(events_with_ff, _CONTEXT)

        def_line = result["t3_formation_defensive_line"]
        mid_line = result["t3_formation_midfield_line"]
        att_line = result["t3_formation_attacking_line"]

        assert def_line is not None
        assert mid_line is not None
        assert att_line is not None

        assert result["t3_formation_def_mid_gap"] == pytest.approx(
            mid_line - def_line,
            abs=1e-6,
        )
        assert result["t3_formation_mid_att_gap"] == pytest.approx(
            att_line - mid_line,
            abs=1e-6,
        )


class TestFormationEccentricity:
    """Verify formation eccentricity is in [0, 1]."""

    def test_formation_eccentricity(
        self,
        formation_ext: FormationFeatureExtractor,
        events_with_ff: tuple[NormalizedEvent, ...],
    ) -> None:
        """Eccentricity should be between 0 (circle) and 1 (line)."""
        result = formation_ext.extract(events_with_ff, _CONTEXT)

        ecc = result["t3_formation_formation_eccentricity"]
        assert ecc is not None
        assert 0.0 <= ecc <= 1.0

    def test_eccentricity_circular_formation(
        self,
        formation_ext: FormationFeatureExtractor,
    ) -> None:
        """A perfectly symmetric formation should have low eccentricity."""
        # Square positions: equal spread in x and y
        square_tm: tuple[FreezeFramePlayer, ...] = (
            FreezeFramePlayer(
                player_id="sq1",
                teammate=True,
                location=(40.0, 40.0),
                position="CB",
            ),
            FreezeFramePlayer(
                player_id="sq2",
                teammate=True,
                location=(40.0, 60.0),
                position="CB",
            ),
            FreezeFramePlayer(
                player_id="sq3",
                teammate=True,
                location=(60.0, 40.0),
                position="CM",
            ),
            FreezeFramePlayer(
                player_id="sq4",
                teammate=True,
                location=(60.0, 60.0),
                position="CM",
            ),
        )
        ff = (*square_tm, *_OPPONENTS)
        events = (_evt("sq_e1", timestamp=400.0, match_minute=6.0, freeze_frame=ff),)
        result = formation_ext.extract(events, _CONTEXT)
        ecc = result["t3_formation_formation_eccentricity"]

        assert ecc is not None
        assert ecc == pytest.approx(0.0, abs=1e-6)


class TestFormationSpreadAndDimensions:
    """Verify spread and dimension calculations."""

    def test_formation_spread(
        self,
        formation_ext: FormationFeatureExtractor,
        events_with_ff: tuple[NormalizedEvent, ...],
    ) -> None:
        """Spread should match numpy std (population)."""
        import numpy as np

        result = formation_ext.extract(events_with_ff, _CONTEXT)

        xs = np.array([20.0, 25.0, 40.0, 45.0, 70.0, 80.0])
        ys = np.array([30.0, 60.0, 20.0, 50.0, 40.0, 55.0])

        assert result["t3_formation_team_spread_x"] == pytest.approx(
            float(np.std(xs)),
            abs=1e-6,
        )
        assert result["t3_formation_team_spread_y"] == pytest.approx(
            float(np.std(ys)),
            abs=1e-6,
        )

    def test_formation_dimensions(
        self,
        formation_ext: FormationFeatureExtractor,
        events_with_ff: tuple[NormalizedEvent, ...],
    ) -> None:
        """Width = max_y - min_y, length = max_x - min_x."""
        result = formation_ext.extract(events_with_ff, _CONTEXT)

        # y: 20..60, x: 20..80
        assert result["t3_formation_team_width"] == pytest.approx(40.0, abs=1e-6)
        assert result["t3_formation_team_length"] == pytest.approx(60.0, abs=1e-6)


# ------------------------------------------------------------------
# Relational tests
# ------------------------------------------------------------------


class TestRelationalNearestOpponent:
    """Verify nearest-opponent distance calculations."""

    def test_relational_nearest_opponent(
        self,
        relational_ext: RelationalFeatureExtractor,
        events_with_ff: tuple[NormalizedEvent, ...],
    ) -> None:
        """Avg and min nearest-opponent should be positive and consistent."""
        result = relational_ext.extract(events_with_ff, _CONTEXT)

        avg_dist = result["t3_relational_avg_nearest_opponent_dist"]
        min_dist = result["t3_relational_min_nearest_opponent_dist"]

        assert avg_dist is not None
        assert min_dist is not None
        assert min_dist > 0.0
        assert avg_dist >= min_dist

    def test_relational_nearest_opponent_exact(
        self,
        relational_ext: RelationalFeatureExtractor,
    ) -> None:
        """Check nearest-opponent distances for simple known positions."""
        tm = (
            FreezeFramePlayer(
                player_id="st1",
                teammate=True,
                location=(0.0, 0.0),
                position="CB",
            ),
            FreezeFramePlayer(
                player_id="st2",
                teammate=True,
                location=(10.0, 0.0),
                position="CM",
            ),
            FreezeFramePlayer(
                player_id="st3",
                teammate=True,
                location=(20.0, 0.0),
                position="CF",
            ),
        )
        opp = (
            FreezeFramePlayer(
                player_id="so1",
                teammate=False,
                location=(3.0, 4.0),
                position="CB",
            ),
            FreezeFramePlayer(
                player_id="so2",
                teammate=False,
                location=(10.0, 3.0),
                position="CM",
            ),
            FreezeFramePlayer(
                player_id="so3",
                teammate=False,
                location=(18.0, 0.0),
                position="CF",
            ),
        )
        ff = (*tm, *opp)
        events = (_evt("ex1", timestamp=500.0, match_minute=8.0, freeze_frame=ff),)
        result = relational_ext.extract(events, _CONTEXT)

        # Teammate (0,0) nearest opponent: (3,4) -> dist=5.0
        # Teammate (10,0) nearest opponent: (10,3) -> dist=3.0
        # Teammate (20,0) nearest opponent: (18,0) -> dist=2.0
        avg_expected = (5.0 + 3.0 + 2.0) / 3.0
        min_expected = 2.0

        assert result["t3_relational_avg_nearest_opponent_dist"] == pytest.approx(
            avg_expected,
            abs=1e-6,
        )
        assert result["t3_relational_min_nearest_opponent_dist"] == pytest.approx(
            min_expected,
            abs=1e-6,
        )


class TestRelationalOpponentMetrics:
    """Verify opponent centroid, spread, and defensive line."""

    def test_opp_centroid_and_spread(
        self,
        relational_ext: RelationalFeatureExtractor,
        events_with_ff: tuple[NormalizedEvent, ...],
    ) -> None:
        """Opponent centroid_x = mean of opponent x positions."""
        import numpy as np

        result = relational_ext.extract(events_with_ff, _CONTEXT)

        opp_xs = np.array([30.0, 40.0, 55.0, 65.0, 75.0])
        expected_cx = float(np.mean(opp_xs))
        expected_sx = float(np.std(opp_xs))

        assert result["t3_relational_opp_centroid_x"] == pytest.approx(
            expected_cx,
            abs=1e-6,
        )
        assert result["t3_relational_opp_spread_x"] == pytest.approx(
            expected_sx,
            abs=1e-6,
        )

    def test_opp_defensive_line(
        self,
        relational_ext: RelationalFeatureExtractor,
        events_with_ff: tuple[NormalizedEvent, ...],
    ) -> None:
        """Opp defensive line = mean x of 4 deepest (highest x) opponents."""
        result = relational_ext.extract(events_with_ff, _CONTEXT)

        # Sorted opp x: 30, 40, 55, 65, 75 -> 4 highest: 40, 55, 65, 75
        expected = (40.0 + 55.0 + 65.0 + 75.0) / 4.0
        assert result["t3_relational_opp_defensive_line"] == pytest.approx(
            expected,
            abs=1e-6,
        )


# ------------------------------------------------------------------
# Edge-case tests
# ------------------------------------------------------------------


class TestNoFreezeFrame:
    """All features return None when no event has freeze-frame data."""

    def test_no_freeze_frame_returns_none(
        self,
        formation_ext: FormationFeatureExtractor,
        relational_ext: RelationalFeatureExtractor,
        events_no_ff: tuple[NormalizedEvent, ...],
    ) -> None:
        """Every feature value should be None."""
        form_result = formation_ext.extract(events_no_ff, _CONTEXT)
        rel_result = relational_ext.extract(events_no_ff, _CONTEXT)

        for name, value in form_result.items():
            assert value is None, f"{name} should be None"

        for name, value in rel_result.items():
            assert value is None, f"{name} should be None"


class TestInsufficientTeammates:
    """Formation features return None with < 3 teammates."""

    def test_insufficient_teammates(
        self,
        formation_ext: FormationFeatureExtractor,
    ) -> None:
        """Two teammates is below the threshold of 3."""
        few_tm: tuple[FreezeFramePlayer, ...] = (
            FreezeFramePlayer(
                player_id="ft1",
                teammate=True,
                location=(30.0, 40.0),
                position="CB",
            ),
            FreezeFramePlayer(
                player_id="ft2",
                teammate=True,
                location=(50.0, 50.0),
                position="CM",
            ),
        )
        ff = (*few_tm, *_OPPONENTS)
        events = (_evt("ins1", timestamp=600.0, match_minute=10.0, freeze_frame=ff),)
        result = formation_ext.extract(events, _CONTEXT)

        for name, value in result.items():
            assert value is None, f"{name} should be None with < 3 teammates"


class TestInsufficientOpponents:
    """Relational features return None with < 3 opponents."""

    def test_insufficient_opponents(
        self,
        relational_ext: RelationalFeatureExtractor,
    ) -> None:
        """Two opponents is below the threshold of 3."""
        few_opp: tuple[FreezeFramePlayer, ...] = (
            FreezeFramePlayer(
                player_id="fo1",
                teammate=False,
                location=(40.0, 50.0),
                position="CB",
            ),
            FreezeFramePlayer(
                player_id="fo2",
                teammate=False,
                location=(60.0, 50.0),
                position="CM",
            ),
        )
        tm = _TEAMMATES[:4]
        ff = (*tm, *few_opp)
        events = (_evt("ins2", timestamp=700.0, match_minute=11.0, freeze_frame=ff),)
        result = relational_ext.extract(events, _CONTEXT)

        for name, value in result.items():
            assert value is None, f"{name} should be None with < 3 opponents"


# ------------------------------------------------------------------
# Tier and naming tests
# ------------------------------------------------------------------


class TestTierIsThree:
    """Both extractors should report tier = 3."""

    def test_tier_is_three(
        self,
        formation_ext: FormationFeatureExtractor,
        relational_ext: RelationalFeatureExtractor,
    ) -> None:
        """Tier property must return 3."""
        assert formation_ext.tier == 3
        assert relational_ext.tier == 3


class TestNamingConventions:
    """Feature names follow the t3_ prefix convention."""

    def test_formation_names_prefixed(
        self,
        formation_ext: FormationFeatureExtractor,
    ) -> None:
        """All formation feature names start with ``t3_formation_``."""
        for name in formation_ext.feature_names:
            assert name.startswith("t3_formation_"), name

    def test_relational_names_prefixed(
        self,
        relational_ext: RelationalFeatureExtractor,
    ) -> None:
        """All relational feature names start with ``t3_relational_``."""
        for name in relational_ext.feature_names:
            assert name.startswith("t3_relational_"), name

    def test_extract_keys_match_feature_names(
        self,
        formation_ext: FormationFeatureExtractor,
        relational_ext: RelationalFeatureExtractor,
        events_with_ff: tuple[NormalizedEvent, ...],
    ) -> None:
        """Extracted dict keys must exactly match declared feature_names."""
        form_result = formation_ext.extract(events_with_ff, _CONTEXT)
        rel_result = relational_ext.extract(events_with_ff, _CONTEXT)

        assert set(form_result.keys()) == set(formation_ext.feature_names)
        assert set(rel_result.keys()) == set(relational_ext.feature_names)

    def test_no_duplicate_names(
        self,
        formation_ext: FormationFeatureExtractor,
        relational_ext: RelationalFeatureExtractor,
    ) -> None:
        """No name overlap between the two extractors."""
        overlap = set(formation_ext.feature_names) & set(
            relational_ext.feature_names,
        )
        assert len(overlap) == 0, f"Overlapping names: {overlap}"


class TestLastFreezeFrameSelection:
    """Extractor should use the LAST event with freeze-frame data."""

    def test_uses_last_freeze_frame(
        self,
        formation_ext: FormationFeatureExtractor,
    ) -> None:
        """When two events have different freeze frames, the last wins."""
        early_tm: tuple[FreezeFramePlayer, ...] = tuple(
            FreezeFramePlayer(
                player_id=f"e{i}",
                teammate=True,
                location=(10.0, 10.0),
                position="CB",
            )
            for i in range(4)
        )
        late_tm: tuple[FreezeFramePlayer, ...] = tuple(
            FreezeFramePlayer(
                player_id=f"l{i}",
                teammate=True,
                location=(50.0, 50.0),
                position="CB",
            )
            for i in range(4)
        )
        early_ff = (*early_tm, *_OPPONENTS)
        late_ff = (*late_tm, *_OPPONENTS)

        events = (
            _evt("lff1", timestamp=100.0, match_minute=1.0, freeze_frame=early_ff),
            _evt("lff2", timestamp=102.0, match_minute=1.5, freeze_frame=late_ff),
            _evt("lff3", timestamp=104.0, match_minute=2.0),
        )
        result = formation_ext.extract(events, _CONTEXT)

        # Late teammates are all at (50, 50) -> centroid = (50, 50)
        assert result["t3_formation_team_centroid_x"] == pytest.approx(50.0, abs=1e-6)
        assert result["t3_formation_team_centroid_y"] == pytest.approx(50.0, abs=1e-6)
