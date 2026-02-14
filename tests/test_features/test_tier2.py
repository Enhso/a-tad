"""Tests for Tier 2 event-derived feature extractors.

Validates zonal transitions, box entries/exits, team shape estimation,
pressing metrics, PPDA, and transition detection including edge cases,
naming conventions, and tier declarations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tactical.adapters.schemas import MatchContext, NormalizedEvent
from tactical.features.tier2 import (
    PressingFeatureExtractor,
    TeamShapeFeatureExtractor,
    TransitionFeatureExtractor,
    ZonalFeatureExtractor,
)

if TYPE_CHECKING:
    from tactical.features.base import FeatureExtractor

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _evt(
    event_id: str,
    *,
    team_id: str = "team_a",
    timestamp: float,
    match_minute: float,
    location: tuple[float, float] | None,
    end_location: tuple[float, float] | None = None,
    period: int = 1,
    event_type: str = "pass",
    event_outcome: str = "complete",
    under_pressure: bool = False,
    body_part: str = "right_foot",
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
        freeze_frame=None,
        score_home=score_home,
        score_away=score_away,
        team_is_home=team_is_home,
    )


# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------

_CONTEXT = MatchContext(
    match_id="match_001",
    team_id="team_a",
    opponent_id="team_b",
    team_is_home=True,
    has_360=False,
)


# ------------------------------------------------------------------
# Zonal tests
# ------------------------------------------------------------------


class TestZonalFeatureExtractor:
    """Tests for :class:`ZonalFeatureExtractor`."""

    def test_zonal_transitions(self) -> None:
        """Events moving from defensive to attacking third are counted."""
        events = (
            # defensive third (x=20)
            _evt("z01", timestamp=0.0, match_minute=0.0, location=(20.0, 50.0)),
            # attacking third (x=80)
            _evt("z02", timestamp=1.0, match_minute=0.02, location=(80.0, 50.0)),
            # stays attacking (x=85)
            _evt("z03", timestamp=2.0, match_minute=0.03, location=(85.0, 50.0)),
        )
        ext = ZonalFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        # 2 transitions total: def->att, att->att
        assert result["t2_zonal_zone_def_to_att"] == pytest.approx(0.5)
        assert result["t2_zonal_zone_att_to_att"] == pytest.approx(0.5)
        assert result["t2_zonal_zone_def_to_def"] == pytest.approx(0.0)

    def test_zonal_all_mid(self) -> None:
        """All events in middle third produce 100% mid_to_mid."""
        events = (
            _evt("z01", timestamp=0.0, match_minute=0.0, location=(40.0, 50.0)),
            _evt("z02", timestamp=1.0, match_minute=0.02, location=(50.0, 50.0)),
            _evt("z03", timestamp=2.0, match_minute=0.03, location=(60.0, 50.0)),
        )
        ext = ZonalFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t2_zonal_zone_mid_to_mid"] == pytest.approx(1.0)
        assert result["t2_zonal_zone_def_to_mid"] == pytest.approx(0.0)

    def test_box_entries(self) -> None:
        """Event entering box zone is counted."""
        events = (
            # outside box
            _evt("z01", timestamp=0.0, match_minute=0.0, location=(70.0, 50.0)),
            # inside box (x>83.3, 21.1<y<78.9)
            _evt("z02", timestamp=1.0, match_minute=0.02, location=(90.0, 50.0)),
            # still inside box
            _evt("z03", timestamp=2.0, match_minute=0.03, location=(92.0, 40.0)),
            # outside box again
            _evt("z04", timestamp=3.0, match_minute=0.05, location=(70.0, 50.0)),
        )
        ext = ZonalFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t2_zonal_box_entries"] == pytest.approx(1.0)
        assert result["t2_zonal_box_exits"] == pytest.approx(1.0)

    def test_box_entries_y_boundary(self) -> None:
        """Events outside box y-range are not counted as entries."""
        events = (
            _evt("z01", timestamp=0.0, match_minute=0.0, location=(70.0, 50.0)),
            # x > 83.3 but y < 21.1 -> outside box
            _evt("z02", timestamp=1.0, match_minute=0.02, location=(90.0, 15.0)),
        )
        ext = ZonalFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t2_zonal_box_entries"] == pytest.approx(0.0)

    def test_zonal_fewer_than_two_located(self) -> None:
        """Fewer than 2 located team events -> transitions None."""
        events = (_evt("z01", timestamp=0.0, match_minute=0.0, location=(50.0, 50.0)),)
        ext = ZonalFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t2_zonal_zone_def_to_def"] is None
        assert result["t2_zonal_zone_mid_to_att"] is None
        assert result["t2_zonal_box_entries"] == pytest.approx(0.0)

    def test_zonal_filters_by_team(self) -> None:
        """Only focal-team events contribute to transitions."""
        events = (
            _evt(
                "z01",
                team_id="team_a",
                timestamp=0.0,
                match_minute=0.0,
                location=(20.0, 50.0),
            ),
            _evt(
                "z02",
                team_id="team_b",
                timestamp=0.5,
                match_minute=0.01,
                location=(50.0, 50.0),
            ),
            _evt(
                "z03",
                team_id="team_a",
                timestamp=1.0,
                match_minute=0.02,
                location=(80.0, 50.0),
            ),
        )
        ext = ZonalFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        # Only 2 team_a events: def->att (100%)
        assert result["t2_zonal_zone_def_to_att"] == pytest.approx(1.0)


# ------------------------------------------------------------------
# Team shape tests
# ------------------------------------------------------------------


class TestTeamShapeFeatureExtractor:
    """Tests for :class:`TeamShapeFeatureExtractor`."""

    def test_shape_centroid(self) -> None:
        """Team centroid is the mean of event x and y locations."""
        events = (
            _evt("s01", timestamp=0.0, match_minute=0.0, location=(20.0, 30.0)),
            _evt("s02", timestamp=1.0, match_minute=0.02, location=(40.0, 50.0)),
            _evt("s03", timestamp=2.0, match_minute=0.03, location=(60.0, 70.0)),
        )
        ext = TeamShapeFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t2_shape_team_centroid_x_est"] == pytest.approx(40.0)
        assert result["t2_shape_team_centroid_y_est"] == pytest.approx(50.0)

    def test_shape_compactness(self) -> None:
        """Compactness proxy is max_x - min_x."""
        events = (
            _evt("s01", timestamp=0.0, match_minute=0.0, location=(30.0, 50.0)),
            _evt("s02", timestamp=1.0, match_minute=0.02, location=(70.0, 50.0)),
        )
        ext = TeamShapeFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t2_shape_compactness_proxy"] == pytest.approx(40.0)

    def test_shape_engagement_line(self) -> None:
        """Opponent tackles at known x produce correct flipped median."""
        events = (
            # team_a pass (ignored for engagement line)
            _evt("s01", timestamp=0.0, match_minute=0.0, location=(50.0, 50.0)),
            # opponent tackles at x=60 and x=80 -> median=70 -> flipped=30
            _evt(
                "s02",
                team_id="team_b",
                timestamp=1.0,
                match_minute=0.02,
                location=(60.0, 50.0),
                event_type="tackle",
            ),
            _evt(
                "s03",
                team_id="team_b",
                timestamp=2.0,
                match_minute=0.03,
                location=(80.0, 50.0),
                event_type="tackle",
            ),
        )
        ext = TeamShapeFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        # median(60, 80) = 70 -> flipped = 100 - 70 = 30
        assert result["t2_shape_engagement_line"] == pytest.approx(30.0)

    def test_shape_engagement_line_mixed_types(self) -> None:
        """Engagement line uses all confrontational types."""
        events = (
            _evt("s01", timestamp=0.0, match_minute=0.0, location=(50.0, 50.0)),
            _evt(
                "s02",
                team_id="team_b",
                timestamp=1.0,
                match_minute=0.02,
                location=(40.0, 50.0),
                event_type="tackle",
            ),
            _evt(
                "s03",
                team_id="team_b",
                timestamp=2.0,
                match_minute=0.03,
                location=(60.0, 50.0),
                event_type="interception",
            ),
            _evt(
                "s04",
                team_id="team_b",
                timestamp=3.0,
                match_minute=0.05,
                location=(50.0, 50.0),
                event_type="block",
            ),
            _evt(
                "s05",
                team_id="team_b",
                timestamp=4.0,
                match_minute=0.07,
                location=(70.0, 50.0),
                event_type="foul_committed",
            ),
        )
        ext = TeamShapeFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        # median(40, 60, 50, 70) = 55 -> flipped = 100 - 55 = 45
        assert result["t2_shape_engagement_line"] == pytest.approx(45.0)

    def test_shape_engagement_line_insufficient(self) -> None:
        """Fewer than 2 confrontational events returns None."""
        events = (
            _evt("s01", timestamp=0.0, match_minute=0.0, location=(50.0, 50.0)),
            _evt(
                "s02",
                team_id="team_b",
                timestamp=1.0,
                match_minute=0.02,
                location=(60.0, 50.0),
                event_type="tackle",
            ),
        )
        ext = TeamShapeFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t2_shape_engagement_line"] is None

    def test_shape_no_team_events(self) -> None:
        """No located team events yield None for centroid/spread/compact."""
        events = (
            _evt(
                "s01",
                team_id="team_b",
                timestamp=0.0,
                match_minute=0.0,
                location=(50.0, 50.0),
            ),
        )
        ext = TeamShapeFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t2_shape_team_centroid_x_est"] is None
        assert result["t2_shape_team_centroid_y_est"] is None
        assert result["t2_shape_team_spread_est"] is None
        assert result["t2_shape_compactness_proxy"] is None

    def test_shape_spread(self) -> None:
        """Team spread is sqrt(var_x + var_y)."""
        import math

        import numpy as np

        events = (
            _evt("s01", timestamp=0.0, match_minute=0.0, location=(20.0, 30.0)),
            _evt("s02", timestamp=1.0, match_minute=0.02, location=(40.0, 50.0)),
            _evt("s03", timestamp=2.0, match_minute=0.03, location=(60.0, 70.0)),
        )
        ext = TeamShapeFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        xs = np.array([20.0, 40.0, 60.0])
        ys = np.array([30.0, 50.0, 70.0])
        expected = math.sqrt(float(np.var(xs)) + float(np.var(ys)))

        assert result["t2_shape_team_spread_est"] == pytest.approx(expected)


# ------------------------------------------------------------------
# Pressing tests
# ------------------------------------------------------------------


class TestPressingFeatureExtractor:
    """Tests for :class:`PressingFeatureExtractor`."""

    def test_pressing_intensity(self) -> None:
        """4 pressures / 8 opponent events = 0.5 intensity."""
        pressures = [
            _evt(
                f"pr{i}",
                timestamp=float(i),
                match_minute=float(i) / 60.0,
                location=(60.0, 50.0),
                event_type="pressure",
            )
            for i in range(4)
        ]
        opp_events = [
            _evt(
                f"op{i}",
                team_id="team_b",
                timestamp=float(i + 4),
                match_minute=float(i + 4) / 60.0,
                location=(40.0, 50.0),
            )
            for i in range(8)
        ]
        events = tuple(pressures + opp_events)
        ext = PressingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t2_press_intensity"] == pytest.approx(0.5)

    def test_pressing_intensity_no_opponent(self) -> None:
        """No opponent events yields None intensity."""
        events = (
            _evt(
                "pr01",
                timestamp=0.0,
                match_minute=0.0,
                location=(60.0, 50.0),
                event_type="pressure",
            ),
        )
        ext = PressingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t2_press_intensity"] is None

    def test_pressing_trigger_location(self) -> None:
        """Trigger x/y is the mean of pressure event locations."""
        events = (
            _evt(
                "pr01",
                timestamp=0.0,
                match_minute=0.0,
                location=(60.0, 40.0),
                event_type="pressure",
            ),
            _evt(
                "pr02",
                timestamp=1.0,
                match_minute=0.02,
                location=(70.0, 60.0),
                event_type="pressure",
            ),
            # opponent event so intensity is computable
            _evt(
                "op01",
                team_id="team_b",
                timestamp=2.0,
                match_minute=0.03,
                location=(40.0, 50.0),
            ),
        )
        ext = PressingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t2_press_trigger_x"] == pytest.approx(65.0)
        assert result["t2_press_trigger_y"] == pytest.approx(50.0)

    def test_pressing_trigger_no_pressures(self) -> None:
        """No pressure events yields None for trigger location."""
        events = (_evt("p01", timestamp=0.0, match_minute=0.0, location=(50.0, 50.0)),)
        ext = PressingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t2_press_trigger_x"] is None
        assert result["t2_press_trigger_y"] is None

    def test_pressing_ppda(self) -> None:
        """10 opponent passes / (3 tackles + 2 interceptions) = 2.0."""
        opp_passes = [
            _evt(
                f"op{i}",
                team_id="team_b",
                timestamp=float(i),
                match_minute=float(i) / 60.0,
                location=(40.0, 50.0),
            )
            for i in range(10)
        ]
        tackles = [
            _evt(
                f"tk{i}",
                timestamp=float(i + 10),
                match_minute=float(i + 10) / 60.0,
                location=(50.0, 50.0),
                event_type="tackle",
            )
            for i in range(3)
        ]
        interceptions = [
            _evt(
                f"ic{i}",
                timestamp=float(i + 13),
                match_minute=float(i + 13) / 60.0,
                location=(50.0, 50.0),
                event_type="interception",
            )
            for i in range(2)
        ]
        events = tuple(opp_passes + tackles + interceptions)
        ext = PressingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t2_press_ppda"] == pytest.approx(2.0)

    def test_pressing_ppda_zero_def_actions(self) -> None:
        """Zero defensive actions yields None PPDA."""
        events = (
            _evt(
                "op01",
                team_id="team_b",
                timestamp=0.0,
                match_minute=0.0,
                location=(40.0, 50.0),
            ),
        )
        ext = PressingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t2_press_ppda"] is None

    def test_pressing_ppda_includes_fouls(self) -> None:
        """Fouls committed count as defensive actions in PPDA."""
        opp_passes = [
            _evt(
                f"op{i}",
                team_id="team_b",
                timestamp=float(i),
                match_minute=float(i) / 60.0,
                location=(40.0, 50.0),
            )
            for i in range(6)
        ]
        fouls = [
            _evt(
                f"fl{i}",
                timestamp=float(i + 6),
                match_minute=float(i + 6) / 60.0,
                location=(50.0, 50.0),
                event_type="foul_committed",
            )
            for i in range(3)
        ]
        events = tuple(opp_passes + fouls)
        ext = PressingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        # 6 opp passes / 3 fouls = 2.0
        assert result["t2_press_ppda"] == pytest.approx(2.0)

    def test_pressing_success_rate(self) -> None:
        """Pressure followed by focal team non-pressure is successful."""
        events = (
            # opponent pass
            _evt(
                "op01",
                team_id="team_b",
                timestamp=0.0,
                match_minute=0.0,
                location=(40.0, 50.0),
            ),
            # team_a pressure (successful: next team_a event is a pass)
            _evt(
                "pr01",
                timestamp=1.0,
                match_minute=0.02,
                location=(60.0, 50.0),
                event_type="pressure",
            ),
            # team_a ball recovery -> turnover (non-pressure focal event)
            _evt(
                "br01",
                timestamp=2.0,
                match_minute=0.03,
                location=(55.0, 50.0),
                event_type="ball_recovery",
            ),
            # another opponent pass
            _evt(
                "op02",
                team_id="team_b",
                timestamp=3.0,
                match_minute=0.05,
                location=(40.0, 50.0),
            ),
            # team_a pressure (unsuccessful: no team_a event within lookahead)
            _evt(
                "pr02",
                timestamp=4.0,
                match_minute=0.07,
                location=(60.0, 50.0),
                event_type="pressure",
            ),
            _evt(
                "op03",
                team_id="team_b",
                timestamp=5.0,
                match_minute=0.08,
                location=(40.0, 50.0),
            ),
            _evt(
                "op04",
                team_id="team_b",
                timestamp=6.0,
                match_minute=0.10,
                location=(40.0, 50.0),
            ),
            _evt(
                "op05",
                team_id="team_b",
                timestamp=7.0,
                match_minute=0.12,
                location=(40.0, 50.0),
            ),
        )
        ext = PressingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        # 1 successful out of 2 pressures = 0.5
        assert result["t2_press_success_rate"] == pytest.approx(0.5)

    def test_pressing_success_rate_no_pressures(self) -> None:
        """No pressures yields None success rate."""
        events = (_evt("p01", timestamp=0.0, match_minute=0.0, location=(50.0, 50.0)),)
        ext = PressingFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t2_press_success_rate"] is None


# ------------------------------------------------------------------
# Transition tests
# ------------------------------------------------------------------


class TestTransitionFeatureExtractor:
    """Tests for :class:`TransitionFeatureExtractor`."""

    def test_transition_counter_attack(self) -> None:
        """Fast forward movement after turnover triggers indicator."""
        events = (
            # opponent event
            _evt(
                "op01",
                team_id="team_b",
                timestamp=0.0,
                match_minute=0.0,
                location=(50.0, 50.0),
            ),
            # turnover: team_a gains ball at x=30
            _evt(
                "ta01",
                timestamp=1.0,
                match_minute=0.02,
                location=(30.0, 50.0),
                event_type="ball_recovery",
            ),
            # team_a carries forward to x=70 (displacement = 40 > 25)
            _evt(
                "ta02",
                timestamp=2.0,
                match_minute=0.03,
                location=(70.0, 50.0),
                event_type="carry",
            ),
        )
        ext = TransitionFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t2_transition_counter_attack_indicator"] == pytest.approx(1.0)

    def test_transition_counter_attack_slow(self) -> None:
        """Small displacement after turnover does not trigger indicator."""
        events = (
            _evt(
                "op01",
                team_id="team_b",
                timestamp=0.0,
                match_minute=0.0,
                location=(50.0, 50.0),
            ),
            # turnover at x=40
            _evt(
                "ta01",
                timestamp=1.0,
                match_minute=0.02,
                location=(40.0, 50.0),
                event_type="ball_recovery",
            ),
            # only advances to x=50 (displacement = 10 < 25)
            _evt(
                "ta02",
                timestamp=2.0,
                match_minute=0.03,
                location=(50.0, 50.0),
                event_type="carry",
            ),
        )
        ext = TransitionFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t2_transition_counter_attack_indicator"] == pytest.approx(0.0)

    def test_transition_no_turnovers(self) -> None:
        """No turnovers returns None for all transition features."""
        events = (
            _evt("ta01", timestamp=0.0, match_minute=0.0, location=(30.0, 50.0)),
            _evt("ta02", timestamp=1.0, match_minute=0.02, location=(50.0, 50.0)),
            _evt("ta03", timestamp=2.0, match_minute=0.03, location=(70.0, 50.0)),
        )
        ext = TransitionFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t2_transition_counter_attack_indicator"] is None
        assert result["t2_transition_counter_press_indicator"] is None
        assert result["t2_transition_transition_speed"] is None

    def test_transition_counter_press(self) -> None:
        """Pressing after losing ball triggers counter-press indicator."""
        events = (
            # team_a has the ball
            _evt("ta01", timestamp=0.0, match_minute=0.0, location=(50.0, 50.0)),
            # turnover against: team_b gains ball
            _evt(
                "op01",
                team_id="team_b",
                timestamp=1.0,
                match_minute=0.02,
                location=(50.0, 50.0),
            ),
            # team_a counter-presses (2 pressures within lookahead)
            _evt(
                "pr01",
                timestamp=2.0,
                match_minute=0.03,
                location=(55.0, 50.0),
                event_type="pressure",
            ),
            _evt(
                "pr02",
                timestamp=3.0,
                match_minute=0.05,
                location=(52.0, 50.0),
                event_type="pressure",
            ),
        )
        ext = TransitionFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t2_transition_counter_press_indicator"] == pytest.approx(1.0)

    def test_transition_counter_press_insufficient(self) -> None:
        """Only 1 pressure after turnover does not trigger indicator."""
        events = (
            _evt("ta01", timestamp=0.0, match_minute=0.0, location=(50.0, 50.0)),
            # turnover against
            _evt(
                "op01",
                team_id="team_b",
                timestamp=1.0,
                match_minute=0.02,
                location=(50.0, 50.0),
            ),
            # only 1 pressure
            _evt(
                "pr01",
                timestamp=2.0,
                match_minute=0.03,
                location=(55.0, 50.0),
                event_type="pressure",
            ),
            _evt(
                "op02",
                team_id="team_b",
                timestamp=3.0,
                match_minute=0.05,
                location=(40.0, 50.0),
            ),
            _evt(
                "op03",
                team_id="team_b",
                timestamp=4.0,
                match_minute=0.07,
                location=(35.0, 50.0),
            ),
        )
        ext = TransitionFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t2_transition_counter_press_indicator"] == pytest.approx(0.0)

    def test_transition_speed(self) -> None:
        """Transition speed = x-displacement / time after turnover."""
        events = (
            _evt(
                "op01",
                team_id="team_b",
                timestamp=0.0,
                match_minute=0.0,
                location=(50.0, 50.0),
            ),
            # turnover at x=30, t=1.0
            _evt(
                "ta01",
                timestamp=1.0,
                match_minute=0.02,
                location=(30.0, 50.0),
                event_type="ball_recovery",
            ),
            # advance to x=50, t=3.0
            _evt(
                "ta02",
                timestamp=3.0,
                match_minute=0.05,
                location=(50.0, 50.0),
                event_type="carry",
            ),
        )
        ext = TransitionFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        # displacement = 50 - 30 = 20, time = 3.0 - 1.0 = 2.0
        assert result["t2_transition_transition_speed"] == pytest.approx(10.0)

    def test_transition_speed_no_valid_turnovers(self) -> None:
        """Turnovers with insufficient lookahead events yield None speed."""
        events = (
            _evt(
                "op01",
                team_id="team_b",
                timestamp=0.0,
                match_minute=0.0,
                location=(50.0, 50.0),
            ),
            # turnover but only 1 team event (need >=2 for speed)
            _evt(
                "ta01",
                timestamp=1.0,
                match_minute=0.02,
                location=(30.0, 50.0),
                event_type="ball_recovery",
            ),
        )
        ext = TransitionFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        assert result["t2_transition_transition_speed"] is None

    def test_transition_counter_press_no_turnover_against(self) -> None:
        """No turnovers against yields None counter-press indicator."""
        events = (
            # only opponent events then team gains ball
            _evt(
                "op01",
                team_id="team_b",
                timestamp=0.0,
                match_minute=0.0,
                location=(50.0, 50.0),
            ),
            _evt("ta01", timestamp=1.0, match_minute=0.02, location=(50.0, 50.0)),
            _evt("ta02", timestamp=2.0, match_minute=0.03, location=(60.0, 50.0)),
        )
        ext = TransitionFeatureExtractor()
        result = ext.extract(events, _CONTEXT)

        # No turnover against (team_a never loses ball in this sequence)
        assert result["t2_transition_counter_press_indicator"] is None


# ------------------------------------------------------------------
# Naming & tier tests (all 4 extractors)
# ------------------------------------------------------------------

_ALL_EXTRACTORS = [
    ZonalFeatureExtractor,
    TeamShapeFeatureExtractor,
    PressingFeatureExtractor,
    TransitionFeatureExtractor,
]

_EXPECTED_PREFIXES = {
    "ZonalFeatureExtractor": "t2_zonal_",
    "TeamShapeFeatureExtractor": "t2_shape_",
    "PressingFeatureExtractor": "t2_press_",
    "TransitionFeatureExtractor": "t2_transition_",
}


class TestNamingAndTier:
    """Verify naming conventions and tier declarations for all extractors."""

    @pytest.mark.parametrize(
        "extractor_cls",
        _ALL_EXTRACTORS,
        ids=[c.__name__ for c in _ALL_EXTRACTORS],
    )
    def test_tier_is_two(self, extractor_cls: type) -> None:
        """All Tier 2 extractors report tier=2."""
        assert extractor_cls().tier == 2

    @pytest.mark.parametrize(
        "extractor_cls",
        _ALL_EXTRACTORS,
        ids=[c.__name__ for c in _ALL_EXTRACTORS],
    )
    def test_feature_names_prefixed(self, extractor_cls: type) -> None:
        """All feature names use the correct t2_ prefix."""
        ext = extractor_cls()
        prefix = _EXPECTED_PREFIXES[extractor_cls.__name__]
        for name in ext.feature_names:
            assert name.startswith(prefix), (
                f"{extractor_cls.__name__}: {name!r} missing prefix {prefix!r}"
            )

    @pytest.mark.parametrize(
        "extractor_cls",
        _ALL_EXTRACTORS,
        ids=[c.__name__ for c in _ALL_EXTRACTORS],
    )
    def test_extract_keys_match_feature_names(self, extractor_cls: type) -> None:
        """Extract keys match declared feature_names for each extractor."""
        ext = extractor_cls()
        events = (
            _evt(
                "e01",
                timestamp=0.0,
                match_minute=0.0,
                location=(50.0, 50.0),
                end_location=(55.0, 50.0),
            ),
            _evt(
                "e02",
                team_id="team_b",
                timestamp=1.0,
                match_minute=0.02,
                location=(40.0, 50.0),
                event_type="tackle",
            ),
            _evt(
                "e03",
                timestamp=2.0,
                match_minute=0.03,
                location=(60.0, 50.0),
                event_type="pressure",
            ),
            _evt(
                "e04",
                team_id="team_b",
                timestamp=3.0,
                match_minute=0.05,
                location=(45.0, 50.0),
                event_type="tackle",
            ),
            _evt(
                "e05",
                timestamp=4.0,
                match_minute=0.07,
                location=(55.0, 50.0),
                event_type="interception",
            ),
        )
        result = ext.extract(events, _CONTEXT)
        assert set(result.keys()) == set(ext.feature_names)

    def test_no_duplicate_names_across_extractors(self) -> None:
        """No two Tier 2 extractors share a feature name."""
        instances: list[FeatureExtractor] = [
            ZonalFeatureExtractor(),
            TeamShapeFeatureExtractor(),
            PressingFeatureExtractor(),
            TransitionFeatureExtractor(),
        ]
        all_names: list[str] = []
        for ext in instances:
            all_names.extend(ext.feature_names)
        assert len(all_names) == len(set(all_names))
