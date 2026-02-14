"""Shared test fixtures for the Tactical State Discovery Engine.

Provides reusable fixtures used across multiple test modules:

* :func:`sample_event` -- a single :class:`NormalizedEvent` (pass).
* :func:`sample_events_window` -- 10 events spanning a 15-second window.
* :func:`sample_events_with_360` -- 8 events with freeze-frame data on
  selected events for Tier 3 feature testing.
* :func:`sample_feature_df` -- synthetic :class:`polars.DataFrame` with
  100 rows and all tier 1/2/3 feature columns.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from tactical.adapters.schemas import FreezeFramePlayer, NormalizedEvent

# ------------------------------------------------------------------
# Tier 1 feature names (55)
# ------------------------------------------------------------------

_T1_SPATIAL: tuple[str, ...] = (
    "t1_event_centroid_x",
    "t1_event_centroid_y",
    "t1_event_spread_x",
    "t1_event_spread_y",
    "t1_avg_distance_to_goal",
    "t1_attacking_third_pct",
    "t1_middle_third_pct",
    "t1_defensive_third_pct",
    "t1_left_channel_pct",
    "t1_center_channel_pct",
    "t1_right_channel_pct",
)

_T1_TEMPORAL: tuple[str, ...] = (
    "t1_event_rate",
    "t1_mean_inter_event_time",
    "t1_std_inter_event_time",
    "t1_max_inter_event_time",
    "t1_match_minute",
    "t1_period",
)

_T1_PASSING: tuple[str, ...] = (
    "t1_pass_count",
    "t1_pass_completion_rate",
    "t1_pass_length_mean",
    "t1_pass_length_std",
    "t1_pass_angle_std",
    "t1_progressive_pass_count",
    "t1_backward_pass_ratio",
    "t1_pass_height_ground_pct",
    "t1_pass_height_low_pct",
    "t1_pass_height_high_pct",
    "t1_switch_count",
    "t1_pass_directness",
)

_T1_CARRYING: tuple[str, ...] = (
    "t1_carry_count",
    "t1_carry_distance_total",
    "t1_carry_distance_mean",
    "t1_carry_directness",
    "t1_progressive_carry_count",
    "t1_dribble_attempt_count",
    "t1_dribble_success_rate",
)

_T1_DEFENDING: tuple[str, ...] = (
    "t1_pressure_count",
    "t1_pressure_rate",
    "t1_tackle_count",
    "t1_tackle_success_rate",
    "t1_interception_count",
    "t1_clearance_count",
    "t1_foul_count",
    "t1_under_pressure_pct",
    "t1_defensive_action_height",
)

_T1_SHOOTING: tuple[str, ...] = (
    "t1_shot_count",
    "t1_shot_distance_mean",
    "t1_xg_sum",
)

_T1_CONTEXT: tuple[str, ...] = (
    "t1_score_differential",
    "t1_possession_share",
    "t1_possession_team",
)

_T1_SEQUENCE: tuple[str, ...] = (
    "t1_chain_length",
    "t1_chain_x_displacement",
    "t1_chain_outcome",
    "t1_buildup_speed",
)

T1_FEATURES: tuple[str, ...] = (
    *_T1_SPATIAL,
    *_T1_TEMPORAL,
    *_T1_PASSING,
    *_T1_CARRYING,
    *_T1_DEFENDING,
    *_T1_SHOOTING,
    *_T1_CONTEXT,
    *_T1_SEQUENCE,
)

# ------------------------------------------------------------------
# Tier 2 feature names (24)
# ------------------------------------------------------------------

_T2_ZONAL: tuple[str, ...] = (
    *(f"t2_zone_transition_{r}{c}" for r in range(3) for c in range(3)),
    "t2_box_entries",
    "t2_box_exits",
)

_T2_SHAPE: tuple[str, ...] = (
    "t2_team_centroid_x_est",
    "t2_team_centroid_y_est",
    "t2_team_spread_est",
    "t2_engagement_line",
    "t2_compactness_proxy",
)

_T2_PRESSING: tuple[str, ...] = (
    "t2_pressing_intensity",
    "t2_press_trigger_x",
    "t2_press_trigger_y",
    "t2_press_success_rate",
    "t2_ppda",
)

_T2_TRANSITIONS: tuple[str, ...] = (
    "t2_counter_attack_indicator",
    "t2_counter_press_indicator",
    "t2_transition_speed",
)

T2_FEATURES: tuple[str, ...] = (
    *_T2_ZONAL,
    *_T2_SHAPE,
    *_T2_PRESSING,
    *_T2_TRANSITIONS,
)

# ------------------------------------------------------------------
# Tier 3 feature names (19)
# ------------------------------------------------------------------

_T3_FORMATION: tuple[str, ...] = (
    "t3_team_centroid_x",
    "t3_team_centroid_y",
    "t3_team_spread_x",
    "t3_team_spread_y",
    "t3_convex_hull_area",
    "t3_convex_hull_perimeter",
    "t3_defensive_line",
    "t3_midfield_line",
    "t3_attacking_line",
    "t3_def_mid_gap",
    "t3_mid_att_gap",
    "t3_team_width",
    "t3_team_length",
    "t3_formation_eccentricity",
)

_T3_RELATIONAL: tuple[str, ...] = (
    "t3_avg_nearest_opponent_dist",
    "t3_min_nearest_opponent_dist",
)

_T3_OPPONENT: tuple[str, ...] = (
    "t3_opp_centroid_x",
    "t3_opp_spread_x",
    "t3_opp_defensive_line",
)

T3_FEATURES: tuple[str, ...] = (
    *_T3_FORMATION,
    *_T3_RELATIONAL,
    *_T3_OPPONENT,
)

ALL_FEATURES: tuple[str, ...] = (*T1_FEATURES, *T2_FEATURES, *T3_FEATURES)

# ------------------------------------------------------------------
# Value ranges for synthetic feature generation
# ------------------------------------------------------------------

# Mapping from feature name to (low, high) bounds for uniform sampling.
# Features not listed default to (0.0, 1.0).
_FEATURE_RANGES: dict[str, tuple[float, float]] = {
    # -- Tier 1: Spatial ------------------------------------------------
    "t1_event_centroid_x": (10.0, 90.0),
    "t1_event_centroid_y": (15.0, 85.0),
    "t1_event_spread_x": (0.0, 40.0),
    "t1_event_spread_y": (0.0, 35.0),
    "t1_avg_distance_to_goal": (10.0, 80.0),
    # pct features default to 0-1
    # -- Tier 1: Temporal -----------------------------------------------
    "t1_event_rate": (0.2, 1.8),
    "t1_mean_inter_event_time": (0.5, 8.0),
    "t1_std_inter_event_time": (0.1, 5.0),
    "t1_max_inter_event_time": (1.0, 15.0),
    "t1_match_minute": (1.0, 90.0),
    "t1_period": (1.0, 2.0),
    # -- Tier 1: Passing ------------------------------------------------
    "t1_pass_count": (0.0, 12.0),
    "t1_pass_length_mean": (3.0, 45.0),
    "t1_pass_length_std": (0.0, 20.0),
    "t1_pass_angle_std": (0.0, 3.15),
    "t1_progressive_pass_count": (0.0, 6.0),
    "t1_switch_count": (0.0, 3.0),
    # -- Tier 1: Carrying -----------------------------------------------
    "t1_carry_count": (0.0, 8.0),
    "t1_carry_distance_total": (0.0, 80.0),
    "t1_carry_distance_mean": (0.0, 25.0),
    "t1_progressive_carry_count": (0.0, 4.0),
    "t1_dribble_attempt_count": (0.0, 3.0),
    # -- Tier 1: Defending ----------------------------------------------
    "t1_pressure_count": (0.0, 10.0),
    "t1_pressure_rate": (0.0, 1.5),
    "t1_tackle_count": (0.0, 4.0),
    "t1_interception_count": (0.0, 3.0),
    "t1_clearance_count": (0.0, 4.0),
    "t1_foul_count": (0.0, 2.0),
    "t1_defensive_action_height": (10.0, 90.0),
    # -- Tier 1: Shooting -----------------------------------------------
    "t1_shot_count": (0.0, 3.0),
    "t1_shot_distance_mean": (8.0, 35.0),
    "t1_xg_sum": (0.0, 0.45),
    # -- Tier 1: Context ------------------------------------------------
    "t1_score_differential": (-3.0, 3.0),
    # -- Tier 1: Sequence -----------------------------------------------
    "t1_chain_length": (1.0, 15.0),
    "t1_chain_x_displacement": (-30.0, 60.0),
    "t1_buildup_speed": (0.0, 8.0),
    # -- Tier 2: Zonal --------------------------------------------------
    "t2_box_entries": (0.0, 5.0),
    "t2_box_exits": (0.0, 4.0),
    # -- Tier 2: Team shape ---------------------------------------------
    "t2_team_centroid_x_est": (15.0, 85.0),
    "t2_team_centroid_y_est": (25.0, 75.0),
    "t2_team_spread_est": (5.0, 40.0),
    "t2_engagement_line": (20.0, 80.0),
    "t2_compactness_proxy": (10.0, 50.0),
    # -- Tier 2: Pressing -----------------------------------------------
    "t2_pressing_intensity": (0.0, 1.5),
    "t2_press_trigger_x": (20.0, 90.0),
    "t2_press_trigger_y": (15.0, 85.0),
    "t2_ppda": (3.0, 30.0),
    # -- Tier 2: Transitions --------------------------------------------
    "t2_transition_speed": (0.0, 8.0),
    # -- Tier 3: Formation ----------------------------------------------
    "t3_team_centroid_x": (15.0, 85.0),
    "t3_team_centroid_y": (25.0, 75.0),
    "t3_team_spread_x": (5.0, 35.0),
    "t3_team_spread_y": (5.0, 30.0),
    "t3_convex_hull_area": (200.0, 4500.0),
    "t3_convex_hull_perimeter": (50.0, 300.0),
    "t3_defensive_line": (10.0, 50.0),
    "t3_midfield_line": (30.0, 65.0),
    "t3_attacking_line": (50.0, 90.0),
    "t3_def_mid_gap": (5.0, 30.0),
    "t3_mid_att_gap": (5.0, 30.0),
    "t3_team_width": (15.0, 70.0),
    "t3_team_length": (20.0, 75.0),
    # -- Tier 3: Relational ---------------------------------------------
    "t3_avg_nearest_opponent_dist": (3.0, 25.0),
    "t3_min_nearest_opponent_dist": (1.0, 15.0),
    # -- Tier 3: Opponent -----------------------------------------------
    "t3_opp_centroid_x": (15.0, 85.0),
    "t3_opp_spread_x": (5.0, 35.0),
    "t3_opp_defensive_line": (10.0, 50.0),
}

# Features that should be rounded to integers.
_INTEGER_FEATURES: frozenset[str] = frozenset(
    {
        "t1_period",
        "t1_pass_count",
        "t1_progressive_pass_count",
        "t1_switch_count",
        "t1_carry_count",
        "t1_progressive_carry_count",
        "t1_dribble_attempt_count",
        "t1_pressure_count",
        "t1_tackle_count",
        "t1_interception_count",
        "t1_clearance_count",
        "t1_foul_count",
        "t1_shot_count",
        "t1_chain_length",
        "t2_box_entries",
        "t2_box_exits",
    }
)

# Features that should be 0 or 1 (binary).
_BINARY_FEATURES: frozenset[str] = frozenset(
    {
        "t1_possession_team",
        "t1_chain_outcome",
        "t2_counter_attack_indicator",
        "t2_counter_press_indicator",
    }
)

N_ROWS: int = 100
N_T1: int = len(T1_FEATURES)
N_T2: int = len(T2_FEATURES)
N_T3: int = len(T3_FEATURES)
# Fraction of rows that have Tier 3 data (simulates Leverkusen-only 360).
_T3_AVAILABILITY: float = 0.2


# ------------------------------------------------------------------
# Helper: build a NormalizedEvent concisely
# ------------------------------------------------------------------


def _evt(
    event_id: str,
    *,
    team_id: str = "team_a",
    player_id: str = "player_01",
    period: int = 1,
    timestamp: float,
    match_minute: float,
    location: tuple[float, float],
    end_location: tuple[float, float] | None = None,
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
        freeze_frame=freeze_frame,
        score_home=score_home,
        score_away=score_away,
        team_is_home=team_is_home,
    )


# ------------------------------------------------------------------
# Reusable freeze-frame tuples
# ------------------------------------------------------------------

_FF_TEAMMATES: tuple[FreezeFramePlayer, ...] = (
    FreezeFramePlayer(
        player_id="p_t1",
        teammate=True,
        location=(55.0, 40.0),
        position="Center Back",
    ),
    FreezeFramePlayer(
        player_id="p_t2",
        teammate=True,
        location=(60.0, 55.0),
        position="Left Back",
    ),
    FreezeFramePlayer(
        player_id="p_t3",
        teammate=True,
        location=(65.0, 30.0),
        position="Right Back",
    ),
    FreezeFramePlayer(
        player_id="p_t4",
        teammate=True,
        location=(70.0, 48.0),
        position="Central Midfield",
    ),
    FreezeFramePlayer(
        player_id="p_t5",
        teammate=True,
        location=(78.0, 35.0),
        position="Right Wing",
    ),
)

_FF_OPPONENTS: tuple[FreezeFramePlayer, ...] = (
    FreezeFramePlayer(
        player_id="p_o1",
        teammate=False,
        location=(50.0, 50.0),
        position="Goalkeeper",
    ),
    FreezeFramePlayer(
        player_id="p_o2",
        teammate=False,
        location=(62.0, 42.0),
        position="Center Back",
    ),
    FreezeFramePlayer(
        player_id="p_o3",
        teammate=False,
        location=(68.0, 60.0),
        position="Left Back",
    ),
    FreezeFramePlayer(
        player_id="p_o4",
        teammate=False,
        location=(72.0, 38.0),
        position="Central Midfield",
    ),
    FreezeFramePlayer(
        player_id="p_o5",
        teammate=False,
        location=(80.0, 52.0),
        position="Center Forward",
    ),
)

_FREEZE_FRAME_FULL: tuple[FreezeFramePlayer, ...] = (*_FF_TEAMMATES, *_FF_OPPONENTS)

_FREEZE_FRAME_SHOT: tuple[FreezeFramePlayer, ...] = (
    FreezeFramePlayer(
        player_id="p_gk",
        teammate=False,
        location=(95.0, 50.0),
        position="Goalkeeper",
    ),
    FreezeFramePlayer(
        player_id="p_cb1",
        teammate=False,
        location=(88.0, 42.0),
        position="Center Back",
    ),
    FreezeFramePlayer(
        player_id="p_cb2",
        teammate=False,
        location=(87.0, 58.0),
        position="Center Back",
    ),
    FreezeFramePlayer(
        player_id="p_att",
        teammate=True,
        location=(85.0, 48.0),
        position="Center Forward",
    ),
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture()
def sample_event() -> NormalizedEvent:
    """A single normalized pass event for unit testing."""
    return _evt(
        "evt_001",
        timestamp=123.4,
        match_minute=2.1,
        location=(45.0, 30.0),
        end_location=(55.0, 35.0),
    )


@pytest.fixture()
def sample_events_window() -> tuple[NormalizedEvent, ...]:
    """Ten events spanning a realistic 15-second window.

    All events are in period 1, from timestamp 300.0 to 314.5
    (match minute ~5.0-5.24).  The sequence models a short passage
    of play: build-up by team_a, pressure from team_b, turnover,
    clearance, and counter-press.

    Events are sorted by ``(period, timestamp)``.
    """
    return (
        # 1. team_a pass (complete) -- build-up from deep
        _evt(
            "w_001",
            timestamp=300.0,
            match_minute=5.0,
            location=(25.0, 45.0),
            end_location=(40.0, 50.0),
            player_id="player_01",
        ),
        # 2. team_a ball_receipt
        _evt(
            "w_002",
            timestamp=301.2,
            match_minute=5.02,
            location=(40.0, 50.0),
            event_type="ball_receipt",
            player_id="player_02",
        ),
        # 3. team_a carry -- advances
        _evt(
            "w_003",
            timestamp=303.0,
            match_minute=5.05,
            location=(40.0, 50.0),
            end_location=(52.0, 48.0),
            event_type="carry",
            player_id="player_02",
        ),
        # 4. team_a pass (complete) -- plays forward
        _evt(
            "w_004",
            timestamp=305.5,
            match_minute=5.09,
            location=(52.0, 48.0),
            end_location=(65.0, 40.0),
            player_id="player_02",
        ),
        # 5. team_a ball_receipt
        _evt(
            "w_005",
            timestamp=306.8,
            match_minute=5.11,
            location=(65.0, 40.0),
            event_type="ball_receipt",
            player_id="player_03",
        ),
        # 6. team_b pressure -- pressing the receiver
        _evt(
            "w_006",
            team_id="team_b",
            player_id="player_11",
            timestamp=307.5,
            match_minute=5.13,
            location=(66.0, 41.0),
            event_type="pressure",
            team_is_home=False,
        ),
        # 7. team_a pass (incomplete) -- forced error
        _evt(
            "w_007",
            timestamp=308.3,
            match_minute=5.14,
            location=(65.0, 40.0),
            end_location=(70.0, 55.0),
            event_outcome="incomplete",
            under_pressure=True,
            player_id="player_03",
        ),
        # 8. team_b ball_recovery -- turnover
        _evt(
            "w_008",
            team_id="team_b",
            player_id="player_12",
            timestamp=309.8,
            match_minute=5.16,
            location=(70.0, 55.0),
            event_type="ball_recovery",
            team_is_home=False,
        ),
        # 9. team_b clearance
        _evt(
            "w_009",
            team_id="team_b",
            player_id="player_12",
            timestamp=311.0,
            match_minute=5.18,
            location=(72.0, 52.0),
            end_location=(45.0, 60.0),
            event_type="clearance",
            body_part="head",
            team_is_home=False,
        ),
        # 10. team_a pressure -- counter-press
        _evt(
            "w_010",
            timestamp=314.5,
            match_minute=5.24,
            location=(46.0, 58.0),
            event_type="pressure",
            player_id="player_04",
        ),
    )


@pytest.fixture()
def sample_events_with_360() -> tuple[NormalizedEvent, ...]:
    """Eight events with freeze-frame data on selected events.

    Five of eight events carry a non-None ``freeze_frame`` containing
    both teammate and opponent :class:`FreezeFramePlayer` snapshots.
    Includes a shot with its own dedicated freeze frame.

    Suitable for testing Tier 3 feature extraction where 360 spatial
    information is required.
    """
    return (
        # 1. pass with full freeze frame
        _evt(
            "ff_001",
            timestamp=600.0,
            match_minute=10.0,
            location=(35.0, 45.0),
            end_location=(50.0, 42.0),
            freeze_frame=_FREEZE_FRAME_FULL,
        ),
        # 2. ball_receipt -- no freeze frame
        _evt(
            "ff_002",
            timestamp=601.5,
            match_minute=10.03,
            location=(50.0, 42.0),
            event_type="ball_receipt",
            player_id="player_02",
        ),
        # 3. carry with freeze frame
        _evt(
            "ff_003",
            timestamp=603.0,
            match_minute=10.05,
            location=(50.0, 42.0),
            end_location=(62.0, 38.0),
            event_type="carry",
            player_id="player_02",
            freeze_frame=_FREEZE_FRAME_FULL,
        ),
        # 4. pass with freeze frame
        _evt(
            "ff_004",
            timestamp=605.0,
            match_minute=10.08,
            location=(62.0, 38.0),
            end_location=(78.0, 44.0),
            player_id="player_02",
            freeze_frame=_FREEZE_FRAME_FULL,
        ),
        # 5. ball_receipt -- no freeze frame
        _evt(
            "ff_005",
            timestamp=606.2,
            match_minute=10.10,
            location=(78.0, 44.0),
            event_type="ball_receipt",
            player_id="player_05",
        ),
        # 6. pressure from team_b with freeze frame
        _evt(
            "ff_006",
            team_id="team_b",
            player_id="player_11",
            timestamp=607.0,
            match_minute=10.12,
            location=(79.0, 45.0),
            event_type="pressure",
            team_is_home=False,
            freeze_frame=_FREEZE_FRAME_FULL,
        ),
        # 7. shot with shot-specific freeze frame
        _evt(
            "ff_007",
            timestamp=608.5,
            match_minute=10.14,
            location=(82.0, 46.0),
            end_location=(95.5, 50.5),
            event_type="shot",
            event_outcome="saved",
            body_part="right_foot",
            under_pressure=True,
            player_id="player_05",
            freeze_frame=_FREEZE_FRAME_SHOT,
        ),
        # 8. clearance -- no freeze frame
        _evt(
            "ff_008",
            team_id="team_b",
            player_id="player_13",
            timestamp=610.0,
            match_minute=10.17,
            location=(90.0, 50.0),
            end_location=(60.0, 55.0),
            event_type="clearance",
            body_part="head",
            team_is_home=False,
        ),
    )


@pytest.fixture()
def sample_feature_df() -> pl.DataFrame:
    """Synthetic feature DataFrame for model-layer testing.

    Contains 100 rows (segments) with:

    * 7 metadata columns: ``match_id``, ``team_id``, ``segment_type``,
      ``start_time``, ``end_time``, ``period``, ``match_minute``.
    * 55 Tier 1 feature columns (always populated).
    * 24 Tier 2 feature columns (always populated).
    * 19 Tier 3 feature columns (populated for ~20 % of rows; ``null``
      for the remainder, simulating matches without 360 data).

    Feature values are drawn from plausible ranges with a fixed random
    seed for reproducibility.
    """
    rng = np.random.default_rng(seed=42)
    n = N_ROWS

    # -- Metadata columns -----------------------------------------------
    match_ids = [f"match_{i // 10 + 1:03d}" for i in range(n)]
    team_ids = [f"team_{'a' if i % 2 == 0 else 'b'}" for i in range(n)]
    segment_types = ["window"] * n
    start_times = np.linspace(0.0, 85.0 * 60, n)
    end_times = start_times + 15.0
    periods = np.where(start_times < 45.0 * 60, 1, 2).tolist()
    match_minutes = (start_times / 60.0).tolist()

    data: dict[str, list[float | str | int | None]] = {
        "match_id": match_ids,  # type: ignore[dict-item]
        "team_id": team_ids,  # type: ignore[dict-item]
        "segment_type": segment_types,  # type: ignore[dict-item]
        "start_time": start_times.tolist(),
        "end_time": end_times.tolist(),
        "period": periods,
        "match_minute": match_minutes,
    }

    # -- Boolean mask: which rows have Tier 3 data ----------------------
    has_360 = rng.random(n) < _T3_AVAILABILITY

    # -- Generate feature columns ---------------------------------------
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
        data[feat] = col  # type: ignore[assignment]

    return pl.DataFrame(data)
