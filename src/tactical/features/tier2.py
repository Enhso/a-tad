"""Tier 2 event-derived feature extractors.

Provides four extractor classes satisfying the
:class:`~tactical.features.base.FeatureExtractor` protocol:

* :class:`ZonalFeatureExtractor` -- zone transition rates, box entries/exits
* :class:`TeamShapeFeatureExtractor` -- estimated centroid, spread,
  engagement line, compactness
* :class:`PressingFeatureExtractor` -- pressing intensity, trigger
  location, success rate, PPDA
* :class:`TransitionFeatureExtractor` -- counter-attack, counter-press,
  and transition speed indicators

All are Tier 2 (event-derived): computed from the normalised event stream
with moderate additional logic but no 360 freeze-frame data.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tactical.adapters.schemas import MatchContext, NormalizedEvent

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_THIRD_LOW: float = 33.3
_THIRD_HIGH: float = 66.7

_BOX_X_THRESHOLD: float = 83.3
_BOX_Y_LOW: float = 21.1
_BOX_Y_HIGH: float = 78.9

_ZONE_LABELS: tuple[str, ...] = ("def", "mid", "att")

_CONFRONTATIONAL_TYPES: frozenset[str] = frozenset(
    {"tackle", "foul_committed", "interception", "block"}
)

_COUNTER_ATTACK_THRESHOLD: float = 25.0
_COUNTER_PRESS_MIN: int = 2
_TURNOVER_LOOKAHEAD: int = 3

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _x_zone(x: float) -> int:
    """Return zone index from an x-coordinate.

    Args:
        x: Normalised x-coordinate (0-100).

    Returns:
        ``0`` (defensive), ``1`` (middle), or ``2`` (attacking).
    """
    if x < _THIRD_LOW:
        return 0
    if x > _THIRD_HIGH:
        return 2
    return 1


def _in_box(x: float, y: float) -> bool:
    """Return whether ``(x, y)`` is inside the penalty box approximation.

    Args:
        x: Normalised x-coordinate.
        y: Normalised y-coordinate.

    Returns:
        ``True`` when the location falls within the 18-yard box
        approximation.
    """
    return x > _BOX_X_THRESHOLD and _BOX_Y_LOW < y < _BOX_Y_HIGH


# ------------------------------------------------------------------
# Zonal extractor
# ------------------------------------------------------------------

_ZONAL_TRANSITION_NAMES: tuple[str, ...] = tuple(
    f"t2_zonal_zone_{src}_to_{dst}" for src in _ZONE_LABELS for dst in _ZONE_LABELS
)

_ZONAL_NAMES: tuple[str, ...] = (
    *_ZONAL_TRANSITION_NAMES,
    "t2_zonal_box_entries",
    "t2_zonal_box_exits",
)


class ZonalFeatureExtractor:
    """Extract zonal transition and box-entry features.

    Divides the pitch into a 3x3 grid by x-coordinate thirds and
    computes the fraction of consecutive-event transitions between
    each source-destination pair.  Also counts events entering or
    leaving the penalty-box approximation.
    """

    __slots__ = ()

    @property
    def tier(self) -> int:
        """Feature tier: 2 (event-derived)."""
        return 2

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names of zonal features produced by this extractor."""
        return _ZONAL_NAMES

    def extract(
        self,
        events: tuple[NormalizedEvent, ...],
        context: MatchContext,
    ) -> dict[str, float | None]:
        """Extract zonal features from consecutive team events.

        Args:
            events: Tuple of normalised events in the segment.
            context: Per-team match context.

        Returns:
            Dict mapping each zonal feature name to its value.
            Transition rates are ``None`` when fewer than two located
            team events exist.
        """
        located = [
            e for e in events if e.team_id == context.team_id and e.location is not None
        ]

        if len(located) < 2:
            result: dict[str, float | None] = {
                name: None for name in _ZONAL_TRANSITION_NAMES
            }
            result["t2_zonal_box_entries"] = 0.0
            result["t2_zonal_box_exits"] = 0.0
            return result

        # Transition counts (3x3 matrix, x-axis thirds only).
        counts = [[0] * 3 for _ in range(3)]
        box_entries = 0
        box_exits = 0

        for prev_evt, curr_evt in zip(located, located[1:], strict=False):
            assert prev_evt.location is not None  # guaranteed by filter
            assert curr_evt.location is not None
            px, py = prev_evt.location
            cx, cy = curr_evt.location

            counts[_x_zone(px)][_x_zone(cx)] += 1

            prev_in = _in_box(px, py)
            curr_in = _in_box(cx, cy)
            if not prev_in and curr_in:
                box_entries += 1
            elif prev_in and not curr_in:
                box_exits += 1

        total = len(located) - 1
        result = {}
        for si, src in enumerate(_ZONE_LABELS):
            for di, dst in enumerate(_ZONE_LABELS):
                name = f"t2_zonal_zone_{src}_to_{dst}"
                result[name] = counts[si][di] / total

        result["t2_zonal_box_entries"] = float(box_entries)
        result["t2_zonal_box_exits"] = float(box_exits)
        return result


# ------------------------------------------------------------------
# Team shape extractor
# ------------------------------------------------------------------

_SHAPE_NAMES: tuple[str, ...] = (
    "t2_shape_team_centroid_x_est",
    "t2_shape_team_centroid_y_est",
    "t2_shape_team_spread_est",
    "t2_shape_engagement_line",
    "t2_shape_compactness_proxy",
)


class TeamShapeFeatureExtractor:
    """Extract estimated team-shape features from event locations.

    Computes proxy centroid, spatial spread, engagement line (from
    opponent confrontational events), and compactness for the focal
    team.
    """

    __slots__ = ()

    @property
    def tier(self) -> int:
        """Feature tier: 2 (event-derived)."""
        return 2

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names of shape features produced by this extractor."""
        return _SHAPE_NAMES

    def extract(
        self,
        events: tuple[NormalizedEvent, ...],
        context: MatchContext,
    ) -> dict[str, float | None]:
        """Extract team-shape features from event locations.

        Args:
            events: Tuple of normalised events in the segment.
            context: Per-team match context.

        Returns:
            Dict mapping each shape feature name to its value,
            or ``None`` when the feature cannot be computed.
        """
        locs = _located_team_coords(events, context.team_id)

        if locs.shape[0] == 0:
            centroid_x: float | None = None
            centroid_y: float | None = None
            spread: float | None = None
            compactness: float | None = None
        else:
            xs: NDArray[np.floating] = locs[:, 0]
            ys: NDArray[np.floating] = locs[:, 1]
            centroid_x = float(np.mean(xs))
            centroid_y = float(np.mean(ys))
            spread = math.sqrt(float(np.var(xs)) + float(np.var(ys)))
            compactness = float(np.max(xs) - np.min(xs))

        engagement_line = _engagement_line(events, context)

        return {
            "t2_shape_team_centroid_x_est": centroid_x,
            "t2_shape_team_centroid_y_est": centroid_y,
            "t2_shape_team_spread_est": spread,
            "t2_shape_engagement_line": engagement_line,
            "t2_shape_compactness_proxy": compactness,
        }


def _located_team_coords(
    events: tuple[NormalizedEvent, ...],
    team_id: str,
) -> NDArray[np.floating]:
    """Return ``(N, 2)`` array of located team-event coordinates.

    Args:
        events: Full segment events.
        team_id: Focal team identifier.

    Returns:
        Array of shape ``(N, 2)`` or ``(0, 2)`` when empty.
    """
    locs = [
        e.location for e in events if e.team_id == team_id and e.location is not None
    ]
    if not locs:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(locs, dtype=np.float64)


def _engagement_line(
    events: tuple[NormalizedEvent, ...],
    context: MatchContext,
) -> float | None:
    """Compute the opponent's engagement line (flipped median x).

    Uses confrontational events (tackles, fouls, interceptions,
    blocks) by the **opposing** team.  Returns ``None`` when fewer
    than two such events have locations.

    Args:
        events: Full segment events.
        context: Per-team match context.

    Returns:
        Flipped median x-coordinate, or ``None``.
    """
    opp_xs = [
        e.location[0]
        for e in events
        if e.team_id != context.team_id
        and e.event_type in _CONFRONTATIONAL_TYPES
        and e.location is not None
    ]
    if len(opp_xs) < 2:
        return None
    arr = np.array(opp_xs, dtype=np.float64)
    return float(100.0 - np.median(arr))


# ------------------------------------------------------------------
# Pressing extractor
# ------------------------------------------------------------------

_PRESS_NAMES: tuple[str, ...] = (
    "t2_press_intensity",
    "t2_press_trigger_x",
    "t2_press_trigger_y",
    "t2_press_success_rate",
    "t2_press_ppda",
)


class PressingFeatureExtractor:
    """Extract pressing features from pressure and defensive events.

    Computes pressing intensity, trigger location, press success
    rate, and passes allowed per defensive action (PPDA).
    """

    __slots__ = ()

    @property
    def tier(self) -> int:
        """Feature tier: 2 (event-derived)."""
        return 2

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names of pressing features produced by this extractor."""
        return _PRESS_NAMES

    def extract(
        self,
        events: tuple[NormalizedEvent, ...],
        context: MatchContext,
    ) -> dict[str, float | None]:
        """Extract pressing features from the event window.

        Args:
            events: Tuple of normalised events in the segment.
            context: Per-team match context.

        Returns:
            Dict mapping each pressing feature name to its value,
            or ``None`` when the feature cannot be computed.
        """
        team_id = context.team_id

        team_pressures = [
            e for e in events if e.team_id == team_id and e.event_type == "pressure"
        ]
        opp_events = [e for e in events if e.team_id != team_id]
        n_opp = len(opp_events)
        n_press = len(team_pressures)

        # Intensity: team pressures / opponent events.
        intensity: float | None = n_press / n_opp if n_opp > 0 else None

        # Trigger location: mean x/y of pressure events.
        trigger_x: float | None = None
        trigger_y: float | None = None
        if n_press > 0:
            press_locs = [p.location for p in team_pressures if p.location is not None]
            if press_locs:
                arr = np.array(press_locs, dtype=np.float64)
                trigger_x = float(np.mean(arr[:, 0]))
                trigger_y = float(np.mean(arr[:, 1]))

        # Success rate: fraction of pressures followed by a turnover
        # (focal team non-pressure event) within the next 3 events.
        success_rate = _press_success_rate(events, team_id, team_pressures)

        # PPDA: opponent passes / (team tackles + interceptions + fouls).
        ppda = _ppda(events, team_id)

        return {
            "t2_press_intensity": intensity,
            "t2_press_trigger_x": trigger_x,
            "t2_press_trigger_y": trigger_y,
            "t2_press_success_rate": success_rate,
            "t2_press_ppda": ppda,
        }


def _press_success_rate(
    events: tuple[NormalizedEvent, ...],
    team_id: str,
    team_pressures: list[NormalizedEvent],
) -> float | None:
    """Compute the fraction of pressures leading to a turnover.

    A pressure is "successful" if at least one of the next
    :data:`_TURNOVER_LOOKAHEAD` events in the full stream is a
    non-pressure event by the focal team (indicating possession
    was regained).

    Args:
        events: Full event stream for the segment.
        team_id: Focal team identifier.
        team_pressures: Pre-filtered list of focal-team pressure events.

    Returns:
        Success rate as a float, or ``None`` if no pressures exist.
    """
    if not team_pressures:
        return None

    pressure_ids = {p.event_id for p in team_pressures}
    successes = 0

    for i, evt in enumerate(events):
        if evt.event_id not in pressure_ids:
            continue
        lookahead = events[i + 1 : i + 1 + _TURNOVER_LOOKAHEAD]
        for la in lookahead:
            if la.team_id == team_id and la.event_type != "pressure":
                successes += 1
                break

    return successes / len(team_pressures)


def _ppda(
    events: tuple[NormalizedEvent, ...],
    team_id: str,
) -> float | None:
    """Compute passes allowed per defensive action.

    PPDA = opponent passes / (team tackles + team interceptions
    + team fouls committed).

    Args:
        events: Full event stream for the segment.
        team_id: Focal team identifier.

    Returns:
        PPDA value, or ``None`` when the denominator is zero.
    """
    opp_passes = sum(
        1 for e in events if e.team_id != team_id and e.event_type == "pass"
    )
    def_actions = sum(
        1
        for e in events
        if e.team_id == team_id
        and e.event_type in {"tackle", "interception", "foul_committed"}
    )
    if def_actions == 0:
        return None
    return opp_passes / def_actions


# ------------------------------------------------------------------
# Transition extractor
# ------------------------------------------------------------------

_TRANSITION_NAMES: tuple[str, ...] = (
    "t2_transition_counter_attack_indicator",
    "t2_transition_counter_press_indicator",
    "t2_transition_transition_speed",
)


class TransitionFeatureExtractor:
    """Extract transition features from possession changes.

    Detects turnovers (changes in possession between teams) and
    measures the speed and nature of the subsequent response:
    counter-attacks, counter-presses, and transition speed.
    """

    __slots__ = ()

    @property
    def tier(self) -> int:
        """Feature tier: 2 (event-derived)."""
        return 2

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names of transition features produced by this extractor."""
        return _TRANSITION_NAMES

    def extract(
        self,
        events: tuple[NormalizedEvent, ...],
        context: MatchContext,
    ) -> dict[str, float | None]:
        """Extract transition features from possession changes.

        Args:
            events: Tuple of normalised events in the segment.
            context: Per-team match context.

        Returns:
            Dict mapping each transition feature name to its value,
            or ``None`` when no relevant turnovers exist.
        """
        team_id = context.team_id
        turnovers_in = _find_turnovers(events, to_team=team_id)
        turnovers_against = _find_turnovers(events, from_team=team_id)

        counter_attack = _counter_attack(events, turnovers_in, team_id)
        counter_press = _counter_press(events, turnovers_against, team_id)
        speed = _transition_speed(events, turnovers_in, team_id)

        return {
            "t2_transition_counter_attack_indicator": counter_attack,
            "t2_transition_counter_press_indicator": counter_press,
            "t2_transition_transition_speed": speed,
        }


def _find_turnovers(
    events: tuple[NormalizedEvent, ...],
    *,
    to_team: str | None = None,
    from_team: str | None = None,
) -> list[int]:
    """Find indices where possession changes between teams.

    A turnover occurs at index ``i + 1`` when ``events[i].team_id``
    differs from ``events[i + 1].team_id``.

    Args:
        events: Full event stream.
        to_team: If given, only return turnovers where the gaining
            team matches this identifier.
        from_team: If given, only return turnovers where the losing
            team matches this identifier.

    Returns:
        List of indices into *events* pointing to the first event
        by the gaining team after each turnover.
    """
    indices: list[int] = []
    for i in range(len(events) - 1):
        if events[i].team_id == events[i + 1].team_id:
            continue
        if to_team is not None and events[i + 1].team_id != to_team:
            continue
        if from_team is not None and events[i].team_id != from_team:
            continue
        indices.append(i + 1)
    return indices


def _counter_attack(
    events: tuple[NormalizedEvent, ...],
    turnover_indices: list[int],
    team_id: str,
) -> float | None:
    """Detect counter-attack potential after turnovers in favour.

    Args:
        events: Full event stream.
        turnover_indices: Indices of turnovers where *team_id* gains
            possession.
        team_id: Focal team identifier.

    Returns:
        ``1.0`` if any turnover produces forward x-displacement
        exceeding :data:`_COUNTER_ATTACK_THRESHOLD` within the next
        events, ``0.0`` otherwise, or ``None`` if no turnovers.
    """
    if not turnover_indices:
        return None

    for idx in turnover_indices:
        ahead = [
            e
            for e in events[idx : idx + _TURNOVER_LOOKAHEAD]
            if e.team_id == team_id and e.location is not None
        ]
        if len(ahead) < 2:
            continue
        assert ahead[0].location is not None
        assert ahead[-1].location is not None
        displacement = ahead[-1].location[0] - ahead[0].location[0]
        if displacement > _COUNTER_ATTACK_THRESHOLD:
            return 1.0

    return 0.0


def _counter_press(
    events: tuple[NormalizedEvent, ...],
    turnover_indices: list[int],
    team_id: str,
) -> float | None:
    """Detect counter-pressing after turnovers against.

    Args:
        events: Full event stream.
        turnover_indices: Indices of turnovers where *team_id* loses
            possession.
        team_id: Focal team identifier.

    Returns:
        ``1.0`` if any turnover against is followed by
        :data:`_COUNTER_PRESS_MIN` or more pressure events by the
        focal team within the lookahead window, ``0.0`` otherwise,
        or ``None`` if no turnovers against.
    """
    if not turnover_indices:
        return None

    for idx in turnover_indices:
        ahead = events[idx : idx + _TURNOVER_LOOKAHEAD]
        press_count = sum(
            1 for e in ahead if e.team_id == team_id and e.event_type == "pressure"
        )
        if press_count >= _COUNTER_PRESS_MIN:
            return 1.0

    return 0.0


def _transition_speed(
    events: tuple[NormalizedEvent, ...],
    turnover_indices: list[int],
    team_id: str,
) -> float | None:
    """Compute mean transition speed after turnovers in favour.

    Speed is measured as forward x-displacement per second across
    the first few events after each turnover.

    Args:
        events: Full event stream.
        turnover_indices: Indices of turnovers where *team_id* gains
            possession.
        team_id: Focal team identifier.

    Returns:
        Mean transition speed (units per second), or ``None`` if no
        valid turnovers exist.
    """
    if not turnover_indices:
        return None

    speeds: list[float] = []
    for idx in turnover_indices:
        ahead = [
            e
            for e in events[idx : idx + _TURNOVER_LOOKAHEAD]
            if e.team_id == team_id and e.location is not None
        ]
        if len(ahead) < 2:
            continue
        assert ahead[0].location is not None
        assert ahead[-1].location is not None
        dx = ahead[-1].location[0] - ahead[0].location[0]
        dt = ahead[-1].timestamp - ahead[0].timestamp
        if dt > 0.0:
            speeds.append(dx / dt)

    if not speeds:
        return None
    return float(np.mean(np.array(speeds, dtype=np.float64)))
