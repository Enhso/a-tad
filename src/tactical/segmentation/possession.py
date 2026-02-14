"""Possession-based sequencing for match events.

Segments a match's normalised event stream into continuous possession
sequences, each owned by a single team and terminated by a turnover,
dead ball, shot, or end of period.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tactical.exceptions import SegmentationError

if TYPE_CHECKING:
    from tactical.adapters.schemas import NormalizedEvent
    from tactical.config import PossessionConfig

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

TURNOVER_EVENTS: frozenset[str] = frozenset({"interception", "tackle", "ball_recovery"})

DEAD_BALL_EVENTS: frozenset[str] = frozenset(
    {
        "set_piece_throw_in",
        "set_piece_goal_kick",
        "set_piece_corner",
        "set_piece_free_kick",
        "set_piece_kick_off",
    }
)

SET_PIECE_STARTS: frozenset[str] = frozenset(
    {
        "set_piece_corner",
        "set_piece_free_kick",
        "set_piece_throw_in",
        "set_piece_goal_kick",
        "set_piece_kick_off",
        "set_piece_penalty",
    }
)

_SKIP_EVENTS: frozenset[str] = frozenset({"substitution", "goal_keeper"})


# ------------------------------------------------------------------
# Dataclass
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PossessionSequence:
    """A continuous possession sequence owned by a single team.

    Attributes:
        match_id: Identifier of the match.
        team_id: Identifier of the possessing team.
        period: Match period this sequence belongs to.
        start_time: First event timestamp in seconds from period start.
        end_time: Last event timestamp in seconds from period start.
        match_minute_start: Match minute of the first event.
        match_minute_end: Match minute of the last event.
        events: Immutable sequence of events in this possession.
        outcome: How possession ended (``"shot"``, ``"turnover"``,
            ``"dead_ball"``, or ``"end_of_period"``).
        is_set_piece: Whether the sequence began with a set piece.
        set_piece_type: Canonical event type of the opening set piece,
            or ``None`` when *is_set_piece* is ``False``.
    """

    match_id: str
    team_id: str
    period: int
    start_time: float
    end_time: float
    match_minute_start: float
    match_minute_end: float
    events: tuple[NormalizedEvent, ...]
    outcome: str
    is_set_piece: bool
    set_piece_type: str | None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _build_sequence(
    events: list[NormalizedEvent],
    outcome: str,
) -> PossessionSequence:
    """Construct a :class:`PossessionSequence` from accumulated events.

    Args:
        events: Non-empty list of events belonging to one possession.
        outcome: Termination reason for this possession.

    Returns:
        A fully populated PossessionSequence.
    """
    first = events[0]
    last = events[-1]
    first_type = first.event_type
    is_set_piece = first_type in SET_PIECE_STARTS
    return PossessionSequence(
        match_id=first.match_id,
        team_id=first.team_id,
        period=first.period,
        start_time=first.timestamp,
        end_time=last.timestamp,
        match_minute_start=first.match_minute,
        match_minute_end=last.match_minute,
        events=tuple(events),
        outcome=outcome,
        is_set_piece=is_set_piece,
        set_piece_type=first_type if is_set_piece else None,
    )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def create_possession_sequences(
    events: list[NormalizedEvent],
    config: PossessionConfig,
) -> list[PossessionSequence]:
    """Segment a match's events into possession sequences.

    A possession is a continuous sequence of events by one team,
    terminated by turnover, dead ball, shot, or end of period.

    Args:
        events: Sorted list of normalized events for a single match.
        config: Possession configuration (min events).

    Returns:
        List of PossessionSequence objects, sorted chronologically.

    Raises:
        SegmentationError: If events list is empty.
    """
    if not events:
        msg = "Cannot create possession sequences from an empty event list"
        raise SegmentationError(msg)

    sorted_events = sorted(events, key=lambda e: (e.period, e.timestamp))

    raw: list[tuple[list[NormalizedEvent], str]] = []
    current: list[NormalizedEvent] = []
    current_team = ""
    current_period = -1

    for event in sorted_events:
        if event.event_type in _SKIP_EVENTS:
            continue

        # -- boundary detection (order matters) --
        if current:
            if event.period != current_period:
                raw.append((current, "end_of_period"))
                current = []
            elif event.event_type in DEAD_BALL_EVENTS and event.team_id != current_team:
                raw.append((current, "dead_ball"))
                current = []
            elif event.team_id != current_team:
                raw.append((current, "turnover"))
                current = []

        # -- initialise new sequence when empty --
        if not current:
            current_team = event.team_id
            current_period = event.period

        current.append(event)

        # -- shot terminates possession immediately --
        if event.event_type == "shot":
            raw.append((current, "shot"))
            current = []

    if current:
        raw.append((current, "end_of_period"))

    return [
        _build_sequence(evts, outcome)
        for evts, outcome in raw
        if len(evts) >= config.min_events
    ]
