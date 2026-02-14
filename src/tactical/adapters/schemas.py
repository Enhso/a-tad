"""Internal data schemas for the Tactical State Discovery Engine.

Defines the canonical data structures used across all adapters and
downstream pipeline stages. Every schema is a frozen, slotted dataclass
to guarantee immutability and memory efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass

CONTROLLED_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "pass",
        "carry",
        "shot",
        "duel",
        "pressure",
        "tackle",
        "interception",
        "clearance",
        "block",
        "foul_committed",
        "foul_won",
        "ball_receipt",
        "ball_recovery",
        "dispossessed",
        "miscontrol",
        "goal_keeper",
        "substitution",
        "set_piece_corner",
        "set_piece_free_kick",
        "set_piece_throw_in",
        "set_piece_goal_kick",
        "set_piece_kick_off",
        "set_piece_penalty",
    }
)


@dataclass(frozen=True, slots=True)
class FreezeFramePlayer:
    """A single player snapshot from a StatsBomb 360 freeze frame.

    Attributes:
        player_id: Unique identifier of the player.
        teammate: Whether the player is on the same team as the actor.
        location: Pitch coordinates ``(x, y)`` normalized to 0-100.
        position: Player position name (e.g. ``"Center Back"``).
    """

    player_id: str
    teammate: bool
    location: tuple[float, float]
    position: str


@dataclass(frozen=True, slots=True)
class NormalizedEvent:
    """Provider-agnostic representation of a single match event.

    All adapter implementations must map their raw events into this
    schema before passing data downstream.

    Attributes:
        event_id: Unique identifier of the event.
        match_id: Identifier of the match this event belongs to.
        team_id: Identifier of the acting team.
        player_id: Identifier of the acting player.
        period: Match period (1 = first half, 2 = second half, etc.).
        timestamp: Seconds elapsed since the start of the period.
        match_minute: Cumulative match minute (including stoppage).
        location: Pitch coordinates ``(x, y)`` of the event origin,
            or ``None`` if unavailable.
        end_location: Pitch coordinates ``(x, y)`` of the event
            destination, or ``None`` if unavailable.
        event_type: Canonical event type; must be a member of
            :data:`CONTROLLED_EVENT_TYPES`.
        event_outcome: Outcome label (e.g. ``"complete"``, ``"saved"``).
        under_pressure: Whether the acting player was under pressure.
        body_part: Body part used (e.g. ``"Right Foot"``, ``"Head"``).
        freeze_frame: Tuple of :class:`FreezeFramePlayer` snapshots
            from 360 data, or ``None`` when unavailable.
        score_home: Home team score at the time of the event.
        score_away: Away team score at the time of the event.
        team_is_home: Whether the acting team is the home team.
    """

    event_id: str
    match_id: str
    team_id: str
    player_id: str
    period: int
    timestamp: float
    match_minute: float
    location: tuple[float, float] | None
    end_location: tuple[float, float] | None
    event_type: str
    event_outcome: str
    under_pressure: bool
    body_part: str
    freeze_frame: tuple[FreezeFramePlayer, ...] | None
    score_home: int
    score_away: int
    team_is_home: bool

    def __post_init__(self) -> None:
        """Validate that *event_type* is a controlled value."""
        if self.event_type not in CONTROLLED_EVENT_TYPES:
            msg = (
                f"event_type must be one of CONTROLLED_EVENT_TYPES, "
                f"got {self.event_type!r}"
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class MatchInfo:
    """High-level metadata for a single match.

    Attributes:
        match_id: Unique identifier of the match.
        competition: Competition name (e.g. ``"La Liga"``).
        season: Season label (e.g. ``"2023/2024"``).
        home_team_id: Identifier of the home team.
        home_team_name: Display name of the home team.
        away_team_id: Identifier of the away team.
        away_team_name: Display name of the away team.
        home_score: Final score of the home team.
        away_score: Final score of the away team.
        match_date: Match date in ISO 8601 format (``YYYY-MM-DD``).
    """

    match_id: str
    competition: str
    season: str
    home_team_id: str
    home_team_name: str
    away_team_id: str
    away_team_name: str
    home_score: int
    away_score: int
    match_date: str


@dataclass(frozen=True, slots=True)
class CardEvent:
    """A card shown to a player during a match.

    Attributes:
        time: Timestamp string in ``MM:SS`` format.
        card_type: Card type label (e.g. ``"Yellow Card"``,
            ``"Second Yellow"``, ``"Red Card"``).
        reason: Reason the card was shown (e.g. ``"Foul Committed"``).
        period: Match period in which the card was shown.
    """

    time: str
    card_type: str
    reason: str
    period: int


@dataclass(frozen=True, slots=True)
class PositionSpell:
    """A contiguous period during which a player occupied one position.

    A player's match involvement is represented as a sequence of
    non-overlapping :class:`PositionSpell` entries ordered by time.

    Attributes:
        position_id: Provider-specific position identifier.
        position: Human-readable position name
            (e.g. ``"Center Forward"``).
        from_time: Start timestamp (``MM:SS``), or ``None`` if unknown.
        to_time: End timestamp (``MM:SS``), or ``None`` if the player
            remained in this position until the final whistle.
        from_period: Period in which this spell began, or ``None``.
        to_period: Period in which this spell ended, or ``None``.
        start_reason: Why the player entered this position
            (e.g. ``"Starting XI"``, ``"Tactical Shift"``).
        end_reason: Why the player left this position
            (e.g. ``"Final Whistle"``, ``"Substitution - Off (Tactical)"``).
    """

    position_id: int
    position: str
    from_time: str | None
    to_time: str | None
    from_period: int | None
    to_period: int | None
    start_reason: str
    end_reason: str


@dataclass(frozen=True, slots=True)
class PlayerLineup:
    """A single player's lineup entry for one match.

    Attributes:
        player_id: Unique identifier of the player.
        player_name: Full display name.
        jersey_number: Shirt number worn in this match.
        starter: ``True`` if the player was in the Starting XI.
        positions: Chronological sequence of position spells.
        cards: Cards received during the match (may be empty).
    """

    player_id: str
    player_name: str
    jersey_number: int
    starter: bool
    positions: tuple[PositionSpell, ...]
    cards: tuple[CardEvent, ...]


@dataclass(frozen=True, slots=True)
class TeamLineup:
    """Complete squad list for one team in a single match.

    Attributes:
        team_id: Unique identifier of the team.
        team_name: Display name of the team.
        players: All players in the squad (starters + substitutes).
    """

    team_id: str
    team_name: str
    players: tuple[PlayerLineup, ...]


@dataclass(frozen=True, slots=True)
class MatchContext:
    """Per-team context for a single match analysis run.

    Attributes:
        match_id: Unique identifier of the match.
        team_id: Identifier of the focal team.
        opponent_id: Identifier of the opposing team.
        team_is_home: Whether the focal team is the home team.
        has_360: Whether StatsBomb 360 freeze-frame data is available.
    """

    match_id: str
    team_id: str
    opponent_id: str
    team_is_home: bool
    has_360: bool
