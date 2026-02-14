"""StatsBomb data adapter for the Tactical State Discovery Engine.

Implements the :class:`~tactical.adapters.base.DataAdapter` protocol by
fetching match and event data from the StatsBomb open-data API via
``statsbombpy``, normalizing coordinates, mapping event types to the
controlled vocabulary, and caching raw results to disk.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import pandas as pd
from statsbombpy import sb  # type: ignore[import-untyped]

from tactical.adapters.cache import MatchCache
from tactical.adapters.schemas import (
    CardEvent,
    FreezeFramePlayer,
    MatchInfo,
    NormalizedEvent,
    PlayerLineup,
    PositionSpell,
    TeamLineup,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

STATSBOMB_PITCH_LENGTH: float = 120.0
STATSBOMB_PITCH_WIDTH: float = 80.0

STATSBOMB_EVENT_TYPE_MAP: dict[str, str] = {
    "Pass": "pass",
    "Ball Receipt*": "ball_receipt",
    "Carry": "carry",
    "Dribble": "carry",
    "Shot": "shot",
    "Pressure": "pressure",
    "Duel": "duel",
    "Foul Committed": "foul_committed",
    "Foul Won": "foul_won",
    "Interception": "interception",
    "Block": "block",
    "Clearance": "clearance",
    "Ball Recovery": "ball_recovery",
    "Dispossessed": "dispossessed",
    "Miscontrol": "miscontrol",
    "Goal Keeper": "goal_keeper",
    "Substitution": "substitution",
}

_PASS_TYPE_TO_SET_PIECE: dict[str, str] = {
    "Corner": "set_piece_corner",
    "Free Kick": "set_piece_free_kick",
    "Throw-in": "set_piece_throw_in",
    "Goal Kick": "set_piece_goal_kick",
    "Kick Off": "set_piece_kick_off",
}

_SHOT_TYPE_TO_SET_PIECE: dict[str, str] = {
    "Penalty": "set_piece_penalty",
}

_BODY_PART_COLUMNS: tuple[str, ...] = (
    "pass_body_part",
    "shot_body_part",
    "clearance_body_part",
    "goalkeeper_body_part",
)

_END_LOCATION_COLUMNS: tuple[str, ...] = (
    "pass_end_location",
    "carry_end_location",
    "shot_end_location",
    "goalkeeper_end_location",
)

_OUTCOME_COLUMNS: tuple[str, ...] = (
    "pass_outcome",
    "shot_outcome",
    "duel_outcome",
    "interception_outcome",
    "ball_receipt_outcome",
    "dribble_outcome",
    "goalkeeper_outcome",
)


def _is_nan(value: object) -> bool:
    """Return True if *value* is NaN or None."""
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


def _parse_timestamp(ts: str) -> float:
    """Convert a StatsBomb ``HH:MM:SS.mmm`` timestamp to seconds.

    Args:
        ts: Timestamp string in ``HH:MM:SS.mmm`` format.

    Returns:
        Total seconds as a float.
    """
    parts = ts.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600.0 + minutes * 60.0 + seconds


class StatsBombAdapter:
    """Adapter for loading and normalizing StatsBomb event and lineup data.

    Satisfies the :class:`~tactical.adapters.base.DataAdapter` protocol.
    Fetches data via ``statsbombpy``, normalizes coordinates to a 0-100
    scale, maps event types to the controlled vocabulary, and caches raw
    API responses to disk.

    Attributes:
        _cache: Disk-based cache for raw match data.
    """

    __slots__ = ("_cache",)

    def __init__(self, cache_dir: Path) -> None:
        """Initialize the adapter with a cache directory.

        Args:
            cache_dir: Directory for caching raw API responses.
        """
        self._cache = MatchCache(cache_dir)

    def list_matches(self, competition_id: int, season_id: int) -> list[MatchInfo]:
        """Return metadata for every match in a competition season.

        Args:
            competition_id: StatsBomb competition identifier.
            season_id: StatsBomb season identifier.

        Returns:
            List of :class:`MatchInfo` sorted by ``match_date``.
        """
        raw: dict[str, Any] = sb.matches(
            competition_id=competition_id,
            season_id=season_id,
            fmt="json",
        )
        matches: list[MatchInfo] = []
        for match_data in raw.values():
            home_team = match_data["home_team"]
            away_team = match_data["away_team"]
            matches.append(
                MatchInfo(
                    match_id=str(match_data["match_id"]),
                    competition=match_data["competition"]["competition_name"],
                    season=match_data["season"]["season_name"],
                    home_team_id=str(home_team["home_team_id"]),
                    home_team_name=home_team["home_team_name"],
                    away_team_id=str(away_team["away_team_id"]),
                    away_team_name=away_team["away_team_name"],
                    home_score=int(match_data["home_score"]),
                    away_score=int(match_data["away_score"]),
                    match_date=match_data["match_date"],
                )
            )
        matches.sort(key=lambda m: m.match_date)
        return matches

    def load_match_events(self, match_id: str) -> list[NormalizedEvent]:
        """Load and normalize all events for a single match.

        Checks the disk cache first. On a cache miss the raw event
        data is fetched from the StatsBomb API and cached for future
        calls.

        Args:
            match_id: StatsBomb match identifier (as string).

        Returns:
            List of :class:`NormalizedEvent` sorted by
            ``(period, timestamp)``.
        """
        cached = self._cache.get(match_id)
        if cached is not None:
            raw_df = pd.DataFrame.from_dict(cached)
        else:
            raw_df = sb.events(match_id=int(match_id), flatten_attrs=True)
            self._cache.put(match_id, raw_df.to_dict(orient="list"))
        return self._normalize_events(raw_df, match_id)

    def load_match_lineups(self, match_id: str) -> dict[str, TeamLineup]:
        """Load and normalize lineup data for both teams in a match.

        Checks the disk cache first.  On a cache miss the raw lineup
        data is fetched from the StatsBomb API and cached for future
        calls.

        Args:
            match_id: StatsBomb match identifier (as string).

        Returns:
            Dictionary mapping team-ID strings to :class:`TeamLineup`
            instances.
        """
        cache_key = f"{match_id}_lineups"
        raw_lineups: dict[str, Any]
        cached = self._cache.get(cache_key)
        if cached is not None:
            raw_lineups = cached
        else:
            raw_lineups = sb.lineups(match_id=int(match_id), fmt="dict")
            self._cache.put(cache_key, raw_lineups)
        return self._normalize_lineups(raw_lineups)

    def supports_360(self) -> bool:
        """Indicate that StatsBomb supports 360 freeze-frame data.

        Returns:
            Always ``True``.
        """
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_lineups(
        raw_lineups: dict[str, Any],
    ) -> dict[str, TeamLineup]:
        """Convert raw StatsBomb lineup dicts to normalized schemas.

        Args:
            raw_lineups: Dictionary keyed by team-ID string, each value
                a dict with ``team_id``, ``team_name``, and ``lineup``
                as returned by ``statsbombpy``.

        Returns:
            Dictionary mapping team-ID strings to :class:`TeamLineup`.
        """
        result: dict[str, TeamLineup] = {}
        for _team_key, team_data in raw_lineups.items():
            team_id = str(team_data["team_id"])
            players: list[PlayerLineup] = []
            for player_data in team_data.get("lineup", []):
                positions = _build_position_spells(player_data.get("positions", []))
                cards = _build_card_events(player_data.get("cards", []))
                starter = any(ps.start_reason == "Starting XI" for ps in positions)
                players.append(
                    PlayerLineup(
                        player_id=str(player_data["player_id"]),
                        player_name=player_data["player_name"],
                        jersey_number=int(player_data.get("jersey_number", 0)),
                        starter=starter,
                        positions=positions,
                        cards=cards,
                    )
                )
            result[team_id] = TeamLineup(
                team_id=team_id,
                team_name=team_data["team_name"],
                players=tuple(players),
            )
        return result

    @staticmethod
    def _normalize_coordinates(x: float, y: float) -> tuple[float, float]:
        """Convert StatsBomb 120x80 coordinates to 0-100 scale.

        StatsBomb normalizes coordinates so the possessing team always
        attacks left-to-right. The y-axis is flipped so that y=0 maps
        to the right touchline in our convention.

        Args:
            x: Raw x-coordinate (0-120).
            y: Raw y-coordinate (0-80, top touchline is 0).

        Returns:
            ``(x_norm, y_norm)`` in the 0-100 range.
        """
        x_norm = x / STATSBOMB_PITCH_LENGTH * 100.0
        y_norm = (STATSBOMB_PITCH_WIDTH - y) / STATSBOMB_PITCH_WIDTH * 100.0
        return (x_norm, y_norm)

    @staticmethod
    def _resolve_event_type(row: pd.Series[Any]) -> str | None:
        """Map a raw StatsBomb event row to a controlled event type.

        Returns ``None`` for event types that should be skipped.

        Args:
            row: A single row from the flattened events DataFrame.

        Returns:
            Controlled event type string, or ``None`` if unmapped.
        """
        sb_type: str = row["type"]

        if sb_type == "Pass":
            pass_type = row.get("pass_type")
            if not _is_nan(pass_type) and pass_type in _PASS_TYPE_TO_SET_PIECE:
                return _PASS_TYPE_TO_SET_PIECE[str(pass_type)]
            return "pass"

        if sb_type == "Shot":
            shot_type = row.get("shot_type")
            if not _is_nan(shot_type) and shot_type in _SHOT_TYPE_TO_SET_PIECE:
                return _SHOT_TYPE_TO_SET_PIECE[str(shot_type)]
            return "shot"

        if sb_type == "Duel":
            duel_type = row.get("duel_type")
            if not _is_nan(duel_type) and str(duel_type) == "Tackle":
                return "tackle"
            return "duel"

        return STATSBOMB_EVENT_TYPE_MAP.get(sb_type)

    @staticmethod
    def _extract_body_part(row: pd.Series[Any]) -> str:
        """Extract body part from the appropriate column.

        Args:
            row: A single row from the flattened events DataFrame.

        Returns:
            Lowercase body part string, or ``"unknown"``.
        """
        for col in _BODY_PART_COLUMNS:
            val = row.get(col)
            if not _is_nan(val):
                return str(val).lower().replace(" ", "_")
        return "unknown"

    @staticmethod
    def _extract_end_location(
        row: pd.Series[Any],
    ) -> tuple[float, float] | None:
        """Extract and normalize the event end location.

        Args:
            row: A single row from the flattened events DataFrame.

        Returns:
            Normalized ``(x, y)`` tuple or ``None``.
        """
        for col in _END_LOCATION_COLUMNS:
            val = row.get(col)
            if val is not None and not _is_nan(val):
                coords = val
                return StatsBombAdapter._normalize_coordinates(
                    float(coords[0]), float(coords[1])
                )
        return None

    @staticmethod
    def _extract_freeze_frame(
        row: pd.Series[Any],
    ) -> tuple[FreezeFramePlayer, ...] | None:
        """Extract freeze frame data from a raw event row.

        Checks the ``shot_freeze_frame`` column (available on shot
        events) and a generic ``freeze_frame`` column (present when
        360 data has been merged into the DataFrame).

        Args:
            row: A single row from the flattened events DataFrame.

        Returns:
            Tuple of :class:`FreezeFramePlayer` instances, or ``None``
            when no freeze frame data is available.
        """
        raw_ff: list[dict[str, Any]] | None = None

        shot_ff = row.get("shot_freeze_frame")
        if shot_ff is not None and not _is_nan(shot_ff):
            raw_ff = shot_ff

        if raw_ff is None:
            generic_ff = row.get("freeze_frame")
            if generic_ff is not None and not _is_nan(generic_ff):
                raw_ff = generic_ff

        if not raw_ff:
            return None

        players: list[FreezeFramePlayer] = []
        for entry in raw_ff:
            loc = entry.get("location")
            if loc is None:
                continue
            norm_loc = StatsBombAdapter._normalize_coordinates(
                float(loc[0]), float(loc[1])
            )

            player_info = entry.get("player")
            if isinstance(player_info, dict):
                player_id = str(player_info.get("id", "unknown"))
            else:
                player_id = "unknown"

            pos_info = entry.get("position")
            if isinstance(pos_info, dict):
                position = str(pos_info.get("name", "unknown"))
            else:
                position = "unknown"

            players.append(
                FreezeFramePlayer(
                    player_id=player_id,
                    teammate=bool(entry.get("teammate", False)),
                    location=norm_loc,
                    position=position,
                )
            )

        return tuple(players) if players else None

    @staticmethod
    def _extract_outcome(row: pd.Series[Any]) -> str:
        """Extract the event outcome from the appropriate column.

        For passes and ball receipts, a NaN outcome means the action
        was successful (``"complete"``).

        Args:
            row: A single row from the flattened events DataFrame.

        Returns:
            Lowercase outcome string.
        """
        sb_type: str = row["type"]

        if sb_type == "Pass":
            val = row.get("pass_outcome")
            if _is_nan(val):
                return "complete"
            return "incomplete"

        if sb_type == "Ball Receipt*":
            val = row.get("ball_receipt_outcome")
            if _is_nan(val):
                return "complete"
            return "incomplete"

        for col in _OUTCOME_COLUMNS:
            val = row.get(col)
            if not _is_nan(val):
                return str(val).lower().replace(" ", "_")

        return "success"

    def _normalize_events(
        self,
        raw_df: pd.DataFrame,
        match_id: str,
    ) -> list[NormalizedEvent]:
        """Convert a raw StatsBomb events DataFrame to normalized events.

        Args:
            raw_df: Flattened events DataFrame from ``statsbombpy``.
            match_id: Match identifier string.

        Returns:
            List of :class:`NormalizedEvent` sorted by
            ``(period, timestamp)``.
        """
        home_team_id = self._determine_home_team_id(raw_df)

        sorted_df = raw_df.sort_values(by=["period", "index"], ascending=True)

        score_home = 0
        score_away = 0
        events: list[NormalizedEvent] = []

        for _, row in sorted_df.iterrows():
            sb_type: str = row["type"]

            is_goal = (
                sb_type == "Shot"
                and not _is_nan(row.get("shot_outcome"))
                and str(row["shot_outcome"]) == "Goal"
            )
            if is_goal:
                team_id_val = int(row["team_id"])
                if team_id_val == home_team_id:
                    score_home += 1
                else:
                    score_away += 1

            event_type = self._resolve_event_type(row)
            if event_type is None:
                if sb_type in STATSBOMB_EVENT_TYPE_MAP:
                    pass
                elif sb_type not in {
                    "Starting XI",
                    "Half Start",
                    "Half End",
                    "Tactical Shift",
                    "Injury Stoppage",
                    "Referee Ball-Drop",
                    "Bad Behaviour",
                    "Error",
                    "50/50",
                    "Dribbled Past",
                    "Shield",
                    "Offside",
                    "Own Goal For",
                    "Own Goal Against",
                    "Player Off",
                    "Player On",
                }:
                    logger.warning(
                        "Unknown StatsBomb event type %r in match %s -- skipping",
                        sb_type,
                        match_id,
                    )
                continue

            player_id_raw = row.get("player_id")
            if _is_nan(player_id_raw) or player_id_raw is None:
                player_id = "unknown"
            else:
                player_id = str(int(float(str(player_id_raw))))

            timestamp_seconds = _parse_timestamp(row["timestamp"])
            period = int(row["period"])

            period_offset = 45.0 * (period - 1)
            match_minute = period_offset + row["minute"] + row["second"] / 60.0

            location_raw = row.get("location")
            location: tuple[float, float] | None = None
            if not _is_nan(location_raw) and location_raw is not None:
                location = self._normalize_coordinates(
                    float(location_raw[0]), float(location_raw[1])
                )

            end_location = self._extract_end_location(row)

            under_pressure_raw = row.get("under_pressure")
            under_pressure = (
                bool(under_pressure_raw) if not _is_nan(under_pressure_raw) else False
            )

            team_id_int = int(row["team_id"])
            team_is_home = team_id_int == home_team_id

            freeze_frame = self._extract_freeze_frame(row)

            events.append(
                NormalizedEvent(
                    event_id=row["id"],
                    match_id=match_id,
                    team_id=str(team_id_int),
                    player_id=player_id,
                    period=period,
                    timestamp=timestamp_seconds,
                    match_minute=match_minute,
                    location=location,
                    end_location=end_location,
                    event_type=event_type,
                    event_outcome=self._extract_outcome(row),
                    under_pressure=under_pressure,
                    body_part=self._extract_body_part(row),
                    freeze_frame=freeze_frame,
                    score_home=score_home,
                    score_away=score_away,
                    team_is_home=team_is_home,
                )
            )

        events.sort(key=lambda e: (e.period, e.timestamp))
        return events

    @staticmethod
    def _determine_home_team_id(raw_df: pd.DataFrame) -> int:
        """Determine the home team ID from Starting XI events.

        In StatsBomb data the first Starting XI event (lowest index)
        always belongs to the home team.

        Args:
            raw_df: Flattened events DataFrame.

        Returns:
            Integer team ID of the home team.
        """
        starting_xi = raw_df[raw_df["type"] == "Starting XI"].sort_values("index")
        return int(starting_xi.iloc[0]["team_id"])


# ------------------------------------------------------------------
# Module-level helpers for lineup normalization
# ------------------------------------------------------------------


def _build_position_spells(
    raw_positions: list[dict[str, Any]],
) -> tuple[PositionSpell, ...]:
    """Convert raw StatsBomb position entries to :class:`PositionSpell` tuples.

    Args:
        raw_positions: List of position dicts from the lineup API.

    Returns:
        Tuple of :class:`PositionSpell` instances.
    """
    spells: list[PositionSpell] = []
    for pos in raw_positions:
        from_time = pos.get("from")
        to_time = pos.get("to")
        from_period = pos.get("from_period")
        to_period = pos.get("to_period")
        spells.append(
            PositionSpell(
                position_id=int(pos["position_id"]),
                position=pos["position"],
                from_time=from_time if from_time is not None else None,
                to_time=to_time if to_time is not None else None,
                from_period=int(from_period) if from_period is not None else None,
                to_period=int(to_period) if to_period is not None else None,
                start_reason=pos.get("start_reason", "unknown"),
                end_reason=pos.get("end_reason", "unknown"),
            )
        )
    return tuple(spells)


def _build_card_events(
    raw_cards: list[dict[str, Any]],
) -> tuple[CardEvent, ...]:
    """Convert raw StatsBomb card entries to :class:`CardEvent` tuples.

    Args:
        raw_cards: List of card dicts from the lineup API.

    Returns:
        Tuple of :class:`CardEvent` instances.
    """
    cards: list[CardEvent] = []
    for card in raw_cards:
        cards.append(
            CardEvent(
                time=card["time"],
                card_type=card["card_type"],
                reason=card.get("reason", "unknown"),
                period=int(card["period"]),
            )
        )
    return tuple(cards)
