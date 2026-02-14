"""Tier 3 freeze-frame / 360 feature extractors.

Provides two extractor classes satisfying the
:class:`~tactical.features.base.FeatureExtractor` protocol:

* :class:`FormationFeatureExtractor` -- team shape metrics derived from
  teammate positions in the freeze frame (centroid, spread, convex hull,
  defensive/midfield/attacking lines, eccentricity).
* :class:`RelationalFeatureExtractor` -- spatial relationships between
  teammates and opponents (nearest-opponent distances, opponent centroid,
  opponent defensive line).

Both are Tier 3: they require StatsBomb 360 freeze-frame data and are
only available for matches where ``MatchContext.has_360`` is ``True``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import ConvexHull, QhullError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tactical.adapters.schemas import (
        FreezeFramePlayer,
        MatchContext,
        NormalizedEvent,
    )

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_MIN_TEAMMATES: int = 3
_MIN_OPPONENTS: int = 3
_DEFENSIVE_LINE_COUNT: int = 3
_ATTACKING_LINE_COUNT: int = 3

# ------------------------------------------------------------------
# Formation feature names
# ------------------------------------------------------------------

_FORMATION_NAMES: tuple[str, ...] = (
    "t3_formation_team_centroid_x",
    "t3_formation_team_centroid_y",
    "t3_formation_team_spread_x",
    "t3_formation_team_spread_y",
    "t3_formation_team_width",
    "t3_formation_team_length",
    "t3_formation_convex_hull_area",
    "t3_formation_convex_hull_perimeter",
    "t3_formation_defensive_line",
    "t3_formation_midfield_line",
    "t3_formation_attacking_line",
    "t3_formation_def_mid_gap",
    "t3_formation_mid_att_gap",
    "t3_formation_formation_eccentricity",
)

# ------------------------------------------------------------------
# Relational feature names
# ------------------------------------------------------------------

_RELATIONAL_NAMES: tuple[str, ...] = (
    "t3_relational_avg_nearest_opponent_dist",
    "t3_relational_min_nearest_opponent_dist",
    "t3_relational_opp_centroid_x",
    "t3_relational_opp_spread_x",
    "t3_relational_opp_defensive_line",
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _last_freeze_frame(
    events: tuple[NormalizedEvent, ...],
) -> tuple[FreezeFramePlayer, ...] | None:
    """Return the freeze frame from the last event that has one.

    Args:
        events: Sequence of normalised events in the segment.

    Returns:
        The freeze-frame tuple, or ``None`` if no event carries one.
    """
    for event in reversed(events):
        if event.freeze_frame is not None:
            return event.freeze_frame
    return None


def _split_players(
    freeze_frame: tuple[FreezeFramePlayer, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Separate teammates and opponents into coordinate arrays.

    Args:
        freeze_frame: Freeze-frame player snapshots.

    Returns:
        ``(teammates, opponents)`` – each an ``(N, 2)`` array of
        ``(x, y)`` positions. Either may be empty (shape ``(0, 2)``).
    """
    teammates: list[tuple[float, float]] = []
    opponents: list[tuple[float, float]] = []
    for player in freeze_frame:
        if player.teammate:
            teammates.append(player.location)
        else:
            opponents.append(player.location)

    tm = np.array(teammates, dtype=np.float64).reshape(-1, 2)
    opp = np.array(opponents, dtype=np.float64).reshape(-1, 2)
    return tm, opp


def _formation_eccentricity(positions: NDArray[np.float64]) -> float | None:
    """Compute formation eccentricity via PCA on player positions.

    Fits an ellipse using eigenvalues of the covariance matrix.
    Eccentricity = sqrt(1 - (minor / major)^2).

    Args:
        positions: ``(N, 2)`` array of teammate ``(x, y)`` positions.

    Returns:
        Eccentricity in ``[0, 1]``, or ``None`` when fewer than 3
        players are available.
    """
    if len(positions) < _MIN_TEAMMATES:
        return None

    cov = np.cov(positions, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    # eigvalsh returns eigenvalues in ascending order
    minor, major = float(eigenvalues[0]), float(eigenvalues[1])

    if major <= 0.0:
        return 0.0

    ratio = minor / major
    return math.sqrt(max(0.0, 1.0 - ratio * ratio))


# ------------------------------------------------------------------
# Formation extractor
# ------------------------------------------------------------------


class FormationFeatureExtractor:
    """Extract formation features from 360 freeze-frame teammate positions.

    Uses the last event in the window that carries a non-``None``
    ``freeze_frame``.  If no such event exists, every feature returns
    ``None``.  Teammate features additionally require at least 3
    teammates in the freeze frame.
    """

    __slots__ = ()

    @property
    def tier(self) -> int:
        """Feature tier: 3 (360/enriched)."""
        return 3

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names of features this extractor produces."""
        return _FORMATION_NAMES

    def extract(
        self,
        events: tuple[NormalizedEvent, ...],
        context: MatchContext,
    ) -> dict[str, float | None]:
        """Extract formation features from freeze-frame data.

        Args:
            events: Tuple of normalised events in the segment.
            context: Per-team match context.

        Returns:
            Dictionary mapping feature names to values.
        """
        nones: dict[str, float | None] = {n: None for n in _FORMATION_NAMES}

        ff = _last_freeze_frame(events)
        if ff is None:
            return nones

        teammates, _ = _split_players(ff)

        if len(teammates) < _MIN_TEAMMATES:
            return nones

        xs: NDArray[np.float64] = teammates[:, 0]
        ys: NDArray[np.float64] = teammates[:, 1]

        centroid_x = float(np.mean(xs))
        centroid_y = float(np.mean(ys))
        spread_x = float(np.std(xs, ddof=0))
        spread_y = float(np.std(ys, ddof=0))
        team_width = float(np.max(ys) - np.min(ys))
        team_length = float(np.max(xs) - np.min(xs))

        # Convex hull
        hull_area: float | None = None
        hull_perimeter: float | None = None
        try:
            hull = ConvexHull(teammates)
            hull_area = float(hull.volume)  # 2D: volume == area
            hull_perimeter = float(hull.area)  # 2D: area == perimeter
        except QhullError:
            pass

        # Line features
        sorted_x = np.sort(xs)
        defensive_line = float(np.mean(sorted_x[:_DEFENSIVE_LINE_COUNT]))
        midfield_line = float(np.median(xs))
        attacking_line = float(np.mean(sorted_x[-_ATTACKING_LINE_COUNT:]))
        def_mid_gap = midfield_line - defensive_line
        mid_att_gap = attacking_line - midfield_line

        # Eccentricity
        eccentricity = _formation_eccentricity(teammates)

        return {
            "t3_formation_team_centroid_x": centroid_x,
            "t3_formation_team_centroid_y": centroid_y,
            "t3_formation_team_spread_x": spread_x,
            "t3_formation_team_spread_y": spread_y,
            "t3_formation_team_width": team_width,
            "t3_formation_team_length": team_length,
            "t3_formation_convex_hull_area": hull_area,
            "t3_formation_convex_hull_perimeter": hull_perimeter,
            "t3_formation_defensive_line": defensive_line,
            "t3_formation_midfield_line": midfield_line,
            "t3_formation_attacking_line": attacking_line,
            "t3_formation_def_mid_gap": def_mid_gap,
            "t3_formation_mid_att_gap": mid_att_gap,
            "t3_formation_formation_eccentricity": eccentricity,
        }


# ------------------------------------------------------------------
# Relational extractor
# ------------------------------------------------------------------


class RelationalFeatureExtractor:
    """Extract relational features between teammates and opponents.

    Uses the last event in the window with a non-``None``
    ``freeze_frame``.  Requires both teammates and at least 3
    opponents.  If either condition is unmet, all features return
    ``None``.
    """

    __slots__ = ()

    @property
    def tier(self) -> int:
        """Feature tier: 3 (360/enriched)."""
        return 3

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names of features this extractor produces."""
        return _RELATIONAL_NAMES

    def extract(
        self,
        events: tuple[NormalizedEvent, ...],
        context: MatchContext,
    ) -> dict[str, float | None]:
        """Extract relational features from freeze-frame data.

        Args:
            events: Tuple of normalised events in the segment.
            context: Per-team match context.

        Returns:
            Dictionary mapping feature names to values.
        """
        nones: dict[str, float | None] = {n: None for n in _RELATIONAL_NAMES}

        ff = _last_freeze_frame(events)
        if ff is None:
            return nones

        teammates, opponents = _split_players(ff)

        if len(teammates) == 0 or len(opponents) < _MIN_OPPONENTS:
            return nones

        # Nearest-opponent distances for each teammate
        # Use broadcasting: (T, 1, 2) - (1, O, 2) -> (T, O, 2)
        diffs = teammates[:, np.newaxis, :] - opponents[np.newaxis, :, :]
        dists: NDArray[np.float64] = np.sqrt(np.sum(diffs * diffs, axis=2))
        nearest_dists: NDArray[np.float64] = np.min(dists, axis=1)

        avg_nearest = float(np.mean(nearest_dists))
        min_nearest = float(np.min(nearest_dists))

        opp_xs: NDArray[np.float64] = opponents[:, 0]
        opp_centroid_x = float(np.mean(opp_xs))
        opp_spread_x = float(np.std(opp_xs, ddof=0))

        # Opponent defensive line: mean x of opponent's deepest players.
        # Opponents attack the other way, so their deepest defenders
        # have the highest x values.
        n_opp_line = min(_DEFENSIVE_LINE_COUNT, len(opponents))
        sorted_opp_x = np.sort(opp_xs)
        opp_defensive_line = float(np.mean(sorted_opp_x[-n_opp_line:]))

        return {
            "t3_relational_avg_nearest_opponent_dist": avg_nearest,
            "t3_relational_min_nearest_opponent_dist": min_nearest,
            "t3_relational_opp_centroid_x": opp_centroid_x,
            "t3_relational_opp_spread_x": opp_spread_x,
            "t3_relational_opp_defensive_line": opp_defensive_line,
        }
