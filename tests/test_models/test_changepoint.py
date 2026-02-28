"""Tests for :mod:`tactical.models.changepoint`.

Uses synthetic signals with known changepoints to exercise detection,
feature delta computation, per-match batch detection, and edge cases.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from tactical.config import ChangepointConfig
from tactical.models.changepoint import (
    Changepoint,
    ChangepointResult,
    detect_changepoints,
    detect_changepoints_per_match,
)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_RNG_SEED = 42
_N_WINDOWS = 100
_N_FEATURES = 4
_CP_INDEX = 50
_FEATURE_NAMES = ["feat_a", "feat_b", "feat_c", "feat_d"]


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(seed=_RNG_SEED)


@pytest.fixture()
def signal_with_changepoint(rng: np.random.Generator) -> np.ndarray:
    """Synthetic 2-D signal with a clear mean shift at index 50.

    Before index 50: mean ~0, after index 50: mean ~10.
    """
    before = rng.normal(loc=0.0, scale=0.3, size=(_CP_INDEX, _N_FEATURES))
    after = rng.normal(loc=10.0, scale=0.3, size=(_N_WINDOWS - _CP_INDEX, _N_FEATURES))
    return np.vstack([before, after])


@pytest.fixture()
def match_minutes() -> np.ndarray:
    """Match minutes array: 0.0 to 90.0 evenly spaced over 100 windows."""
    return np.linspace(0.0, 90.0, _N_WINDOWS)


@pytest.fixture()
def constant_signal(rng: np.random.Generator) -> np.ndarray:
    """Constant signal with no structural change (tiny noise)."""
    return rng.normal(loc=5.0, scale=0.01, size=(_N_WINDOWS, _N_FEATURES))


@pytest.fixture()
def default_config() -> ChangepointConfig:
    """Default PELT config with rbf model."""
    return ChangepointConfig(
        method="pelt",
        model="rbf",
        penalty=10.0,
        min_segment_size=5,
    )


@pytest.fixture()
def two_match_df(rng: np.random.Generator) -> pl.DataFrame:
    """Polars DataFrame with two matches, each having a changepoint."""
    rows: list[dict[str, object]] = []
    for match_id in ("match_a", "match_b"):
        shift = 0.0 if match_id == "match_a" else 5.0
        for i in range(_N_WINDOWS):
            mean = shift if i < _CP_INDEX else shift + 10.0
            rows.append(
                {
                    "match_id": match_id,
                    "match_minute": float(i) * 90.0 / (_N_WINDOWS - 1),
                    "feat_a": float(rng.normal(mean, 0.3)),
                    "feat_b": float(rng.normal(mean, 0.3)),
                }
            )
    return pl.DataFrame(rows)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestDetectSingleChangepoint:
    """Verify detection of a single obvious changepoint."""

    def test_detect_single_changepoint(
        self,
        signal_with_changepoint: np.ndarray,
        match_minutes: np.ndarray,
        default_config: ChangepointConfig,
    ) -> None:
        """A clear mean-shift at index 50 should be detected nearby."""
        result = detect_changepoints(
            signal=signal_with_changepoint,
            match_minutes=match_minutes,
            match_id="test_match",
            config=default_config,
            feature_names=_FEATURE_NAMES,
        )

        assert isinstance(result, ChangepointResult)
        assert result.match_id == "test_match"
        assert len(result.changepoints) >= 1

        # At least one changepoint should be within +/- 5 of index 50
        cp_indices = [cp.index for cp in result.changepoints]
        assert any(abs(idx - _CP_INDEX) <= 5 for idx in cp_indices), (
            f"Expected changepoint near {_CP_INDEX}, got {cp_indices}"
        )


class TestDetectNoChangepoint:
    """Verify that a constant signal produces no changepoints."""

    def test_detect_no_changepoint(
        self,
        constant_signal: np.ndarray,
        match_minutes: np.ndarray,
    ) -> None:
        """Constant signal with high penalty should yield 0 changepoints."""
        config = ChangepointConfig(
            method="pelt",
            model="rbf",
            penalty=1000.0,
            min_segment_size=5,
        )
        result = detect_changepoints(
            signal=constant_signal,
            match_minutes=match_minutes,
            match_id="flat_match",
            config=config,
        )

        assert len(result.changepoints) == 0
        assert result.n_segments == 1


class TestChangepointMatchMinute:
    """Verify that match_minute is populated correctly."""

    def test_changepoint_match_minute(
        self,
        signal_with_changepoint: np.ndarray,
        match_minutes: np.ndarray,
        default_config: ChangepointConfig,
    ) -> None:
        """Each changepoint should carry the correct match minute."""
        result = detect_changepoints(
            signal=signal_with_changepoint,
            match_minutes=match_minutes,
            match_id="minute_check",
            config=default_config,
        )

        for cp in result.changepoints:
            expected_minute = float(match_minutes[cp.index])
            assert cp.match_minute == pytest.approx(expected_minute)


class TestChangepointFeatureDeltas:
    """Verify that feature deltas are computed for top features."""

    def test_changepoint_feature_deltas(
        self,
        signal_with_changepoint: np.ndarray,
        match_minutes: np.ndarray,
        default_config: ChangepointConfig,
    ) -> None:
        """Feature deltas should be populated when feature_names given."""
        result = detect_changepoints(
            signal=signal_with_changepoint,
            match_minutes=match_minutes,
            match_id="delta_check",
            config=default_config,
            feature_names=_FEATURE_NAMES,
        )

        assert len(result.changepoints) >= 1
        cp = result.changepoints[0]
        assert len(cp.feature_deltas) > 0
        assert len(cp.feature_deltas) <= len(_FEATURE_NAMES)

        # All delta keys should be valid feature names
        for name in cp.feature_deltas:
            assert name in _FEATURE_NAMES

        # The shift is ~+10, so deltas should be positive and large
        for delta_val in cp.feature_deltas.values():
            assert delta_val > 5.0

    def test_no_deltas_without_feature_names(
        self,
        signal_with_changepoint: np.ndarray,
        match_minutes: np.ndarray,
        default_config: ChangepointConfig,
    ) -> None:
        """Omitting feature_names should produce empty deltas."""
        result = detect_changepoints(
            signal=signal_with_changepoint,
            match_minutes=match_minutes,
            match_id="no_names",
            config=default_config,
            feature_names=None,
        )

        for cp in result.changepoints:
            assert cp.feature_deltas == {}


class TestMinSegmentSizeRespected:
    """Verify that no changepoints are closer than min_segment_size."""

    def test_min_segment_size_respected(
        self,
        rng: np.random.Generator,
    ) -> None:
        """Changepoints must be at least min_segment_size apart."""
        min_seg = 15
        config = ChangepointConfig(
            method="pelt",
            model="rbf",
            penalty=5.0,
            min_segment_size=min_seg,
        )

        # Create signal with multiple shifts
        n = 200
        signal = np.empty((n, 2))
        signal[:50] = rng.normal(0.0, 0.3, (50, 2))
        signal[50:100] = rng.normal(10.0, 0.3, (50, 2))
        signal[100:150] = rng.normal(-5.0, 0.3, (50, 2))
        signal[150:] = rng.normal(8.0, 0.3, (50, 2))
        minutes = np.linspace(0.0, 90.0, n)

        result = detect_changepoints(
            signal=signal,
            match_minutes=minutes,
            match_id="min_seg_test",
            config=config,
        )

        indices = [0] + [cp.index for cp in result.changepoints] + [n]
        for i in range(1, len(indices)):
            segment_len = indices[i] - indices[i - 1]
            assert segment_len >= min_seg, (
                f"Segment [{indices[i - 1]}, {indices[i]}) has length "
                f"{segment_len} < min_segment_size={min_seg}"
            )


class TestDetectPerMatch:
    """Verify batch per-match detection on a DataFrame."""

    def test_detect_per_match(
        self,
        two_match_df: pl.DataFrame,
    ) -> None:
        """Two-match DataFrame should produce two ChangepointResults."""
        config = ChangepointConfig(
            method="pelt",
            model="rbf",
            penalty=10.0,
            min_segment_size=5,
        )
        results = detect_changepoints_per_match(
            features_df=two_match_df,
            signal_columns=["feat_a", "feat_b"],
            config=config,
        )

        assert len(results) == 2
        match_ids = {r.match_id for r in results}
        assert match_ids == {"match_a", "match_b"}

        for r in results:
            assert isinstance(r, ChangepointResult)
            assert r.n_segments == len(r.changepoints) + 1
            # Each match has an obvious shift, so at least 1 cp expected
            assert len(r.changepoints) >= 1


class TestEmptySignalRaises:
    """Verify that invalid inputs raise ValueError."""

    def test_empty_signal_raises(
        self,
        default_config: ChangepointConfig,
    ) -> None:
        """An empty 2-D signal should raise ValueError."""
        signal = np.empty((0, 3))
        minutes = np.array([])

        with pytest.raises(ValueError, match="empty"):
            detect_changepoints(
                signal=signal,
                match_minutes=minutes,
                match_id="empty",
                config=default_config,
            )

    def test_1d_signal_raises(
        self,
        default_config: ChangepointConfig,
    ) -> None:
        """A 1-D signal should raise ValueError."""
        signal = np.ones(50)
        minutes = np.linspace(0.0, 45.0, 50)

        with pytest.raises(ValueError, match="2-D"):
            detect_changepoints(
                signal=signal,
                match_minutes=minutes,
                match_id="one_d",
                config=default_config,
            )


class TestNSegmentsConsistency:
    """Verify n_segments == len(changepoints) + 1."""

    def test_n_segments_equals_changepoints_plus_one(
        self,
        signal_with_changepoint: np.ndarray,
        match_minutes: np.ndarray,
        default_config: ChangepointConfig,
    ) -> None:
        """n_segments should always be one more than changepoints count."""
        result = detect_changepoints(
            signal=signal_with_changepoint,
            match_minutes=match_minutes,
            match_id="seg_check",
            config=default_config,
        )

        assert result.n_segments == len(result.changepoints) + 1


class TestAlternativeMethods:
    """Verify binseg and window methods work."""

    @pytest.mark.parametrize(
        ("method", "model", "penalty"),
        [
            ("binseg", "rbf", 10.0),
            ("window", "l2", 5.0),
        ],
    )
    def test_alternative_method(
        self,
        signal_with_changepoint: np.ndarray,
        match_minutes: np.ndarray,
        method: str,
        model: str,
        penalty: float,
    ) -> None:
        """Non-PELT methods should also detect the obvious changepoint."""
        config = ChangepointConfig(
            method=method,
            model=model,
            penalty=penalty,
            min_segment_size=5,
        )
        result = detect_changepoints(
            signal=signal_with_changepoint,
            match_minutes=match_minutes,
            match_id=f"{method}_test",
            config=config,
        )

        cp_indices = [cp.index for cp in result.changepoints]
        assert any(abs(idx - _CP_INDEX) <= 5 for idx in cp_indices), (
            f"Method '{method}' missed changepoint near {_CP_INDEX}: {cp_indices}"
        )

    def test_unknown_method_raises(self) -> None:
        """An unknown method string should raise ValueError."""
        config = ChangepointConfig(method="unknown", model="rbf")
        signal = np.ones((20, 2))
        minutes = np.linspace(0.0, 18.0, 20)

        with pytest.raises(ValueError, match="Unknown changepoint method"):
            detect_changepoints(
                signal=signal,
                match_minutes=minutes,
                match_id="bad_method",
                config=config,
            )


class TestDataclassesImmutable:
    """Verify that dataclasses are frozen."""

    def test_changepoint_frozen(self) -> None:
        """Changepoint should be immutable."""
        cp = Changepoint(index=10, match_minute=9.0, feature_deltas={})
        with pytest.raises(AttributeError):
            cp.index = 20  # type: ignore[misc]

    def test_changepointresult_frozen(self) -> None:
        """ChangepointResult should be immutable."""
        result = ChangepointResult(match_id="m", changepoints=(), n_segments=1)
        with pytest.raises(AttributeError):
            result.match_id = "x"  # type: ignore[misc]
