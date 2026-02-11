"""Tests for tactical.config dataclasses.

Verifies frozen/slotted invariants, default values, and PipelineConfig
validation logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tactical.config import (
    ChangepointConfig,
    GMMConfig,
    HMMConfig,
    PipelineConfig,
    PossessionConfig,
    VAEConfig,
    WindowConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_CONFIGS = [
    WindowConfig,
    PossessionConfig,
    GMMConfig,
    HMMConfig,
    VAEConfig,
    ChangepointConfig,
    PipelineConfig,
]


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


class TestFrozenSlottedInvariants:
    """All config dataclasses must be frozen and slotted."""

    @pytest.mark.parametrize("cls", ALL_CONFIGS)
    def test_frozen(self, cls: type) -> None:
        """Assigning to a field on a frozen dataclass must raise."""
        instance = cls()
        first_field = next(iter(instance.__dataclass_fields__))
        with pytest.raises(AttributeError):
            setattr(instance, first_field, None)

    @pytest.mark.parametrize("cls", ALL_CONFIGS)
    def test_slotted(self, cls: type) -> None:
        """Slotted dataclasses must define __slots__."""
        assert hasattr(cls, "__slots__")

    @pytest.mark.parametrize("cls", ALL_CONFIGS)
    def test_no_instance_dict(self, cls: type) -> None:
        """Slotted instances must not have a __dict__."""
        instance = cls()
        assert not hasattr(instance, "__dict__")


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


class TestWindowConfigDefaults:
    """WindowConfig should expose correct defaults."""

    def test_defaults(self) -> None:
        cfg = WindowConfig()
        assert cfg.window_seconds == 15.0
        assert cfg.overlap_seconds == 5.0
        assert cfg.min_events == 3


class TestPossessionConfigDefaults:
    """PossessionConfig should expose correct defaults."""

    def test_defaults(self) -> None:
        cfg = PossessionConfig()
        assert cfg.min_events == 3


class TestGMMConfigDefaults:
    """GMMConfig should expose correct defaults."""

    def test_defaults(self) -> None:
        cfg = GMMConfig()
        assert cfg.k_min == 3
        assert cfg.k_max == 12
        assert cfg.covariance_type == "full"
        assert cfg.n_init == 20
        assert cfg.random_state == 42
        assert cfg.use_bayesian is False


class TestHMMConfigDefaults:
    """HMMConfig should expose correct defaults."""

    def test_defaults(self) -> None:
        cfg = HMMConfig()
        assert cfg.n_states == 6
        assert cfg.covariance_type == "full"
        assert cfg.n_iter == 100
        assert cfg.random_state == 42
        assert cfg.init_from_gmm is True


class TestVAEConfigDefaults:
    """VAEConfig should expose correct defaults."""

    def test_defaults(self) -> None:
        cfg = VAEConfig()
        assert cfg.latent_dim == 8
        assert cfg.hidden_dims == (256, 128, 64)
        assert cfg.beta == 4.0
        assert cfg.learning_rate == 1e-3
        assert cfg.batch_size == 512
        assert cfg.n_epochs == 150
        assert cfg.dropout == 0.2
        assert cfg.random_state == 42


class TestChangepointConfigDefaults:
    """ChangepointConfig should expose correct defaults."""

    def test_defaults(self) -> None:
        cfg = ChangepointConfig()
        assert cfg.method == "pelt"
        assert cfg.model == "rbf"
        assert cfg.penalty == 10.0
        assert cfg.min_segment_size == 5


class TestPipelineConfigDefaults:
    """PipelineConfig should expose correct defaults and sub-configs."""

    def test_scalar_defaults(self) -> None:
        cfg = PipelineConfig()
        assert cfg.cache_dir == Path("data/cache")
        assert cfg.output_dir == Path("data/output")
        assert cfg.model_dir == Path("data/models")
        assert cfg.max_feature_tier == 2
        assert cfg.pca_variance_threshold == 0.95
        assert cfg.null_strategy == "drop_rows"
        assert cfg.match_phases == (
            (0, 15),
            (15, 45),
            (45, 60),
            (60, 75),
            (75, 90),
        )

    def test_nested_config_types(self) -> None:
        cfg = PipelineConfig()
        assert isinstance(cfg.window, WindowConfig)
        assert isinstance(cfg.possession, PossessionConfig)
        assert isinstance(cfg.gmm, GMMConfig)
        assert isinstance(cfg.hmm, HMMConfig)
        assert isinstance(cfg.vae, VAEConfig)
        assert isinstance(cfg.changepoint, ChangepointConfig)

    def test_nested_configs_are_independent_instances(self) -> None:
        """Each PipelineConfig must get its own sub-config instances."""
        a = PipelineConfig()
        b = PipelineConfig()
        assert a.window is not b.window
        assert a.gmm is not b.gmm


# ---------------------------------------------------------------------------
# PipelineConfig validation
# ---------------------------------------------------------------------------


class TestPipelineConfigValidation:
    """PipelineConfig.__post_init__ must reject invalid inputs."""

    @pytest.mark.parametrize("tier", [0, -1, 4, 100])
    def test_invalid_max_feature_tier(self, tier: int) -> None:
        with pytest.raises(ValueError, match="max_feature_tier"):
            PipelineConfig(max_feature_tier=tier)

    @pytest.mark.parametrize("tier", [1, 2, 3])
    def test_valid_max_feature_tier(self, tier: int) -> None:
        cfg = PipelineConfig(max_feature_tier=tier)
        assert cfg.max_feature_tier == tier

    @pytest.mark.parametrize("strategy", ["", "drop_all", "mean", "zero_fill"])
    def test_invalid_null_strategy(self, strategy: str) -> None:
        with pytest.raises(ValueError, match="null_strategy"):
            PipelineConfig(null_strategy=strategy)

    @pytest.mark.parametrize("strategy", ["drop_rows", "impute_median"])
    def test_valid_null_strategy(self, strategy: str) -> None:
        cfg = PipelineConfig(null_strategy=strategy)
        assert cfg.null_strategy == strategy

    @pytest.mark.parametrize("threshold", [0.0, -0.5, -1.0])
    def test_pca_variance_threshold_too_low(self, threshold: float) -> None:
        with pytest.raises(ValueError, match="pca_variance_threshold"):
            PipelineConfig(pca_variance_threshold=threshold)

    def test_pca_variance_threshold_above_one(self) -> None:
        with pytest.raises(ValueError, match="pca_variance_threshold"):
            PipelineConfig(pca_variance_threshold=1.01)

    @pytest.mark.parametrize("threshold", [0.01, 0.5, 0.99, 1.0])
    def test_valid_pca_variance_threshold(self, threshold: float) -> None:
        cfg = PipelineConfig(pca_variance_threshold=threshold)
        assert cfg.pca_variance_threshold == threshold

    def test_overlap_ge_window_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap_seconds"):
            PipelineConfig(
                window=WindowConfig(window_seconds=10.0, overlap_seconds=10.0)
            )

    def test_overlap_gt_window_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap_seconds"):
            PipelineConfig(
                window=WindowConfig(window_seconds=5.0, overlap_seconds=10.0)
            )

    def test_overlap_lt_window_valid(self) -> None:
        cfg = PipelineConfig(
            window=WindowConfig(window_seconds=10.0, overlap_seconds=3.0)
        )
        assert cfg.window.overlap_seconds < cfg.window.window_seconds
