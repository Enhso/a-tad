"""Tests for :mod:`tactical.models.vae`.

Uses synthetic data with 200 samples and 20 features drawn from 3
well-separated clusters to exercise the VAE module, loss function,
and model wrapper (fit, encode, predict, predict_proba, save/load).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from tactical.config import VAEConfig
from tactical.models.vae import TacticalVAEModule, VAEModel, vae_loss

if TYPE_CHECKING:
    from pathlib import Path

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_N_SAMPLES = 200
_N_FEATURES = 20
_LATENT_DIM = 4
_HIDDEN_DIMS = (32, 16)
_N_STATES = 3
_N_EPOCHS = 40
_BATCH_SIZE = 64

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def synthetic_features() -> np.ndarray:
    """200 samples x 20 features from 3 well-separated clusters."""
    rng = np.random.default_rng(seed=42)
    centers = rng.uniform(-10, 10, size=(_N_STATES, _N_FEATURES))
    # Scale centers apart
    centers *= 5.0
    parts: list[np.ndarray] = []
    per_cluster = _N_SAMPLES // _N_STATES
    for i in range(_N_STATES):
        n = (
            per_cluster
            if i < _N_STATES - 1
            else _N_SAMPLES - per_cluster * (_N_STATES - 1)
        )
        pts = rng.normal(loc=centers[i], scale=0.5, size=(n, _N_FEATURES))
        parts.append(pts)
    return np.vstack(parts)


@pytest.fixture()
def vae_config() -> VAEConfig:
    """Small VAEConfig for fast testing."""
    return VAEConfig(
        latent_dim=_LATENT_DIM,
        hidden_dims=_HIDDEN_DIMS,
        beta=1.0,
        learning_rate=1e-3,
        batch_size=_BATCH_SIZE,
        n_epochs=_N_EPOCHS,
        dropout=0.1,
        random_state=42,
    )


@pytest.fixture()
def fitted_vae(
    synthetic_features: np.ndarray,
    vae_config: VAEConfig,
) -> VAEModel:
    """A VAEModel fitted on the synthetic data with K=3."""
    model = VAEModel(vae_config)
    model.set_n_clusters(_N_STATES)
    model.fit(synthetic_features)
    return model


# ------------------------------------------------------------------
# Tests: TacticalVAEModule
# ------------------------------------------------------------------


class TestTacticalVAEModule:
    """Unit tests for :class:`TacticalVAEModule`."""

    def test_vae_module_forward_shape(self) -> None:
        """Forward pass output shape matches input shape."""
        module = TacticalVAEModule(
            input_dim=_N_FEATURES,
            latent_dim=_LATENT_DIM,
            hidden_dims=_HIDDEN_DIMS,
            dropout=0.1,
        )
        module.eval()
        x = torch.randn(16, _N_FEATURES)
        recon, mu, logvar = module(x)

        assert recon.shape == x.shape
        assert mu.shape == (16, _LATENT_DIM)
        assert logvar.shape == (16, _LATENT_DIM)

    def test_vae_module_latent_shape(self) -> None:
        """Encode returns mu and logvar of shape (batch, latent_dim)."""
        module = TacticalVAEModule(
            input_dim=_N_FEATURES,
            latent_dim=_LATENT_DIM,
            hidden_dims=_HIDDEN_DIMS,
        )
        module.eval()
        x = torch.randn(8, _N_FEATURES)
        mu, logvar = module.encode(x)

        assert mu.shape == (8, _LATENT_DIM)
        assert logvar.shape == (8, _LATENT_DIM)

    def test_vae_module_decode_shape(self) -> None:
        """Decode maps latent code back to input dimension."""
        module = TacticalVAEModule(
            input_dim=_N_FEATURES,
            latent_dim=_LATENT_DIM,
            hidden_dims=_HIDDEN_DIMS,
        )
        module.eval()
        z = torch.randn(8, _LATENT_DIM)
        recon = module.decode(z)

        assert recon.shape == (8, _N_FEATURES)

    def test_vae_module_reparameterize_shape(self) -> None:
        """Reparameterize returns tensor with same shape as mu."""
        module = TacticalVAEModule(
            input_dim=_N_FEATURES,
            latent_dim=_LATENT_DIM,
            hidden_dims=_HIDDEN_DIMS,
        )
        mu = torch.zeros(8, _LATENT_DIM)
        logvar = torch.zeros(8, _LATENT_DIM)
        z = module.reparameterize(mu, logvar)

        assert z.shape == mu.shape


# ------------------------------------------------------------------
# Tests: vae_loss
# ------------------------------------------------------------------


class TestVAELoss:
    """Unit tests for :func:`vae_loss`."""

    def test_vae_loss_components(self) -> None:
        """All three loss components are positive tensors."""
        x = torch.randn(16, _N_FEATURES)
        recon_x = torch.randn(16, _N_FEATURES)
        mu = torch.randn(16, _LATENT_DIM)
        logvar = torch.randn(16, _LATENT_DIM)

        total, recon, kl = vae_loss(recon_x, x, mu, logvar, beta=4.0)

        assert isinstance(total, torch.Tensor)
        assert isinstance(recon, torch.Tensor)
        assert isinstance(kl, torch.Tensor)
        assert total.item() > 0.0
        assert recon.item() > 0.0
        assert kl.item() > 0.0

    def test_vae_loss_perfect_reconstruction(self) -> None:
        """Reconstruction loss is zero when recon_x equals x."""
        x = torch.randn(16, _N_FEATURES)
        mu = torch.zeros(16, _LATENT_DIM)
        logvar = torch.zeros(16, _LATENT_DIM)

        total, recon, kl = vae_loss(x, x, mu, logvar, beta=1.0)

        assert recon.item() == pytest.approx(0.0, abs=1e-6)
        # KL with mu=0, logvar=0: -0.5 * mean(1 + 0 - 0 - 1) = 0
        assert kl.item() == pytest.approx(0.0, abs=1e-6)

    def test_vae_loss_beta_scaling(self) -> None:
        """Higher beta increases total loss when KL > 0."""
        x = torch.randn(16, _N_FEATURES)
        recon_x = x.clone()
        mu = torch.ones(16, _LATENT_DIM)  # non-zero -> KL > 0
        logvar = torch.zeros(16, _LATENT_DIM)

        total_b1, _, _ = vae_loss(recon_x, x, mu, logvar, beta=1.0)
        total_b4, _, _ = vae_loss(recon_x, x, mu, logvar, beta=4.0)

        assert total_b4.item() > total_b1.item()


# ------------------------------------------------------------------
# Tests: VAEModel
# ------------------------------------------------------------------


class TestVAEModel:
    """Unit tests for :class:`VAEModel`."""

    def test_vae_model_fit(
        self,
        synthetic_features: np.ndarray,
        vae_config: VAEConfig,
    ) -> None:
        """Fit completes without error and records training losses."""
        model = VAEModel(vae_config)
        model.set_n_clusters(_N_STATES)
        model.fit(synthetic_features)

        assert len(model.training_losses) == _N_EPOCHS
        assert all(isinstance(v, float) for v in model.training_losses)

    def test_vae_model_encode_shape(
        self,
        synthetic_features: np.ndarray,
        fitted_vae: VAEModel,
    ) -> None:
        """Latent codes shape is (n_samples, latent_dim)."""
        latent = fitted_vae.encode(synthetic_features)

        assert latent.shape == (_N_SAMPLES, _LATENT_DIM)

    def test_vae_model_predict_shape(
        self,
        synthetic_features: np.ndarray,
        fitted_vae: VAEModel,
    ) -> None:
        """predict returns (n_samples,) array of integer labels."""
        labels = fitted_vae.predict(synthetic_features)

        assert labels.shape == (_N_SAMPLES,)
        assert labels.dtype in (np.int32, np.int64)

    def test_vae_model_predict_proba_shape(
        self,
        synthetic_features: np.ndarray,
        fitted_vae: VAEModel,
    ) -> None:
        """predict_proba returns (n_samples, n_states) array."""
        proba = fitted_vae.predict_proba(synthetic_features)

        assert proba.shape == (_N_SAMPLES, _N_STATES)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_vae_model_n_states(
        self,
        fitted_vae: VAEModel,
    ) -> None:
        """n_states returns the configured cluster count."""
        assert fitted_vae.n_states == _N_STATES

    def test_vae_model_save_load(
        self,
        synthetic_features: np.ndarray,
        fitted_vae: VAEModel,
        tmp_path: Path,
    ) -> None:
        """Roundtrip save/load produces identical latent codes."""
        expected = fitted_vae.encode(synthetic_features)

        save_path = tmp_path / "vae_model.pt"
        fitted_vae.save(save_path)

        loaded = VAEModel.load(save_path)
        result = loaded.encode(synthetic_features)

        np.testing.assert_allclose(result, expected, atol=1e-5)
        assert loaded.n_states == fitted_vae.n_states
        assert len(loaded.training_losses) == len(fitted_vae.training_losses)

    def test_vae_training_loss_decreases(
        self,
        fitted_vae: VAEModel,
    ) -> None:
        """Last training loss is lower than first training loss."""
        losses = fitted_vae.training_losses
        assert len(losses) >= 2
        assert losses[-1] < losses[0]

    def test_vae_predict_before_fit_raises(
        self,
        synthetic_features: np.ndarray,
        vae_config: VAEConfig,
    ) -> None:
        """predict on unfitted model raises RuntimeError."""
        model = VAEModel(vae_config)

        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(synthetic_features)

    def test_vae_encode_before_fit_raises(
        self,
        synthetic_features: np.ndarray,
        vae_config: VAEConfig,
    ) -> None:
        """encode on unfitted model raises RuntimeError."""
        model = VAEModel(vae_config)

        with pytest.raises(RuntimeError, match="not been fitted"):
            model.encode(synthetic_features)
