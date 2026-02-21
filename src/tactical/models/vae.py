"""Variational Autoencoder backend for tactical state discovery.

Provides a beta-VAE (:class:`TacticalVAEModule`) for learning a
continuous latent representation of tactical feature vectors, and a
:class:`VAEModel` wrapper that satisfies the
:class:`~tactical.models.base.TacticalModel` protocol by clustering
the latent space with k-means.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from sklearn.cluster import KMeans  # type: ignore[import-untyped]
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from tactical.exceptions import ModelFitError

if TYPE_CHECKING:
    from pathlib import Path

    from tactical.config import VAEConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Loss function
# ------------------------------------------------------------------


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute beta-VAE loss.

    The total loss is the sum of a mean-squared-error reconstruction
    term and a KL-divergence regularisation term weighted by *beta*.

    Args:
        recon_x: Reconstructed input, shape ``(batch, input_dim)``.
        x: Original input, shape ``(batch, input_dim)``.
        mu: Latent mean, shape ``(batch, latent_dim)``.
        logvar: Latent log-variance, shape ``(batch, latent_dim)``.
        beta: Weight applied to the KL-divergence term.

    Returns:
        Tuple of ``(total_loss, reconstruction_loss, kl_divergence)``.
    """
    recon = F.mse_loss(recon_x, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon + beta * kl
    return total, recon, kl


# ------------------------------------------------------------------
# PyTorch module
# ------------------------------------------------------------------


class TacticalVAEModule(nn.Module):
    """Beta-VAE for tactical latent space discovery.

    Architecture::

        Encoder: Input -> [Linear+BN+LeakyReLU+Dropout] x L -> mu, logvar
        Decoder: z     -> [Linear+BN+LeakyReLU+Dropout] x L -> Output

    where *L* is ``len(hidden_dims)``.

    Args:
        input_dim: Number of input features.
        latent_dim: Dimensionality of the latent space.
        hidden_dims: Sizes of hidden layers in the encoder (reversed
            for the decoder).
        dropout: Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # -- Encoder ---------------------------------------------------
        encoder_layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout(p=dropout),
                ]
            )
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent projections
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # -- Decoder ---------------------------------------------------
        decoder_layers: list[nn.Module] = []
        reversed_dims = list(reversed(hidden_dims))
        prev_dim = latent_dim
        for h_dim in reversed_dims:
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout(p=dropout),
                ]
            )
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.

        Returns:
            Tuple of ``(mu, logvar)``, each of shape
            ``(batch, latent_dim)``.
        """
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample latent code via the reparameterisation trick.

        ``z = mu + std * epsilon`` where ``epsilon ~ N(0, I)``.

        Args:
            mu: Latent mean, shape ``(batch, latent_dim)``.
            logvar: Latent log-variance, shape ``(batch, latent_dim)``.

        Returns:
            Sampled latent code of shape ``(batch, latent_dim)``.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent code to reconstructed input.

        Args:
            z: Latent code of shape ``(batch, latent_dim)``.

        Returns:
            Reconstruction of shape ``(batch, input_dim)``.
        """
        recon: torch.Tensor = self.decoder(z)
        return recon

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode, reparameterise, decode.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.

        Returns:
            Tuple of ``(reconstruction, mu, logvar)``.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ------------------------------------------------------------------
# VAEModel wrapper (TacticalModel protocol)
# ------------------------------------------------------------------


class VAEModel:
    """VAE wrapper for tactical state discovery.

    Trains a beta-VAE, extracts latent codes (mu vectors), and
    provides discrete state labels by running k-means on the latent
    space.  Satisfies the :class:`~tactical.models.base.TacticalModel`
    protocol.

    Args:
        config: VAE configuration dataclass.
    """

    def __init__(self, config: VAEConfig) -> None:
        self._config = config
        self._module: TacticalVAEModule | None = None
        self._kmeans: KMeans | None = None
        self._n_clusters: int = 6
        self._training_losses: list[float] = []
        self._device = torch.device("cpu")

    # ------------------------------------------------------------------
    # TacticalModel interface
    # ------------------------------------------------------------------

    def fit(
        self,
        features: np.ndarray,
        sequence_lengths: list[int] | None = None,
    ) -> None:
        """Train the VAE on feature data.

        Uses Adam optimiser with :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau`
        scheduler.  After training, k-means is fitted on the latent
        codes to enable discrete state prediction.

        ``sequence_lengths`` is accepted for protocol compatibility
        but ignored because the VAE treats samples independently.

        Args:
            features: Array of shape ``(n_samples, n_features)``.
            sequence_lengths: Ignored.

        Raises:
            ModelFitError: If training fails.
        """
        cfg = self._config
        torch.manual_seed(cfg.random_state)
        np.random.seed(cfg.random_state)  # noqa: NPY002

        input_dim = features.shape[1]
        module = TacticalVAEModule(
            input_dim=input_dim,
            latent_dim=cfg.latent_dim,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
        ).to(self._device)

        try:
            self._train(module, features, cfg)
        except Exception as exc:
            msg = f"VAE training failed: {exc}"
            raise ModelFitError(msg) from exc

        self._module = module

        # Fit k-means on latent codes
        latent = self._encode_array(features)
        km = KMeans(
            n_clusters=self._n_clusters,
            random_state=cfg.random_state,
            n_init=10,
        )
        km.fit(latent)
        self._kmeans = km

    def predict(
        self,
        features: np.ndarray,
        sequence_lengths: list[int] | None = None,
    ) -> np.ndarray:
        """Predict discrete states by running k-means on latent codes.

        Args:
            features: Array of shape ``(n_samples, n_features)``.
            sequence_lengths: Ignored.

        Returns:
            Array of shape ``(n_samples,)`` with integer state labels.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        self._check_fitted()
        assert self._kmeans is not None
        latent = self.encode(features)
        labels = np.asarray(self._kmeans.predict(latent), dtype=np.intp)
        return labels

    def predict_proba(
        self,
        features: np.ndarray,
        sequence_lengths: list[int] | None = None,
    ) -> np.ndarray:
        """Approximate state probabilities via softmax of negative distances.

        Computes the Euclidean distance from each latent code to every
        k-means centroid, then applies ``softmax(-distances)`` to
        obtain pseudo-probabilities.

        Args:
            features: Array of shape ``(n_samples, n_features)``.
            sequence_lengths: Ignored.

        Returns:
            Array of shape ``(n_samples, n_states)`` where each row
            sums to 1.0.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        self._check_fitted()
        assert self._kmeans is not None
        latent = self.encode(features)
        centroids: np.ndarray = self._kmeans.cluster_centers_

        # (n_samples, 1, latent) - (1, k, latent) -> (n_samples, k)
        diff = latent[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)

        # Softmax of negative distances
        neg_dist = -distances
        neg_dist -= neg_dist.max(axis=1, keepdims=True)  # stability
        exp_neg = np.exp(neg_dist)
        proba: np.ndarray = exp_neg / exp_neg.sum(axis=1, keepdims=True)
        return proba

    @property
    def n_states(self) -> int:
        """Number of discovered states (k-means clusters).

        Raises:
            RuntimeError: If model has not been fitted.
        """
        self._check_fitted()
        return self._n_clusters

    @property
    def training_losses(self) -> list[float]:
        """Per-epoch mean training losses."""
        return list(self._training_losses)

    def set_n_clusters(self, k: int) -> None:
        """Set the number of k-means clusters for state assignment.

        Must be called **before** :meth:`fit` to take effect.

        Args:
            k: Number of clusters.
        """
        self._n_clusters = k

    def encode(self, features: np.ndarray) -> np.ndarray:
        """Extract latent codes (mu vectors) for the input.

        Processes data in batches to limit memory usage.

        Args:
            features: Array of shape ``(n_samples, n_features)``.

        Returns:
            Array of shape ``(n_samples, latent_dim)``.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        self._check_fitted()
        return self._encode_array(features)

    def _encode_array(self, features: np.ndarray) -> np.ndarray:
        """Extract latent codes without the fitted guard.

        Called internally by :meth:`fit` (before k-means is ready)
        and by :meth:`encode` (after the guard passes).

        Args:
            features: Array of shape ``(n_samples, n_features)``.

        Returns:
            Array of shape ``(n_samples, latent_dim)``.
        """
        assert self._module is not None

        self._module.eval()
        batch_size = self._config.batch_size
        tensor = torch.as_tensor(features, dtype=torch.float32, device=self._device)
        dataset = TensorDataset(tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        parts: list[np.ndarray] = []
        with torch.no_grad():
            for (batch,) in loader:
                mu, _ = self._module.encode(batch)
                parts.append(mu.cpu().numpy())

        return np.concatenate(parts, axis=0)

    def save(self, path: Path) -> None:
        """Persist model to disk.

        Saves the VAE state dict, configuration, k-means centroids,
        cluster count, and training losses in a single file via
        :func:`torch.save`.

        Args:
            path: Destination file path.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        self._check_fitted()
        assert self._module is not None
        assert self._kmeans is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self._module.state_dict(),
            "input_dim": self._module.input_dim,
            "config_latent_dim": self._config.latent_dim,
            "config_hidden_dims": self._config.hidden_dims,
            "config_beta": self._config.beta,
            "config_learning_rate": self._config.learning_rate,
            "config_batch_size": self._config.batch_size,
            "config_n_epochs": self._config.n_epochs,
            "config_dropout": self._config.dropout,
            "config_random_state": self._config.random_state,
            "n_clusters": self._n_clusters,
            "kmeans_centers": self._kmeans.cluster_centers_,
            "kmeans_labels": self._kmeans.labels_,
            "training_losses": self._training_losses,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path) -> VAEModel:
        """Load a fitted model from disk.

        Args:
            path: Path to a previously saved model.

        Returns:
            The deserialized :class:`VAEModel`.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        from tactical.config import VAEConfig

        raw = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        assert isinstance(raw, dict)
        payload: dict[str, object] = raw

        def _int(key: str) -> int:
            v = payload[key]
            assert isinstance(v, (int, float))
            return int(v)

        def _float(key: str) -> float:
            v = payload[key]
            assert isinstance(v, (int, float))
            return float(v)

        raw_dims = payload["config_hidden_dims"]
        assert isinstance(raw_dims, (list, tuple))
        config = VAEConfig(
            latent_dim=_int("config_latent_dim"),
            hidden_dims=tuple(int(x) for x in raw_dims),
            beta=_float("config_beta"),
            learning_rate=_float("config_learning_rate"),
            batch_size=_int("config_batch_size"),
            n_epochs=_int("config_n_epochs"),
            dropout=_float("config_dropout"),
            random_state=_int("config_random_state"),
        )

        model = cls(config)
        input_dim = _int("input_dim")
        module = TacticalVAEModule(
            input_dim=input_dim,
            latent_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        )
        state_dict = payload["state_dict"]
        assert isinstance(state_dict, dict)
        module.load_state_dict(state_dict)
        module.eval()

        model._module = module
        model._n_clusters = _int("n_clusters")

        raw_losses = payload["training_losses"]
        assert isinstance(raw_losses, list)
        model._training_losses = [float(v) for v in raw_losses]

        # Reconstruct KMeans from saved centroids
        centers = np.asarray(payload["kmeans_centers"])
        km = KMeans(
            n_clusters=model._n_clusters,
            random_state=config.random_state,
            n_init=10,
        )
        # Warm-start KMeans with saved centroids so predict() works
        km.cluster_centers_ = centers
        km._n_threads = 1  # noqa: SLF001
        km.n_features_in_ = centers.shape[1]
        km.labels_ = np.asarray(payload["kmeans_labels"])
        model._kmeans = km

        return model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train(
        self,
        module: TacticalVAEModule,
        features: np.ndarray,
        cfg: VAEConfig,
    ) -> None:
        """Run the VAE training loop.

        Args:
            module: The PyTorch VAE module to train.
            features: Training data array.
            cfg: VAE configuration.
        """
        tensor = torch.as_tensor(features, dtype=torch.float32, device=self._device)
        dataset = TensorDataset(tensor)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=False,
        )

        optimiser = torch.optim.Adam(module.parameters(), lr=cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=10
        )

        self._training_losses = []
        module.train()

        for _epoch in tqdm(range(cfg.n_epochs), desc="VAE training", leave=False):
            epoch_loss = 0.0
            n_batches = 0
            for (batch,) in loader:
                recon, mu, logvar = module(batch)
                total, _recon_l, _kl_l = vae_loss(
                    recon, batch, mu, logvar, beta=cfg.beta
                )
                optimiser.zero_grad()
                total.backward()  # type: ignore[no-untyped-call]
                optimiser.step()
                epoch_loss += total.item()
                n_batches += 1

            mean_loss = epoch_loss / max(n_batches, 1)
            self._training_losses.append(mean_loss)
            scheduler.step(mean_loss)

        module.eval()

    def _check_fitted(self) -> None:
        """Raise :class:`RuntimeError` if the model is not fitted."""
        if self._module is None or self._kmeans is None:
            msg = "Model has not been fitted. Call fit() before predict/save."
            raise RuntimeError(msg)
