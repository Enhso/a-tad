"""Configuration dataclasses for the Tactical State Discovery Engine.

All configuration containers are frozen (immutable) and slotted for
memory efficiency and safety. Each dataclass provides sensible defaults
so that a zero-argument ``PipelineConfig()`` is always valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """Configuration for time-based segmentation windows.

    Attributes:
        window_seconds: Duration of each window in seconds.
        overlap_seconds: Overlap between consecutive windows in seconds.
        min_events: Minimum number of events required in a window.
    """

    window_seconds: float = 15.0
    overlap_seconds: float = 5.0
    min_events: int = 3


@dataclass(frozen=True, slots=True)
class PossessionConfig:
    """Configuration for possession-based segmentation.

    Attributes:
        min_events: Minimum number of events required in a possession
            sequence to be retained.
    """

    min_events: int = 3


@dataclass(frozen=True, slots=True)
class GMMConfig:
    """Configuration for the Gaussian Mixture Model backend.

    Attributes:
        k_min: Minimum number of clusters to evaluate.
        k_max: Maximum number of clusters to evaluate.
        covariance_type: Type of covariance parameters
            (``"full"``, ``"tied"``, ``"diag"``, ``"spherical"``).
        n_init: Number of random initializations to attempt.
        random_state: Seed for reproducibility.
        use_bayesian: Whether to use a Bayesian GMM (DPGMM).
    """

    k_min: int = 3
    k_max: int = 12
    covariance_type: str = "full"
    n_init: int = 20
    random_state: int = 42
    use_bayesian: bool = False


@dataclass(frozen=True, slots=True)
class HMMConfig:
    """Configuration for the Hidden Markov Model backend.

    Attributes:
        n_states: Number of hidden states.
        covariance_type: Type of covariance parameters
            (``"full"``, ``"tied"``, ``"diag"``, ``"spherical"``).
        n_iter: Maximum number of EM iterations.
        random_state: Seed for reproducibility.
        init_from_gmm: Whether to initialize HMM parameters from a
            fitted GMM.
    """

    n_states: int = 6
    covariance_type: str = "full"
    n_iter: int = 100
    random_state: int = 42
    init_from_gmm: bool = True


@dataclass(frozen=True, slots=True)
class VAEConfig:
    """Configuration for the Variational Autoencoder backend.

    Attributes:
        latent_dim: Dimensionality of the latent space.
        hidden_dims: Sizes of hidden layers in the encoder/decoder.
        beta: Weight of the KL-divergence term (beta-VAE).
        learning_rate: Optimizer learning rate.
        batch_size: Training mini-batch size.
        n_epochs: Number of training epochs.
        dropout: Dropout probability applied to hidden layers.
        random_state: Seed for reproducibility.
    """

    latent_dim: int = 8
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    beta: float = 4.0
    learning_rate: float = 1e-3
    batch_size: int = 512
    n_epochs: int = 150
    dropout: float = 0.2
    random_state: int = 42


@dataclass(frozen=True, slots=True)
class ChangepointConfig:
    """Configuration for changepoint detection.

    Attributes:
        method: Detection algorithm (``"pelt"``, ``"binseg"``, etc.).
        model: Cost model (``"rbf"``, ``"l2"``, ``"linear"``, etc.).
        penalty: Penalty value controlling the number of changepoints.
        min_segment_size: Minimum number of observations between
            changepoints.
    """

    method: str = "pelt"
    model: str = "rbf"
    penalty: float = 10.0
    min_segment_size: int = 5


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Master configuration for the full tactical analysis pipeline.

    Aggregates all sub-configurations and global settings. Validation
    is performed in ``__post_init__`` to ensure invariants hold.

    Attributes:
        cache_dir: Directory for caching raw data downloads.
        output_dir: Directory for analysis outputs.
        model_dir: Directory for persisted model artifacts.
        max_feature_tier: Highest feature tier to extract (1, 2, or 3).
        window: Time-based windowing configuration.
        possession: Possession-based segmentation configuration.
        gmm: Gaussian Mixture Model configuration.
        hmm: Hidden Markov Model configuration.
        vae: Variational Autoencoder configuration.
        changepoint: Changepoint detection configuration.
        pca_variance_threshold: Cumulative variance ratio retained by
            PCA dimensionality reduction.
        null_strategy: Strategy for handling null feature values
            (``"drop_rows"`` or ``"impute_median"``).
        match_phases: Tuple of ``(start_minute, end_minute)`` pairs
            defining canonical match phases for aggregation.

    Raises:
        ValueError: If any configuration invariant is violated.
    """

    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))
    output_dir: Path = field(default_factory=lambda: Path("data/output"))
    model_dir: Path = field(default_factory=lambda: Path("data/models"))
    max_feature_tier: int = 2
    window: WindowConfig = field(default_factory=WindowConfig)
    possession: PossessionConfig = field(default_factory=PossessionConfig)
    gmm: GMMConfig = field(default_factory=GMMConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    changepoint: ChangepointConfig = field(default_factory=ChangepointConfig)
    pca_variance_threshold: float = 0.95
    null_strategy: str = "drop_rows"
    match_phases: tuple[tuple[int, int], ...] = (
        (0, 15),
        (15, 45),
        (45, 60),
        (60, 75),
        (75, 90),
    )

    def __post_init__(self) -> None:
        """Validate configuration invariants after initialization."""
        if self.max_feature_tier not in (1, 2, 3):
            msg = f"max_feature_tier must be 1, 2, or 3, got {self.max_feature_tier}"
            raise ValueError(msg)

        valid_strategies = ("drop_rows", "impute_median")
        if self.null_strategy not in valid_strategies:
            msg = (
                f"null_strategy must be one of {valid_strategies}, "
                f"got {self.null_strategy!r}"
            )
            raise ValueError(msg)

        if self.pca_variance_threshold <= 0.0 or self.pca_variance_threshold > 1.0:
            msg = (
                f"pca_variance_threshold must be in (0.0, 1.0], "
                f"got {self.pca_variance_threshold}"
            )
            raise ValueError(msg)

        if self.window.overlap_seconds >= self.window.window_seconds:
            msg = (
                f"window.overlap_seconds ({self.window.overlap_seconds}) "
                f"must be less than window.window_seconds "
                f"({self.window.window_seconds})"
            )
            raise ValueError(msg)
