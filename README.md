# a-tad

A modular analytics engine for discovering latent tactical states in football match data.

## Installation

```bash
# Clone the repository
git clone https://github.com/Enhso/a-tad.git
cd a-tad

# Install dependencies with uv
uv sync

# Copy environment config
cp .env.example .env
```

## Development

```bash
# Run tests
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .

# Type checking
uv run mypy src/
```

## Project Structure

```
src/tactical/          # Core package
  adapters/            # Data abstraction layer (StatsBomb, etc.)
  segmentation/        # Time-window and possession-based segmentation
  features/            # Feature extraction engine (tiered)
  models/              # Clustering & sequence models (GMM, HMM, VAE)
  annotation/          # State profiling and heuristic labeling
  analysis/            # Tactical fingerprints, transitions, narratives
  io/                  # Export and reporting utilities
tests/                 # Test suite (mirrors src structure)
notebooks/             # Exploratory and analysis notebooks
```

## License

MIT