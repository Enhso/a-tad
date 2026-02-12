"""Disk-based cache for raw match data.

Provides a simple pickle-backed cache keyed by match ID. Writes are
atomic (write to a temporary file, then rename) to prevent corruption
if the process is interrupted mid-write.
"""

from __future__ import annotations

import logging
import pickle
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MatchCache:
    """Disk-based cache for raw match data.

    Stores pickled data keyed by match_id in a configurable
    cache directory. Creates the directory if it doesn't exist.

    Attributes:
        _cache_dir: Root directory where cache files are stored.
    """

    __slots__ = ("_cache_dir",)

    def __init__(self, cache_dir: Path) -> None:
        """Initialize the cache and ensure the directory exists.

        Args:
            cache_dir: Directory for storing cached pickle files.
        """
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(self, match_id: str) -> Path:
        """Return the file path for a given match ID.

        Args:
            match_id: Unique identifier of the match.

        Returns:
            Path to the pickle file for this match.
        """
        return self._cache_dir / f"match_{match_id}.pkl"

    def exists(self, match_id: str) -> bool:
        """Check whether a cache entry exists for the given match ID.

        Args:
            match_id: Unique identifier of the match.

        Returns:
            True if a cached file exists, False otherwise.
        """
        return self.cache_path(match_id).is_file()

    def get(self, match_id: str) -> dict[str, Any] | None:
        """Load cached data for a match.

        Args:
            match_id: Unique identifier of the match.

        Returns:
            The cached dictionary, or ``None`` on a cache miss.

        Note:
            ``Any`` is used here because we cache raw provider data
            whose schema varies by provider.
        """
        path = self.cache_path(match_id)
        if not path.is_file():
            logger.debug("Cache miss for match %s", match_id)
            return None

        logger.debug("Cache hit for match %s", match_id)
        with path.open("rb") as f:
            result: dict[str, Any] = pickle.load(f)  # noqa: S301
        return result

    def put(self, match_id: str, data: dict[str, Any]) -> None:
        """Save data to the cache using an atomic write.

        Writes to a temporary file in the same directory and renames
        it to the final path, preventing corruption on interruption.

        Args:
            match_id: Unique identifier of the match.
            data: Raw provider data to cache.
        """
        path = self.cache_path(match_id)
        fd, tmp_path_str = tempfile.mkstemp(dir=self._cache_dir, suffix=".tmp")
        tmp_path = Path(tmp_path_str)
        try:
            with open(fd, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
        logger.debug("Cached data for match %s at %s", match_id, path)
