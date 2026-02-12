"""Tests for the disk-based match cache.

Validates cache miss behavior, put/get round-trips, existence checks,
automatic directory creation, and cache path formatting for
:class:`tactical.adapters.cache.MatchCache`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tactical.adapters.cache import MatchCache

if TYPE_CHECKING:
    from pathlib import Path


class TestCacheMiss:
    """Cache misses must return None without raising."""

    def test_cache_miss_returns_none(self, tmp_path: Path) -> None:
        """Getting a non-existent key returns None."""
        cache = MatchCache(cache_dir=tmp_path / "cache")
        assert cache.get("nonexistent") is None


class TestCachePutAndGet:
    """Round-trip through put then get must preserve data."""

    def test_cache_put_and_get(self, tmp_path: Path) -> None:
        """Data stored with put is retrievable with get."""
        cache = MatchCache(cache_dir=tmp_path / "cache")
        data = {
            "events": [{"id": 1, "type": "pass"}],
            "meta": {"source": "statsbomb"},
            "nested": {"a": [1, 2, 3]},
        }
        cache.put("match-001", data)
        result = cache.get("match-001")
        assert result == data

    def test_overwrite_existing(self, tmp_path: Path) -> None:
        """A second put overwrites the first without error."""
        cache = MatchCache(cache_dir=tmp_path / "cache")
        cache.put("match-001", {"version": 1})
        cache.put("match-001", {"version": 2})
        result = cache.get("match-001")
        assert result == {"version": 2}


class TestCacheExists:
    """Existence checks must reflect actual cache state."""

    def test_cache_exists(self, tmp_path: Path) -> None:
        """exists returns False before put and True after."""
        cache = MatchCache(cache_dir=tmp_path / "cache")
        assert cache.exists("match-001") is False
        cache.put("match-001", {"key": "value"})
        assert cache.exists("match-001") is True


class TestCacheCreatesDirectory:
    """Cache directory must be created automatically."""

    def test_cache_creates_directory(self, tmp_path: Path) -> None:
        """Initializing MatchCache creates the cache directory."""
        nested = tmp_path / "deep" / "nested" / "cache"
        assert not nested.exists()
        MatchCache(cache_dir=nested)
        assert nested.is_dir()


class TestCachePath:
    """cache_path must return a deterministic, well-formed path."""

    def test_cache_path(self, tmp_path: Path) -> None:
        """cache_path returns the expected file path for a match_id."""
        cache = MatchCache(cache_dir=tmp_path / "cache")
        expected = tmp_path / "cache" / "match_match-007.pkl"
        assert cache.cache_path("match-007") == expected
