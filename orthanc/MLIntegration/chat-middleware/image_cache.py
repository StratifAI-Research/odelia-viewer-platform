"""
Per-series image cache with LRU eviction.

Entries are keyed by series *and* by the recipe that produced them (see
`make_cache_key`). Keying on the series alone -- which this cache originally did
-- silently answers the second message in a conversation with the first
message's slices the moment the two ask for different ones.
"""

import hashlib
import logging
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime


def _utcnow() -> datetime:
    return datetime.now(UTC)


logger = logging.getLogger(__name__)


@dataclass
class CachedSeries:
    """Cached preprocessed images for one series under one recipe"""

    series_uid: str
    base64_images: list[str]  # Central slices as base64 PNG
    created_at: datetime = field(default_factory=_utcnow)
    last_accessed: datetime = field(default_factory=_utcnow)


def make_cache_key(series_uid: str, recipe_parts: Iterable[object]) -> str:
    """
    Cache key for one series preprocessed under one recipe.

    The recipe belongs in the key because it decides which pixels the entry
    holds: two messages can ask about the same series and mean different slices.
    Hashed rather than concatenated so a key stays short and loggable however
    many SOPInstanceUIDs a selection names.

    Args:
        series_uid: The SeriesInstanceUID
        recipe_parts: Everything that affects the produced images, in a stable
            order. Callers build this (see `preprocessing.recipe_signature`) so
            that adding a new preprocessing input has exactly one place to change.

    Returns:
        A key of the form `<series_uid>#<digest>`
    """
    canonical = "|".join(str(part) for part in recipe_parts)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{series_uid}#{digest}"


class ImageCache:
    """
    LRU cache for preprocessed DICOM series images.
    Global across all sessions to avoid redundant preprocessing.

    Keys come from `make_cache_key` -- series plus recipe, not series alone.
    """

    def __init__(self, max_entries: int = 100) -> None:
        """
        Initialize the image cache.

        Args:
            max_entries: Maximum number of series to cache before eviction
        """
        # OrderedDict maintains insertion order for LRU
        self._cache: OrderedDict[str, CachedSeries] = OrderedDict()
        self.max_entries = max_entries

    def get(self, key: str) -> CachedSeries | None:
        """
        Get a cached entry by key.
        Updates last_accessed and moves to end of LRU queue.

        Args:
            key: A key from `make_cache_key`

        Returns:
            CachedSeries if found, None otherwise
        """
        if key not in self._cache:
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)

        # Update last accessed time
        cached = self._cache[key]
        cached.last_accessed = _utcnow()

        logger.debug(f"Cache hit for {key}")
        return cached

    def put(self, key: str, data: CachedSeries) -> None:
        """
        Store an entry in the cache.
        Evicts LRU entries if cache is full.

        Args:
            key: A key from `make_cache_key`
            data: CachedSeries to store
        """
        # If already exists, update and move to end
        if key in self._cache:
            self._cache[key] = data
            self._cache.move_to_end(key)
            logger.debug(f"Updated cache entry {key}")
            return

        # Evict if at capacity
        while len(self._cache) >= self.max_entries:
            self._evict_lru()

        # Add new entry
        self._cache[key] = data
        logger.info(f"Cached {key} ({len(data.base64_images)} images)")

    def has(self, key: str) -> bool:
        """
        Check if an entry is in the cache.

        Args:
            key: A key from `make_cache_key`

        Returns:
            True if cached, False otherwise
        """
        return key in self._cache

    def _evict_lru(self) -> str | None:
        """
        Evict the least recently used entry.

        Returns:
            The evicted key, or None if cache was empty
        """
        if not self._cache:
            return None

        # popitem(last=False) removes the first (oldest) item
        key, _ = self._cache.popitem(last=False)
        logger.info(f"Evicted LRU cache entry: {key}")
        return key

    def clear(self) -> int:
        """
        Clear all entries from the cache.

        Returns:
            Number of entries that were cleared
        """
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared {count} entries from cache")
        return count

    def size(self) -> int:
        """
        Get the current number of cached entries.

        Returns:
            Number of entries in cache
        """
        return len(self._cache)

    def stats(self) -> dict:
        """
        Get cache statistics for debugging.

        Returns:
            Dict with cache stats
        """
        return {
            "size": len(self._cache),
            "max_entries": self.max_entries,
            # Both: `keys` identifies entries (a series can hold several, one per
            # recipe), `series_uids` answers the question the debug UI asks --
            # which series are cached at all.
            "keys": list(self._cache.keys()),
            "series_uids": sorted({entry.series_uid for entry in self._cache.values()}),
        }


# Global image cache instance
_image_cache: ImageCache | None = None


def get_image_cache(max_entries: int = 100) -> ImageCache:
    """Get the global image cache singleton"""
    global _image_cache
    if _image_cache is None:
        _image_cache = ImageCache(max_entries=max_entries)
    elif _image_cache.max_entries != max_entries:
        logger.warning(
            f"ImageCache already initialized with max_entries={_image_cache.max_entries}, "
            f"ignoring requested max_entries={max_entries}"
        )
    return _image_cache


def reset_image_cache(max_entries: int = 100) -> ImageCache:
    """Reset the image cache (useful for testing)"""
    global _image_cache
    _image_cache = ImageCache(max_entries=max_entries)
    return _image_cache
