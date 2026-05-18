"""
Per-series image cache with LRU eviction
"""
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CachedSeries:
    """Cached preprocessed images for a DICOM series"""
    series_uid: str
    base64_images: List[str]  # Central slices as base64 PNG
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)


class ImageCache:
    """
    LRU cache for preprocessed DICOM series images.
    Global across all sessions to avoid redundant preprocessing.
    """

    def __init__(self, max_entries: int = 100):
        """
        Initialize the image cache.

        Args:
            max_entries: Maximum number of series to cache before eviction
        """
        # OrderedDict maintains insertion order for LRU
        self._cache: OrderedDict[str, CachedSeries] = OrderedDict()
        self.max_entries = max_entries

    def get(self, series_uid: str) -> Optional[CachedSeries]:
        """
        Get cached series by UID.
        Updates last_accessed and moves to end of LRU queue.

        Args:
            series_uid: The SeriesInstanceUID to look up

        Returns:
            CachedSeries if found, None otherwise
        """
        if series_uid not in self._cache:
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(series_uid)

        # Update last accessed time
        cached = self._cache[series_uid]
        cached.last_accessed = datetime.now()

        logger.debug(f"Cache hit for series: {series_uid}")
        return cached

    def put(self, series_uid: str, data: CachedSeries) -> None:
        """
        Store a series in the cache.
        Evicts LRU entries if cache is full.

        Args:
            series_uid: The SeriesInstanceUID
            data: CachedSeries to store
        """
        # If already exists, update and move to end
        if series_uid in self._cache:
            self._cache[series_uid] = data
            self._cache.move_to_end(series_uid)
            logger.debug(f"Updated cache entry for series: {series_uid}")
            return

        # Evict if at capacity
        while len(self._cache) >= self.max_entries:
            self._evict_lru()

        # Add new entry
        self._cache[series_uid] = data
        logger.info(f"Cached series: {series_uid} ({len(data.base64_images)} images)")

    def has(self, series_uid: str) -> bool:
        """
        Check if a series is in the cache.

        Args:
            series_uid: The SeriesInstanceUID to check

        Returns:
            True if cached, False otherwise
        """
        return series_uid in self._cache

    def _evict_lru(self) -> Optional[str]:
        """
        Evict the least recently used entry.

        Returns:
            The evicted series_uid, or None if cache was empty
        """
        if not self._cache:
            return None

        # popitem(last=False) removes the first (oldest) item
        series_uid, _ = self._cache.popitem(last=False)
        logger.info(f"Evicted LRU cache entry: {series_uid}")
        return series_uid

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
            "series_uids": list(self._cache.keys())
        }


# Global image cache instance
_image_cache: Optional[ImageCache] = None


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
