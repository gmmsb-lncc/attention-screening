"""
Pipeline output caching system for DockTkinase.

Simplified, robust caching for intermediate pipeline outputs (embeddings, MSA,
structure predictions) to avoid recomputation on repeated predictions.

Key Features:
  - Content-based hashing with SHA256
  - Thread-safe with automatic cleanup
  - FIFO eviction policy for memory management
  - Size and entry count limits
  - Basic metrics tracking

Recommendation: Call cache.clear() between training runs to avoid stale data.
"""

import hashlib
import logging
import pickle
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""

    key: str
    """Cache key (SHA256 of input)."""

    value: Any
    """Cached value (embeddings, MSA, etc.)."""

    timestamp: float
    """Entry creation time (Unix timestamp)."""

    ttl: Optional[float] = None
    """Time-to-live in seconds. None = no expiration."""

    size_bytes: int = 0
    """Estimated size in bytes."""

    hit_count: int = 0
    """Number of times this entry was accessed."""

@dataclass
class CacheKey:
    """Generate cache keys from inputs with content-based hashing."""

    @staticmethod
    def from_sequence(sequence: str, stage: str = "embedding", **kwargs) -> str:
        """Generate cache key from protein sequence."""
        content = f"{stage}:{sequence}"
        for key in sorted(kwargs.keys()):
            content += f":{key}={kwargs[key]}"
        return hashlib.sha256(content.encode()).hexdigest()

    @staticmethod
    def from_dict(data: Dict[str, Any], stage: str = "config", **kwargs) -> str:
        """Generate cache key from dictionary."""
        import json
        json_str = json.dumps(data, sort_keys=True, default=str)
        content = f"{stage}:{hashlib.sha256(json_str.encode()).hexdigest()}"
        for key in sorted(kwargs.keys()):
            content += f":{key}={kwargs[key]}"
        return hashlib.sha256(content.encode()).hexdigest()


class PipelineCache:
    """
    Simplified, thread-safe cache with FIFO eviction.
    
    Features:
    - O(1) lookup and insertion
    - FIFO eviction when size exceeded
    - Thread-safe (RLock)
    - Hit/miss tracking
    
    Recommendation: Call clear() between training runs to avoid stale data.
    """

    def __init__(self, max_size_mb: int = 256, max_entries: int = 10000):
        """
        Initialize pipeline cache.

        Args:
            max_size_mb: Maximum cache size in MB
            max_entries: Maximum number of entries
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self._cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, timestamp)
        self._lock = threading.RLock()
        self._current_size_bytes = 0
        self._entry_order = []  # Track insertion order for FIFO

        # Metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Tuple[Optional[Any], bool]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Tuple of (value, found) where found indicates hit/miss
        """
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                self._hits += 1
                return value, True

            self._misses += 1
            return None, False

    def set(self, key: str, value: Any) -> bool:
        """
        Store value in cache.

        Args:
            key: Cache key
            value: Value to cache

        Returns:
            True if stored successfully
        """
        with self._lock:
            # Calculate entry size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = 1024  # Default estimate

            # Evict if necessary (FIFO + size limit)
            while (
                (self._current_size_bytes + size_bytes > self.max_size_bytes)
                or (len(self._cache) >= self.max_entries)
            ) and self._entry_order:
                self._evict_oldest()

            # Remove old entry if exists
            if key in self._cache:
                old_size, _ = self._cache[key]
                self._current_size_bytes -= old_size
                if key in self._entry_order:
                    self._entry_order.remove(key)

            # Add new entry
            self._cache[key] = (value, time.time())
            self._entry_order.append(key)
            self._current_size_bytes += size_bytes

            return True

    def _evict_oldest(self) -> None:
        """Evict oldest (FIFO) entry."""
        if self._entry_order:
            key = self._entry_order.pop(0)
            value, _ = self._cache.pop(key, (None, 0))
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = 1024
            self._current_size_bytes -= size_bytes
            self._evictions += 1

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._entry_order.clear()
            self._current_size_bytes = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_accesses = self._hits + self._misses
            hit_rate = (
                self._hits / total_accesses if total_accesses > 0 else 0.0
            )

            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "current_size_mb": self._current_size_bytes / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "num_entries": len(self._cache),
            }


# Global cache instance
_global_cache: Optional[PipelineCache] = None


def get_pipeline_cache(memory_size_mb: int = 256) -> PipelineCache:
    """Get or create global pipeline cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = PipelineCache(memory_size_mb=memory_size_mb)
    return _global_cache


def reset_pipeline_cache() -> None:
    """Reset global cache instance."""
    global _global_cache
    if _global_cache:
        _global_cache.clear()
    _global_cache = None
