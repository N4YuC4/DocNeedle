"""In-memory cache for parsed document chunks to eliminate redundant OCR runs.

Privacy Guarantee:
All cached data is kept strictly in-process in RAM and is never persisted to disk or databases.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict

from docneedle.ingest.models import IngestResult

logger = logging.getLogger(__name__)


class DocumentIngestionCache:
    """Thread-safe in-memory LRU cache for ingested document results."""

    def __init__(self, max_size: int = 64) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[str, IngestResult] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, content_hash: str) -> IngestResult | None:
        """Retrieve cached IngestResult by file content hash if available."""
        with self._lock:
            if content_hash in self._cache:
                self._cache.move_to_end(content_hash)
                logger.debug(f"Ingestion cache hit for hash: {content_hash[:12]}")
                return self._cache[content_hash]
            return None

    def put(self, content_hash: str, result: IngestResult) -> None:
        """Store an IngestResult in the in-memory cache."""
        with self._lock:
            if content_hash in self._cache:
                self._cache.move_to_end(content_hash)
            self._cache[content_hash] = result
            if len(self._cache) > self._max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug(f"Evicted old ingestion cache entry: {evicted_key[:12]}")

    def remove(self, content_hash: str) -> bool:
        """Remove a specific entry from the cache."""
        with self._lock:
            if content_hash in self._cache:
                del self._cache[content_hash]
                return True
            return False

    def clear(self) -> None:
        """Wipe all cached document chunks from RAM."""
        with self._lock:
            self._cache.clear()
            logger.debug("Ingestion cache cleared.")

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def contains(self, content_hash: str) -> bool:
        with self._lock:
            return content_hash in self._cache


# Process-wide in-memory ingestion cache
global_ingestion_cache = DocumentIngestionCache(max_size=64)
