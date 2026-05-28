"""
cache/retrieval_cache.py
-------------------------
Retrieval Cache Management.
Caches (query_embedding → results) to avoid redundant ChromaDB lookups.

Two layers:
  1. Exact cache   : hash of query string → results
  2. Semantic cache: if a near-identical query was asked before (cosine sim > threshold)
"""
from __future__ import annotations
import hashlib
import json
import time
from dataclasses import dataclass, field
from src.vectordb.vector_store import SearchResult
from src.utils.logger import get_logger

log = get_logger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x*y for x,y in zip(a,b))
    na  = sum(x*x for x in a) ** 0.5
    nb  = sum(x*x for x in b) ** 0.5
    return dot / (na * nb + 1e-9)


@dataclass
class CacheEntry:
    query:        str
    embedding:    list[float]
    results:      list[SearchResult]
    created_at:   float = field(default_factory=time.time)
    hits:         int   = 0


class RetrievalCache:
    def __init__(
        self,
        max_size:          int   = 500,
        ttl_seconds:       int   = 3600,         # 1 hour
        semantic_threshold: float = 0.95,         # cosine sim to consider a hit
    ):
        self.max_size           = max_size
        self.ttl                = ttl_seconds
        self.semantic_threshold = semantic_threshold
        self._exact:    dict[str, CacheEntry] = {}
        self._semantic: list[CacheEntry]      = []

    def _hash(self, query: str) -> str:
        return hashlib.md5(query.strip().lower().encode()).hexdigest()

    def get(self, query: str, embedding: list[float]) -> list[SearchResult] | None:
        now = time.time()

        # 1. Exact match
        key   = self._hash(query)
        entry = self._exact.get(key)
        if entry and (now - entry.created_at) < self.ttl:
            entry.hits += 1
            log.info(f"cache_hit  type=exact  query={query[:40]!r}")
            return entry.results

        # 2. Semantic match
        for entry in self._semantic:
            if (now - entry.created_at) > self.ttl:
                continue
            sim = _cosine(embedding, entry.embedding)
            if sim >= self.semantic_threshold:
                entry.hits += 1
                log.info(f"cache_hit  type=semantic  sim={sim:.3f}  query={query[:40]!r}")
                return entry.results

        return None

    def set(self, query: str, embedding: list[float], results: list[SearchResult]) -> None:
        self._evict_expired()
        if len(self._exact) >= self.max_size:
            # Remove oldest
            oldest = min(self._exact, key=lambda k: self._exact[k].created_at)
            del self._exact[oldest]

        entry = CacheEntry(query=query, embedding=embedding, results=results)
        self._exact[self._hash(query)] = entry
        self._semantic.append(entry)
        log.info(f"cache_set  query={query[:40]!r}  results={len(results)}")

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [k for k, v in self._exact.items() if (now - v.created_at) > self.ttl]
        for k in expired:
            del self._exact[k]
        self._semantic = [e for e in self._semantic if (now - e.created_at) <= self.ttl]

    def clear(self) -> None:
        self._exact.clear()
        self._semantic.clear()

    def stats(self) -> dict:
        total_hits = sum(e.hits for e in self._exact.values())
        return {
            "size":       len(self._exact),
            "total_hits": total_hits,
            "max_size":   self.max_size,
            "ttl_seconds": self.ttl,
        }
