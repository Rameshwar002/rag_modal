"""
vectordb/vector_store.py
-------------------------
Persistent ChromaDB vector store.
Collections:
  confluence_docs | sharepoint_docs | uploaded_files
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.chunking.chunker import Chunk
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SearchResult:
    chunk_id: str
    text:     str
    source:   str
    title:    str
    score:    float
    metadata: dict


class VectorStore:
    COLLECTIONS = {
        "confluence": "confluence_docs",
        "sharepoint": "sharepoint_docs",
        "file":       "uploaded_files",
    }

    def __init__(self, persist_directory: str = "./chroma_store"):
        try:
            import chromadb
        except ImportError:
            raise ImportError("pip install chromadb")

        self.persist_dir = Path(persist_directory)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        # anonymized_telemetry=False disables ChromaDB phone-home
        settings = chromadb.config.Settings(anonymized_telemetry=False)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir), settings=settings
        )
        self._cols: dict[str, Any] = {}

        for name in self.COLLECTIONS.values():
            self._get_or_create(name)

        log.info(f"vectordb_ready  persist_dir={self.persist_dir}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_or_create(self, name: str):
        if name not in self._cols:
            self._cols[name] = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._cols[name]

    def collection_for(self, source: str) -> str:
        return self.COLLECTIONS.get(source, f"{source}_docs")

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_chunks(self, collection_name: str,
                   chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")

        col = self._get_or_create(collection_name)
        # Build safe metadata (ChromaDB requires str/int/float/bool values only)
        metadatas = []
        for c in chunks:
            meta = {"source": c.source, "title": c.title, "chunk_index": c.chunk_index}
            for k, v in c.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v)
            metadatas.append(meta)

        col.upsert(
            ids        = [c.id for c in chunks],
            embeddings = embeddings,
            documents  = [c.text for c in chunks],
            metadatas  = metadatas,
        )
        log.info(f"chunks_added  collection={collection_name}  count={len(chunks)}")

    def clear_collection(self, name: str) -> None:
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        self._cols.pop(name, None)
        self._get_or_create(name)
        log.info(f"collection_cleared  name={name}")

    def delete_by_meta(self, collection_name: str, where: dict) -> None:
        col     = self._get_or_create(collection_name)
        results = col.get(where=where)
        if results["ids"]:
            col.delete(ids=results["ids"])
            log.info(f"chunks_deleted  collection={collection_name}  count={len(results['ids'])}")

    # ── Read ──────────────────────────────────────────────────────────────────

    def search(self, collection_name: str, query_embedding: list[float],
               top_k: int = 6, score_threshold: float = 0.0,
               where: dict | None = None) -> list[SearchResult]:
        col   = self._get_or_create(collection_name)
        count = col.count()
        if count == 0:
            return []

        kwargs: dict = dict(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        raw     = col.query(**kwargs)
        results = []
        for doc, meta, dist in zip(raw["documents"][0], raw["metadatas"][0], raw["distances"][0]):
            score = max(0.0, 1.0 - float(dist))
            if score < score_threshold:
                continue
            results.append(SearchResult(
                chunk_id=meta.get("chunk_id", ""),
                text=doc,
                source=meta.get("source", ""),
                title=meta.get("title", ""),
                score=round(score, 4),
                metadata=meta,
            ))
        return results

    def search_all(self, query_embedding: list[float],
                   top_k: int = 6, score_threshold: float = 0.0) -> list[SearchResult]:
        all_results = []
        for name in self.COLLECTIONS.values():
            all_results.extend(self.search(name, query_embedding, top_k, score_threshold))
        return sorted(all_results, key=lambda r: r.score, reverse=True)[:top_k]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        return {name: self._get_or_create(name).count() for name in self.COLLECTIONS.values()}

    def total_chunks(self) -> int:
        return sum(self.stats().values())
