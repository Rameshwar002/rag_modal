"""
retrieval/retriever.py
-----------------------
Embed query → search ChromaDB → return ranked results.
"""
from __future__ import annotations

from src.embeddings.embedder import BaseEmbedder
from src.vectordb.vector_store import SearchResult, VectorStore
from src.utils.logger import get_logger

log = get_logger(__name__)


class Retriever:
    def __init__(self, embedder: BaseEmbedder, vector_store: VectorStore,
                 top_k: int = 6, score_threshold: float = 0.20):
        self.embedder        = embedder
        self.vector_store    = vector_store
        self.top_k           = top_k
        self.score_threshold = score_threshold

    def retrieve(self, query: str, collection: str | None = None,
                 top_k: int | None = None,
                 score_threshold: float | None = None) -> list[SearchResult]:
        k        = top_k           if top_k           is not None else self.top_k
        min_score = score_threshold if score_threshold is not None else self.score_threshold

        query_embedding = self.embedder.embed_texts([query])[0]
        log.info(f"query_embedded  preview={query[:60]!r}")

        if collection:
            results = self.vector_store.search(
                collection_name=collection,
                query_embedding=query_embedding,
                top_k=k,
                score_threshold=min_score,
            )
        else:
            results = self.vector_store.search_all(
                query_embedding=query_embedding,
                top_k=k,
                score_threshold=min_score,
            )

        log.info(f"retrieval_done  results={len(results)}")
        return results

    def retrieve_with_context(self, query: str, **kwargs) -> str:
        results = self.retrieve(query, **kwargs)
        if not results:
            return ""
        parts = [
            f"[{i}] Source: {r.title} ({r.source}) | Score: {r.score:.2f}\n{r.text}"
            for i, r in enumerate(results, 1)
        ]
        return "\n\n---\n\n".join(parts)
