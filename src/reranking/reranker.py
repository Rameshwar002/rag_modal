"""
reranking/reranker.py
----------------------
Cross-Encoder style reranking — re-scores retrieved chunks
against the query for higher precision.

Modes:
  llm    : use the LLM to score relevance (works with any LLM, no extra deps)
  bm25   : keyword overlap scoring (zero-dep fallback)
  cohere : Cohere Rerank API (optional, needs `pip install cohere`)
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from src.vectordb.vector_store import SearchResult
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RankedResult:
    result:        SearchResult
    original_score: float
    rerank_score:  float


class Reranker:
    def __init__(self, mode: str = "bm25", llm=None, cohere_api_key: str | None = None):
        self.mode            = mode
        self.llm             = llm
        self.cohere_api_key  = cohere_api_key

    def rerank(self, query: str, results: list[SearchResult],
               top_k: int | None = None) -> list[SearchResult]:
        if not results:
            return results

        if self.mode == "llm" and self.llm:
            ranked = self._llm_rerank(query, results)
        elif self.mode == "cohere" and self.cohere_api_key:
            ranked = self._cohere_rerank(query, results)
        else:
            ranked = self._bm25_rerank(query, results)

        k = top_k or len(ranked)
        final = [r.result for r in ranked[:k]]
        log.info(f"rerank_done  mode={self.mode}  input={len(results)}  output={len(final)}")
        return final

    # ── BM25-style keyword rerank (no deps) ───────────────────────────────────
    def _bm25_rerank(self, query: str, results: list[SearchResult]) -> list[RankedResult]:
        query_tokens = set(re.findall(r"\w+", query.lower()))
        ranked = []
        for r in results:
            doc_tokens  = re.findall(r"\w+", r.text.lower())
            doc_set     = set(doc_tokens)
            overlap     = len(query_tokens & doc_set)
            # TF component
            tf          = sum(doc_tokens.count(t) for t in query_tokens) / max(len(doc_tokens), 1)
            bm25_score  = overlap * (1 + tf)
            # Combine with original embedding score
            combined    = 0.4 * r.score + 0.6 * min(bm25_score / 10, 1.0)
            ranked.append(RankedResult(result=r, original_score=r.score, rerank_score=combined))

        return sorted(ranked, key=lambda x: x.rerank_score, reverse=True)

    # ── LLM rerank ────────────────────────────────────────────────────────────
    def _llm_rerank(self, query: str, results: list[SearchResult]) -> list[RankedResult]:
        scored = []
        for r in results:
            messages = [{
                "role": "user",
                "content": (
                    f"Rate how relevant this passage is to the query on a scale 0-10.\n"
                    f"Reply with ONLY a number.\n\n"
                    f"Query: {query}\n\nPassage: {r.text[:500]}"
                )
            }]
            try:
                raw   = self.llm.complete(messages).strip()
                score = float(re.search(r"\d+(?:\.\d+)?", raw).group()) / 10.0
            except Exception:
                score = r.score
            scored.append(RankedResult(result=r, original_score=r.score, rerank_score=score))

        return sorted(scored, key=lambda x: x.rerank_score, reverse=True)

    # ── Cohere rerank ─────────────────────────────────────────────────────────
    def _cohere_rerank(self, query: str, results: list[SearchResult]) -> list[RankedResult]:
        import cohere
        co       = cohere.Client(self.cohere_api_key)
        docs     = [r.text[:512] for r in results]
        response = co.rerank(query=query, documents=docs, model="rerank-english-v3.0", top_n=len(docs))
        ranked   = [None] * len(results)
        for item in response.results:
            ranked[item.index] = RankedResult(
                result=results[item.index],
                original_score=results[item.index].score,
                rerank_score=item.relevance_score,
            )
        return sorted([r for r in ranked if r], key=lambda x: x.rerank_score, reverse=True)
