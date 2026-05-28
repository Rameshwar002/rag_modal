"""
query/reformulator.py
----------------------
Query Reformulation — improve retrieval quality by:
  1. HyDE  : generate a hypothetical answer, embed that instead of the raw query
  2. Expand: generate multiple query variants, retrieve for each, merge results
  3. Step-back: ask a more abstract question to retrieve broader context
"""
from __future__ import annotations
from src.utils.logger import get_logger

log = get_logger(__name__)


class QueryReformulator:
    def __init__(self, llm):
        self.llm = llm   # any llm_client with .complete(messages)

    # ── HyDE ─────────────────────────────────────────────────────────────────
    def hyde(self, query: str) -> str:
        """
        Hypothetical Document Embeddings.
        Ask the LLM to write a short hypothetical answer,
        then embed THAT instead of the raw query.
        """
        messages = [{
            "role": "user",
            "content": (
                f"Write a short 2-3 sentence document that would directly answer "
                f"this question. Only write the document, no preamble:\n\n{query}"
            )
        }]
        hypothetical = self.llm.complete(messages)
        log.info(f"hyde_generated  query={query[:50]!r}")
        return hypothetical.strip()

    # ── Query expansion ───────────────────────────────────────────────────────
    def expand(self, query: str, n: int = 3) -> list[str]:
        """
        Return `n` query variants + the original.
        Retrieve for all, merge & deduplicate results.
        """
        messages = [{
            "role": "user",
            "content": (
                f"Generate {n} different ways to ask the same question. "
                f"Return only the questions, one per line, no numbering:\n\n{query}"
            )
        }]
        raw      = self.llm.complete(messages)
        variants = [q.strip() for q in raw.strip().split("\n") if q.strip()][:n]
        all_queries = [query] + variants
        log.info(f"query_expanded  original={query[:50]!r}  variants={len(variants)}")
        return all_queries

    # ── Step-back prompting ───────────────────────────────────────────────────
    def step_back(self, query: str) -> str:
        """
        Generate a broader, more abstract version of the query
        to retrieve higher-level context.
        """
        messages = [{
            "role": "user",
            "content": (
                f"What is a broader, more general question that would give "
                f"background context to answer this specific question? "
                f"Return only the broader question:\n\n{query}"
            )
        }]
        broader = self.llm.complete(messages)
        log.info(f"step_back  original={query[:50]!r}")
        return broader.strip()

    # ── Multi-hop ─────────────────────────────────────────────────────────────
    def decompose(self, query: str) -> list[str]:
        """
        Break a complex query into sub-questions (multi-hop retrieval).
        """
        messages = [{
            "role": "user",
            "content": (
                f"Break this complex question into 2-4 simpler sub-questions "
                f"that together would answer the original. "
                f"Return only the sub-questions, one per line:\n\n{query}"
            )
        }]
        raw  = self.llm.complete(messages)
        subs = [q.strip() for q in raw.strip().split("\n") if q.strip()]
        log.info(f"decomposed  original={query[:50]!r}  sub_questions={len(subs)}")
        return subs
