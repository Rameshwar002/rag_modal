"""
evaluation/metrics.py
----------------------
RAG Evaluation Metrics:
  - Faithfulness    : is the answer supported by context?
  - Answer Relevance: does the answer address the question?
  - Context Recall  : did we retrieve the right chunks?
  - Latency tracker : end-to-end pipeline timing
"""
from __future__ import annotations
import time
import re
from dataclasses import dataclass, field
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class EvalResult:
    question:          str
    answer:            str
    faithfulness:      float       # 0–1: answer grounded in context
    answer_relevance:  float       # 0–1: answer addresses the question
    context_recall:    float       # 0–1: retrieved chunks cover the answer
    latency_ms:        float       # end-to-end ms
    chunks_used:       int
    verdict:           str         # "good" | "acceptable" | "poor"
    details:           dict = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        return round(
            0.4 * self.faithfulness +
            0.4 * self.answer_relevance +
            0.2 * self.context_recall, 3
        )


class RAGEvaluator:
    def __init__(self, llm=None):
        self.llm      = llm
        self._timings: list[float] = []

    # ── Main eval ─────────────────────────────────────────────────────────────
    def evaluate(self, question: str, answer: str, context: str,
                 latency_ms: float = 0.0, chunks_used: int = 0) -> EvalResult:

        faithfulness     = self._score_faithfulness(answer, context)
        answer_relevance = self._score_relevance(question, answer)
        context_recall   = self._score_context_recall(answer, context)

        overall = 0.4 * faithfulness + 0.4 * answer_relevance + 0.2 * context_recall
        verdict = "good" if overall >= 0.7 else "acceptable" if overall >= 0.4 else "poor"

        self._timings.append(latency_ms)
        result = EvalResult(
            question=question, answer=answer,
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_recall=context_recall,
            latency_ms=latency_ms,
            chunks_used=chunks_used,
            verdict=verdict,
        )
        log.info(f"eval  verdict={verdict}  overall={overall:.2f}  latency={latency_ms:.0f}ms")
        return result

    # ── Faithfulness ──────────────────────────────────────────────────────────
    def _score_faithfulness(self, answer: str, context: str) -> float:
        if self.llm:
            return self._llm_score(
                f"Rate 0-10 how well this ANSWER is supported by the CONTEXT.\n"
                f"Reply with ONLY a number.\n\nCONTEXT:\n{context[:1500]}\n\nANSWER:\n{answer}"
            )
        # Fallback: word overlap
        ctx_words = set(re.findall(r"\w+", context.lower()))
        ans_words = set(re.findall(r"\w+", answer.lower()))
        if not ans_words:
            return 0.5
        return min(len(ans_words & ctx_words) / len(ans_words), 1.0)

    # ── Answer relevance ──────────────────────────────────────────────────────
    def _score_relevance(self, question: str, answer: str) -> float:
        if self.llm:
            return self._llm_score(
                f"Rate 0-10 how well this ANSWER addresses the QUESTION.\n"
                f"Reply with ONLY a number.\n\nQUESTION: {question}\n\nANSWER:\n{answer}"
            )
        q_words = set(re.findall(r"\w+", question.lower()))
        a_words = set(re.findall(r"\w+", answer.lower()))
        if not q_words:
            return 0.5
        return min(len(q_words & a_words) / len(q_words), 1.0)

    # ── Context recall ────────────────────────────────────────────────────────
    def _score_context_recall(self, answer: str, context: str) -> float:
        ans_words = set(re.findall(r"\w+", answer.lower()))
        ctx_words = set(re.findall(r"\w+", context.lower()))
        if not ans_words:
            return 0.0
        return min(len(ans_words & ctx_words) / len(ans_words), 1.0)

    # ── LLM scoring helper ────────────────────────────────────────────────────
    def _llm_score(self, prompt: str) -> float:
        try:
            raw   = self.llm.complete([{"role": "user", "content": prompt}])
            match = re.search(r"\d+(?:\.\d+)?", raw)
            return min(float(match.group()) / 10.0, 1.0) if match else 0.5
        except Exception:
            return 0.5

    # ── Aggregate stats ───────────────────────────────────────────────────────
    def latency_stats(self) -> dict:
        if not self._timings:
            return {}
        return {
            "count":   len(self._timings),
            "avg_ms":  round(sum(self._timings) / len(self._timings), 1),
            "min_ms":  round(min(self._timings), 1),
            "max_ms":  round(max(self._timings), 1),
            "p95_ms":  round(sorted(self._timings)[int(len(self._timings) * 0.95)], 1),
        }


# ── Convenience timer ─────────────────────────────────────────────────────────
class Timer:
    def __enter__(self):
        self._t = time.perf_counter()
        return self
    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._t) * 1000
