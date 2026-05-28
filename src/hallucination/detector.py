"""
hallucination/detector.py
--------------------------
Hallucination Detection & Control.

Checks whether the LLM answer is grounded in the retrieved context.
Returns a confidence score and flagged sentences.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class HallucinationReport:
    is_grounded:       bool
    confidence:        float          # 0.0 – 1.0  (1 = fully grounded)
    flagged_sentences: list[str]      # sentences not supported by context
    verdict:           str            # "grounded" | "partial" | "hallucination"


class HallucinationDetector:
    def __init__(self, llm=None, mode: str = "overlap"):
        """
        mode:
          overlap  — fast, no LLM needed. Checks word overlap.
          llm      — uses the LLM to verify each claim.
        """
        self.llm  = llm
        self.mode = mode if llm or mode == "overlap" else "overlap"

    def check(self, answer: str, context: str) -> HallucinationReport:
        if self.mode == "llm" and self.llm:
            return self._llm_check(answer, context)
        return self._overlap_check(answer, context)

    # ── Overlap check (no LLM needed) ─────────────────────────────────────────
    def _overlap_check(self, answer: str, context: str) -> HallucinationReport:
        context_tokens = set(re.findall(r"\w+", context.lower()))
        sentences      = re.split(r"(?<=[.!?])\s+", answer.strip())
        flagged        = []

        scores = []
        for sent in sentences:
            sent_tokens = set(re.findall(r"\w+", sent.lower()))
            if not sent_tokens:
                continue
            # Remove stopwords for scoring
            stopwords   = {"the","a","an","is","are","was","were","be","been","i","it",
                           "in","on","at","to","for","of","and","or","but","this","that"}
            key_tokens  = sent_tokens - stopwords
            if not key_tokens:
                scores.append(1.0)
                continue
            overlap     = len(key_tokens & context_tokens) / len(key_tokens)
            scores.append(overlap)
            if overlap < 0.3:
                flagged.append(sent)

        confidence = sum(scores) / len(scores) if scores else 0.5

        if confidence >= 0.7:
            verdict = "grounded"
        elif confidence >= 0.4:
            verdict = "partial"
        else:
            verdict = "hallucination"

        log.info(f"hallucination_check  mode=overlap  confidence={confidence:.2f}  verdict={verdict}")
        return HallucinationReport(
            is_grounded=confidence >= 0.5,
            confidence=round(confidence, 3),
            flagged_sentences=flagged,
            verdict=verdict,
        )

    # ── LLM check ─────────────────────────────────────────────────────────────
    def _llm_check(self, answer: str, context: str) -> HallucinationReport:
        messages = [{
            "role": "user",
            "content": (
                f"You are a fact-checker. Given this CONTEXT and ANSWER, "
                f"identify any claims in the answer NOT supported by the context.\n\n"
                f"CONTEXT:\n{context[:2000]}\n\n"
                f"ANSWER:\n{answer}\n\n"
                f"Reply in this exact format:\n"
                f"SCORE: <0-100>\n"
                f"UNSUPPORTED: <list unsupported sentences, or 'none'>"
            )
        }]
        try:
            raw        = self.llm.complete(messages)
            score_match = re.search(r"SCORE:\s*(\d+)", raw)
            unsup_match = re.search(r"UNSUPPORTED:\s*(.+)", raw, re.DOTALL)
            score      = int(score_match.group(1)) / 100 if score_match else 0.5
            unsupported_raw = unsup_match.group(1).strip() if unsup_match else ""
            flagged    = [] if "none" in unsupported_raw.lower() else [
                s.strip("•- ") for s in unsupported_raw.split("\n") if s.strip()
            ]
        except Exception as e:
            log.warning(f"llm_hallucination_check_failed  error={e}")
            return self._overlap_check(answer, context)

        verdict = "grounded" if score >= 0.7 else "partial" if score >= 0.4 else "hallucination"
        log.info(f"hallucination_check  mode=llm  confidence={score:.2f}  verdict={verdict}")
        return HallucinationReport(
            is_grounded=score >= 0.5,
            confidence=round(score, 3),
            flagged_sentences=flagged,
            verdict=verdict,
        )
