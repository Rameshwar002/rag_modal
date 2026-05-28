"""
safety/ethical_checks.py
--------------------------
Secure Retrieval + Ethical Bias Checks.
  1. SecureRetriever  : source-level access control (per-user allowed sources)
  2. BiasChecker      : detect and flag potentially biased language in answers
  3. ContentFilter    : block harmful / toxic content
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from src.vectordb.vector_store import SearchResult
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── 1. Secure Retrieval ───────────────────────────────────────────────────────

class SecureRetriever:
    """
    Filters retrieved chunks based on the current user's allowed sources.

    Usage:
        sec = SecureRetriever(allowed_sources=["file", "confluence"])
        safe_results = sec.filter(results)
    """
    def __init__(self, allowed_sources: list[str] | None = None):
        # None means all sources allowed
        self.allowed_sources = allowed_sources

    def filter(self, results: list[SearchResult]) -> list[SearchResult]:
        if self.allowed_sources is None:
            return results
        filtered = [r for r in results if r.source in self.allowed_sources]
        removed  = len(results) - len(filtered)
        if removed:
            log.info(f"secure_filter  removed={removed}  allowed={self.allowed_sources}")
        return filtered

    def is_allowed(self, source: str) -> bool:
        return self.allowed_sources is None or source in self.allowed_sources


# ── 2. Bias Checker ───────────────────────────────────────────────────────────

# Patterns that may signal biased language
BIAS_PATTERNS = [
    (r"\ball\s+(men|women|blacks|whites|muslims|christians|jews|asians)\b", "demographic_generalization"),
    (r"\b(always|never)\s+(lie|cheat|steal|fail)\b",                        "absolute_negative_claim"),
    (r"\b(inferior|superior)\s+(race|gender|religion|culture)\b",           "supremacy_language"),
    (r"\billegal\s+alien",                                                   "dehumanizing_language"),
]


@dataclass
class BiasReport:
    has_bias:    bool
    bias_types:  list[str]
    flagged:     list[str]    # the matched snippets
    severity:    str          # "none" | "low" | "medium" | "high"


class BiasChecker:
    def __init__(self, llm=None):
        self.llm = llm

    def check(self, text: str) -> BiasReport:
        bias_types = []
        flagged    = []

        for pattern, bias_type in BIAS_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                bias_types.append(bias_type)
                flagged.extend(matches if isinstance(matches[0], str) else
                               [" ".join(m) for m in matches])

        # LLM-enhanced check if available
        if self.llm and not bias_types:
            try:
                resp = self.llm.complete([{"role": "user", "content":
                    f"Does this text contain biased, discriminatory, or harmful language? "
                    f"Reply YES or NO only:\n\n{text[:500]}"}])
                if "yes" in resp.lower():
                    bias_types.append("llm_detected")
            except Exception:
                pass

        severity = "none" if not bias_types else \
                   "high" if len(bias_types) >= 2 else \
                   "medium" if "supremacy_language" in bias_types else "low"

        if bias_types:
            log.warning(f"bias_detected  types={bias_types}  severity={severity}")

        return BiasReport(
            has_bias  = bool(bias_types),
            bias_types= bias_types,
            flagged   = flagged,
            severity  = severity,
        )


# ── 3. Content Filter ─────────────────────────────────────────────────────────

HARMFUL_PATTERNS = [
    r"\b(how\s+to\s+(make|build|create)\s+(bomb|weapon|poison|malware))\b",
    r"\b(suicide\s+method|self[\s-]harm\s+instruction)\b",
    r"\b(child\s+(abuse|exploitation|porn))\b",
]


class ContentFilter:
    def is_safe(self, text: str) -> tuple[bool, str]:
        """Returns (is_safe, reason)."""
        for pattern in HARMFUL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                reason = f"Harmful content pattern detected: {pattern}"
                log.warning(f"content_blocked  pattern={pattern}")
                return False, reason
        return True, ""

    def safe_answer(self, text: str) -> str:
        """Return the text or a safe refusal if harmful."""
        safe, reason = self.is_safe(text)
        if not safe:
            return "I'm unable to provide that information."
        return text
