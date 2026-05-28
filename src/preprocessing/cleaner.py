"""
preprocessing/cleaner.py
-------------------------
1. Text cleaning & normalization
2. PII Masking (emails, phones, SSNs, credit cards, names via regex)
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── PII patterns ──────────────────────────────────────────────────────────────
PII_PATTERNS = {
    "email":       (r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[EMAIL]"),
    "phone":       (r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]"),
    "ssn":         (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
    "credit_card": (r"\b(?:\d[ -]?){13,16}\b", "[CREDIT_CARD]"),
    "ip_address":  (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP_ADDRESS]"),
    "url":         (r"https?://[^\s]+", "[URL]"),
}


@dataclass
class CleanResult:
    original:     str
    cleaned:      str
    pii_found:    list[str]    # which PII types were detected
    char_removed: int


class TextCleaner:
    def __init__(
        self,
        mask_pii:         bool = True,
        remove_html:      bool = True,
        normalize_whitespace: bool = True,
        min_length:       int  = 20,
    ):
        self.mask_pii              = mask_pii
        self.remove_html           = remove_html
        self.normalize_whitespace  = normalize_whitespace
        self.min_length            = min_length

    def clean(self, text: str) -> CleanResult:
        original = text
        pii_found = []

        # 1. Remove HTML tags
        if self.remove_html:
            text = re.sub(r"<[^>]+>", " ", text)

        # 2. Normalize unicode
        text = text.encode("ascii", "ignore").decode("ascii") if text else text

        # 3. Remove control characters (keep newlines/tabs)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # 4. Mask PII
        if self.mask_pii:
            for pii_type, (pattern, replacement) in PII_PATTERNS.items():
                if re.search(pattern, text, re.IGNORECASE):
                    pii_found.append(pii_type)
                    text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 5. Normalize whitespace
        if self.normalize_whitespace:
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text.strip()

        if pii_found:
            log.info(f"pii_masked  types={pii_found}")

        return CleanResult(
            original     = original,
            cleaned      = text,
            pii_found    = pii_found,
            char_removed = len(original) - len(text),
        )

    def clean_many(self, texts: list[str]) -> list[CleanResult]:
        return [self.clean(t) for t in texts]

    def is_valid(self, text: str) -> bool:
        """Return False if text is too short or mostly garbage."""
        stripped = text.strip()
        if len(stripped) < self.min_length:
            return False
        # Reject if >40% non-alphanumeric chars
        alpha = sum(c.isalnum() or c.isspace() for c in stripped)
        return (alpha / len(stripped)) > 0.6
