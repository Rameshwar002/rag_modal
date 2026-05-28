"""
feedback/feedback_store.py
---------------------------
Error Analysis & Feedback Loops.
Collects user feedback (thumbs up/down, corrections) and
stores them for continuous fine-tuning / prompt improvement.
"""
from __future__ import annotations
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from src.utils.logger import get_logger

log = get_logger(__name__)

FEEDBACK_FILE = "logs/feedback.jsonl"


@dataclass
class FeedbackEntry:
    id:           str
    question:     str
    answer:       str
    rating:       int           # 1 = thumbs up, -1 = thumbs down, 0 = neutral
    correction:   str = ""      # user-provided correct answer
    comment:      str = ""
    sources_used: list[str] = field(default_factory=list)
    eval_score:   float = 0.0   # from RAGEvaluator if available
    timestamp:    float = field(default_factory=time.time)

    @property
    def is_positive(self) -> bool:
        return self.rating > 0


class FeedbackStore:
    def __init__(self, filepath: str = FEEDBACK_FILE):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[FeedbackEntry] = self._load()

    def _load(self) -> list[FeedbackEntry]:
        if not self.filepath.exists():
            return []
        entries = []
        with open(self.filepath) as f:
            for line in f:
                try:
                    entries.append(FeedbackEntry(**json.loads(line)))
                except Exception:
                    pass
        return entries

    def add(self, entry: FeedbackEntry) -> None:
        self._entries.append(entry)
        with open(self.filepath, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")
        log.info(f"feedback_saved  id={entry.id}  rating={entry.rating}")

    def get_all(self) -> list[FeedbackEntry]:
        return list(self._entries)

    def get_negative(self) -> list[FeedbackEntry]:
        """Return all negative feedback — useful for error analysis."""
        return [e for e in self._entries if e.rating < 0]

    def get_with_corrections(self) -> list[FeedbackEntry]:
        """Return entries where user provided a correction — gold for fine-tuning."""
        return [e for e in self._entries if e.correction.strip()]

    def stats(self) -> dict:
        if not self._entries:
            return {"total": 0}
        pos = sum(1 for e in self._entries if e.rating > 0)
        neg = sum(1 for e in self._entries if e.rating < 0)
        return {
            "total":       len(self._entries),
            "positive":    pos,
            "negative":    neg,
            "neutral":     len(self._entries) - pos - neg,
            "satisfaction": round(pos / len(self._entries) * 100, 1),
            "corrections": len(self.get_with_corrections()),
        }

    def export_for_finetuning(self) -> list[dict]:
        """
        Export positive feedback + corrections as training pairs.
        Format: [{"prompt": question, "completion": answer}, ...]
        """
        pairs = []
        for e in self._entries:
            if e.rating > 0:
                pairs.append({"prompt": e.question, "completion": e.answer})
            elif e.correction:
                pairs.append({"prompt": e.question, "completion": e.correction})
        return pairs
