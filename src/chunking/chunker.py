"""
chunking/chunker.py
-------------------
Splits Document text into Chunk objects ready for embedding.
Strategies: recursive | sentence | token
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.ingestion.loader import Document
from src.utils.helpers import generate_id
from src.utils.logger import get_logger

log = get_logger(__name__)

Strategy = Literal["recursive", "sentence", "token"]


@dataclass
class Chunk:
    id:          str
    text:        str
    source:      str
    title:       str
    chunk_index: int
    metadata:    dict = field(default_factory=dict)

    def __repr__(self):
        return f"Chunk(id={self.id}, index={self.chunk_index}, chars={len(self.text)})"


# ── Splitters ─────────────────────────────────────────────────────────────────

def _simple_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Pure-Python word-based fallback."""
    words  = text.split()
    step   = max(1, chunk_size - chunk_overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def _recursive_split(text: str, chunk_size: int, chunk_overlap: int,
                     separators: list[str]) -> list[str]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )
        return splitter.split_text(text)
    except ImportError:
        log.warning("langchain_text_splitters not installed — using simple splitter")
        return _simple_split(text, chunk_size, chunk_overlap)


def _sentence_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current, current_len = [], [], 0
    for sent in sentences:
        if current_len + len(sent) > chunk_size and current:
            chunks.append(" ".join(current))
            while current and sum(len(s) for s in current) > chunk_overlap:
                current.pop(0)
            current_len = sum(len(s) for s in current)
        current.append(sent)
        current_len += len(sent)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _token_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    # ~4 chars per token for English
    return _simple_split(text, chunk_size * 4, chunk_overlap * 4)


STRATEGIES = {
    "recursive": _recursive_split,
    "sentence":  _sentence_split,
    "token":     _token_split,
}


# ── Public API ────────────────────────────────────────────────────────────────

class Chunker:
    def __init__(
        self,
        strategy:      Strategy = "recursive",
        chunk_size:    int       = 512,
        chunk_overlap: int       = 64,
        separators:    list[str] | None = None,
    ):
        self.strategy      = strategy
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators    = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, doc: Document) -> list[Chunk]:
        fn = STRATEGIES.get(self.strategy, _recursive_split)

        if self.strategy == "recursive":
            raw = fn(doc.text, self.chunk_size, self.chunk_overlap, self.separators)
        else:
            raw = fn(doc.text, self.chunk_size, self.chunk_overlap)

        chunks = []
        for i, text in enumerate(raw):
            text = text.strip()
            if not text:
                continue
            chunks.append(Chunk(
                id          = generate_id(f"{doc.title}_{i}_{text[:30]}"),
                text        = text,
                source      = doc.source,
                title       = doc.title,
                chunk_index = i,
                metadata    = {
                    **doc.metadata,
                    "chunk_index":  i,
                    "total_chunks": len(raw),
                    "char_count":   len(text),
                },
            ))

        log.info(f"chunking_done  title={doc.title}  strategy={self.strategy}  chunks={len(chunks)}")
        return chunks

    def chunk_many(self, docs: list[Document]) -> list[Chunk]:
        result = []
        for doc in docs:
            result.extend(self.chunk(doc))
        return result
