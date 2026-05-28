"""Common utility helpers."""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any


def generate_id(text: str | None = None) -> str:
    if text:
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    return uuid.uuid4().hex[:16]


def chunk_list(lst: list, size: int) -> list[list]:
    return [lst[i: i + size] for i in range(0, len(lst), size)]


def file_extension(path: str | Path) -> str:
    return Path(path).suffix.lower()


def file_size_mb(path: str | Path) -> float:
    return Path(path).stat().st_size / (1024 * 1024)


def flatten(nested: list[list[Any]]) -> list[Any]:
    return [item for sub in nested for item in sub]
