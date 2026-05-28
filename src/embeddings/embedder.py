"""
embeddings/embedder.py
----------------------
Providers:
  - ollama     : local (nomic-embed-text, mxbai-embed-large)
  - anthropic  : Voyage via Anthropic SDK
  - openai     : text-embedding-3-small / large
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod

from tenacity import retry, stop_after_attempt, wait_exponential

from src.chunking.chunker import Chunk
from src.utils.helpers import chunk_list
from src.utils.logger import get_logger

log = get_logger(__name__)


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        return self.embed_texts([c.text for c in chunks])


# ── Ollama ────────────────────────────────────────────────────────────────────
class OllamaEmbedder(BaseEmbedder):
    def __init__(self, model="nomic-embed-text", base_url="http://localhost:11434", batch_size=16):
        self.model      = model
        self.base_url   = base_url.rstrip("/")
        self.batch_size = batch_size

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def _embed_one(self, text: str) -> list[float]:
        import requests
        resp = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        batches = chunk_list(texts, self.batch_size)
        for i, batch in enumerate(batches):
            log.info(f"embedding_batch  provider=ollama  batch={i+1}/{len(batches)}  size={len(batch)}")
            for text in batch:
                embeddings.append(self._embed_one(text))
        return embeddings


# ── Anthropic (Voyage) ────────────────────────────────────────────────────────
class AnthropicEmbedder(BaseEmbedder):
    def __init__(self, model="voyage-3", batch_size=32, api_key=None):
        import anthropic
        self.model      = model
        self.batch_size = batch_size
        self.client     = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _embed_batch(self, texts):
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for i, batch in enumerate(chunk_list(texts, self.batch_size)):
            log.info(f"embedding_batch  provider=anthropic  batch={i+1}  size={len(batch)}")
            embeddings.extend(self._embed_batch(batch))
            time.sleep(0.1)
        return embeddings


# ── OpenAI ────────────────────────────────────────────────────────────────────
class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model="text-embedding-3-small", batch_size=100, api_key=None):
        from openai import OpenAI
        self.model      = model
        self.batch_size = batch_size
        self.client     = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _embed_batch(self, texts):
        texts = [t.replace("\n", " ") for t in texts]
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for i, batch in enumerate(chunk_list(texts, self.batch_size)):
            log.info(f"embedding_batch  provider=openai  batch={i+1}  size={len(batch)}")
            embeddings.extend(self._embed_batch(batch))
        return embeddings


# ── Factory ───────────────────────────────────────────────────────────────────
def get_embedder(
    provider:   str = "ollama",
    model:      str | None = None,
    batch_size: int = 16,
    base_url:   str = "http://localhost:11434",
) -> BaseEmbedder:
    """
    provider: ollama | anthropic | openai
    """
    if provider == "ollama":
        return OllamaEmbedder(
            model=model or "nomic-embed-text",
            base_url=base_url,
            batch_size=batch_size,
        )
    elif provider == "anthropic":
        return AnthropicEmbedder(model=model or "voyage-3", batch_size=batch_size)
    elif provider == "openai":
        return OpenAIEmbedder(model=model or "text-embedding-3-small", batch_size=batch_size)
    else:
        raise ValueError(f"Unknown embedding provider: {provider!r}. Choose ollama | anthropic | openai")
