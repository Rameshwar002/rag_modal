"""
llm/llm_client.py
-----------------
LLM client supporting:
  - Ollama  (local)  — provider: "ollama"
  - Anthropic Claude — provider: "anthropic"
  - OpenAI           — provider: "openai"
"""

from __future__ import annotations
import os
from tenacity import retry, stop_after_attempt, wait_exponential
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Ollama (local) ────────────────────────────────────────────────────────────

class OllamaClient:
    """
    Calls a locally running Ollama server.
    Install:  https://ollama.com
    Run:      ollama run llama3.2
    Default:  http://localhost:11434
    """

    def __init__(
        self,
        model:       str   = "llama3.2",
        base_url:    str   = "http://localhost:11434",
        temperature: float = 0.2,
        max_tokens:  int   = 1024,
    ):
        self.model       = model
        self.base_url    = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens  = max_tokens

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def complete(self, messages: list[dict]) -> str:
        import requests, json

        # Merge all messages into a single prompt (simple approach)
        prompt = "\n\n".join(
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
            for m in messages
        )

        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model":  self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "")
        log.info(f"ollama_response  model={self.model}  chars={len(text)}")
        return text

    def stream(self, messages: list[dict]):
        import requests, json

        prompt = "\n\n".join(
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
            for m in messages
        )

        with requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": True,
                  "options": {"temperature": self.temperature}},
            stream=True, timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    yield chunk.get("response", "")
                    if chunk.get("done"):
                        break

    def list_models(self) -> list[str]:
        """Return models available in your local Ollama installation."""
        import requests
        resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]


# ── Anthropic Claude ──────────────────────────────────────────────────────────

class AnthropicClient:
    def __init__(self, model="claude-sonnet-4-20250514", max_tokens=1024, temperature=0.2, **_):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.client      = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def complete(self, messages: list[dict]) -> str:
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            temperature=self.temperature, messages=messages,
        )
        return "".join(b.text for b in resp.content if hasattr(b, "text"))

    def stream(self, messages: list[dict]):
        with self.client.messages.stream(
            model=self.model, max_tokens=self.max_tokens, messages=messages
        ) as s:
            for text in s.text_stream:
                yield text


# ── OpenAI ────────────────────────────────────────────────────────────────────

class OpenAIClient:
    def __init__(self, model="gpt-4o-mini", max_tokens=1024, temperature=0.2,
                 base_url=None, **_):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.client      = OpenAI(
            api_key  = os.environ.get("OPENAI_API_KEY"),
            base_url = base_url,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def complete(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages,
            max_tokens=self.max_tokens, temperature=self.temperature,
        )
        return resp.choices[0].message.content

    def stream(self, messages: list[dict]):
        for chunk in self.client.chat.completions.create(
            model=self.model, messages=messages,
            max_tokens=self.max_tokens, stream=True,
        ):
            yield chunk.choices[0].delta.content or ""


# ── Factory ───────────────────────────────────────────────────────────────────

def get_llm_client(
    provider:    str   = "ollama",
    model:       str   | None = None,
    max_tokens:  int   = 1024,
    temperature: float = 0.2,
    base_url:    str   | None = None,
):
    """
    Factory — pick provider in config.yaml:

      provider: ollama      model: llama3.2        (local, free)
      provider: anthropic   model: claude-sonnet-4-20250514
      provider: openai      model: gpt-4o-mini
    """
    kwargs = dict(max_tokens=max_tokens, temperature=temperature, base_url=base_url)

    if provider == "ollama":
        return OllamaClient(
            model    = model or "llama3.2",
            base_url = base_url or "http://localhost:11434",
            **{k: v for k, v in kwargs.items() if k in ("temperature", "max_tokens")},
        )
    elif provider == "anthropic":
        return AnthropicClient(model=model or "claude-sonnet-4-20250514", **kwargs)
    elif provider == "openai":
        return OpenAIClient(model=model or "gpt-4o-mini", **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Choose ollama | anthropic | openai")
