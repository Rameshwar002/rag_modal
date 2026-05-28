"""
prompts/prompt_templates.py
----------------------------
All LLM prompt templates in one place.
"""
from __future__ import annotations

RAG_SYSTEM = """You are a precise, helpful RAG assistant.
Answer the user's question using ONLY the provided context.
- Be concise and factual.
- Cite the source title in brackets e.g. [API Authentication].
- If the context does not contain the answer say: "I don't have that information in the connected knowledge sources."
- Never make up information."""

NO_DOCS_SYSTEM = """You are a helpful assistant. No documents are currently indexed.
Tell the user they need to:
1. Connect Confluence or SharePoint via the Sources tab, or
2. Upload files via the Upload tab.
Then they can ask questions about their documents."""


def build_rag_prompt(context: str, question: str) -> list[dict]:
    if not context.strip():
        content = f"{NO_DOCS_SYSTEM}\n\nUser question: {question}"
    else:
        content = f"{RAG_SYSTEM}\n\nCONTEXT:\n{context}\n\nQUESTION: {question}"
    return [{"role": "user", "content": content}]
