"""
main.py  —  Flask entry point
Run:   python main.py
Open:  http://localhost:5000
"""
from __future__ import annotations

import os
import sys

# ── Disable ALL telemetry before any imports ──────────────────────────────────
os.environ["ANONYMIZED_TELEMETRY"]        = "False"   # ChromaDB
os.environ["CHROMA_TELEMETRY"]            = "False"   # ChromaDB alt key
os.environ["POSTHOG_DISABLED"]            = "1"       # PostHog (used by Chroma)
os.environ["DISABLE_TELEMETRY"]           = "1"       # generic
os.environ["SCARF_NO_ANALYTICS"]          = "true"    # Scarf (langchain etc.)
os.environ["DO_NOT_TRACK"]               = "1"        # standard DNT

import yaml
from dotenv import load_dotenv
from flask import Flask, send_from_directory
from flask_cors import CORS

load_dotenv()

# ── Load config ───────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

from src.utils.logger import get_logger, setup_logging
setup_logging(level=cfg["logging"]["level"], log_file=cfg["logging"]["file"])
log = get_logger(__name__)

from src.api.routes import bp, init_dependencies
from src.chunking.chunker import Chunker
from src.embeddings.embedder import get_embedder
from src.llm.llm_client import get_llm_client
from src.retrieval.retriever import Retriever
from src.vectordb.vector_store import VectorStore

# ── Flask app ─────────────────────────────────────────────────────────────────
UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")

app = Flask(__name__, static_folder=UI_DIR, static_url_path="/static")
CORS(app)
app.register_blueprint(bp)


# ── Serve UI at root ──────────────────────────────────────────────────────────
@app.route("/")
def serve_index():
    index = os.path.join(UI_DIR, "index.html")
    if not os.path.exists(index):
        return (
            "<h2 style='font-family:monospace'>RAG API is running ✓</h2>"
            "<p>UI file not found at <code>ui/index.html</code></p>"
            "<p>API docs: <a href='/health'>/health</a></p>"
        ), 200
    return send_from_directory(UI_DIR, "index.html")


# ── Bootstrap ─────────────────────────────────────────────────────────────────
def _boot():
    store = VectorStore(persist_directory=cfg["vectordb"]["persist_directory"])

    embedder = get_embedder(
        provider   = cfg["embeddings"]["provider"],
        model      = cfg["embeddings"]["model"],
        batch_size = cfg["embeddings"]["batch_size"],
        base_url   = cfg["embeddings"].get("base_url", "http://localhost:11434"),
    )

    chunker = Chunker(
        strategy      = cfg["chunking"]["strategy"],
        chunk_size    = cfg["chunking"]["chunk_size"],
        chunk_overlap = cfg["chunking"]["chunk_overlap"],
        separators    = cfg["chunking"]["separators"],
    )

    llm          = None
    llm_provider = cfg["llm"].get("provider", "none")
    llm_model    = cfg["llm"].get("model", "")

    if llm_provider and llm_provider != "none":
        try:
            llm = get_llm_client(
                provider    = llm_provider,
                model       = llm_model,
                max_tokens  = cfg["llm"].get("max_tokens", 1024),
                temperature = cfg["llm"].get("temperature", 0.2),
                base_url    = cfg["llm"].get("base_url"),
            )
            log.info(f"llm_connected  provider={llm_provider}  model={llm_model}")
        except Exception as e:
            log.warning(f"llm_unavailable  provider={llm_provider}  error={e}")
    else:
        log.info("llm_disabled — set provider in config.yaml to enable")

    retriever = Retriever(
        embedder        = embedder,
        vector_store    = store,
        top_k           = cfg["retrieval"]["top_k"],
        score_threshold = cfg["retrieval"]["score_threshold"],
    )

    init_dependencies(
        store, embedder, chunker, llm, retriever,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    log.info(f"rag_ready  chunks={store.total_chunks()}  llm={llm_provider}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        _boot()
    except Exception as e:
        log.error(f"startup_failed  error={e}")
        sys.exit(1)

    port = int(os.environ.get("PORT", 5000))
    log.info(f"→  http://localhost:{port}")
    app.run(debug=True, port=port, use_reloader=False)
