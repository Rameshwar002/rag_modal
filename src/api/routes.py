"""
src/api/routes.py  —  Flask Blueprint
All features wired: preprocessing, reranking, query reformulation,
hallucination detection, evaluation, cache, feedback, safety.
"""
from __future__ import annotations
import time
import tempfile
import uuid
from pathlib import Path

from flask import Blueprint, jsonify, request

from src.chunking.chunker import Chunker
from src.ingestion.loader import load_confluence, load_file, load_sharepoint
from src.preprocessing.cleaner import TextCleaner
from src.prompts.prompt_templates import build_rag_prompt
from src.retrieval.retriever import Retriever
from src.vectordb.vector_store import VectorStore
from src.cache.retrieval_cache import RetrievalCache
from src.hallucination.detector import HallucinationDetector
from src.evaluation.metrics import RAGEvaluator, Timer
from src.feedback.feedback_store import FeedbackStore, FeedbackEntry
from src.safety.ethical_checks import BiasChecker, ContentFilter, SecureRetriever
from src.reranking.reranker import Reranker
from src.utils.logger import get_logger

log     = get_logger(__name__)
bp      = Blueprint("rag", __name__)
cleaner = TextCleaner(mask_pii=True)
content_filter = ContentFilter()

# ── Singletons ────────────────────────────────────────────────────────────────
_store:        VectorStore | None = None
_embedder                         = None
_chunker:      Chunker | None     = None
_llm                              = None
_retriever:    Retriever | None   = None
_llm_provider: str                = "none"
_llm_model:    str                = ""
_cache         = RetrievalCache()
_hallucination = HallucinationDetector()
_evaluator     = RAGEvaluator()
_feedback      = FeedbackStore()
_bias_checker  = BiasChecker()
_reranker      = Reranker(mode="bm25")


def init_dependencies(store, embedder, chunker, llm, retriever,
                      llm_provider="none", llm_model=""):
    global _store, _embedder, _chunker, _llm, _retriever
    global _llm_provider, _llm_model
    global _hallucination, _evaluator, _bias_checker, _reranker

    _store        = store
    _embedder     = embedder
    _chunker      = chunker
    _llm          = llm
    _retriever    = retriever
    _llm_provider = llm_provider
    _llm_model    = llm_model

    # Wire LLM into advanced features when available
    if llm:
        _hallucination = HallucinationDetector(llm=llm, mode="llm")
        _evaluator     = RAGEvaluator(llm=llm)
        _bias_checker  = BiasChecker(llm=llm)
        _reranker      = Reranker(mode="llm", llm=llm)
    else:
        _hallucination = HallucinationDetector(mode="overlap")
        _reranker      = Reranker(mode="bm25")


def _err(msg: str, code: int = 500):
    return jsonify({"error": msg}), code


def _ingest(docs, collection_name: str) -> dict:
    # Clean docs before chunking
    for doc in docs:
        result  = cleaner.clean(doc.text)
        doc.text = result.cleaned
    chunks     = _chunker.chunk_many(docs)
    embeddings = _embedder.embed_chunks(chunks)
    _store.add_chunks(collection_name, chunks, embeddings)
    return {"documents": len(docs), "chunks": len(chunks), "collection": collection_name}


# ── Health ────────────────────────────────────────────────────────────────────

@bp.get("/health")
def health():
    return jsonify({
        "status":       "ok",
        "total_chunks": _store.total_chunks(),
        "llm":          _llm_provider,
        "llm_model":    _llm_model,
        "cache":        _cache.stats(),
    })


@bp.get("/llm/status")
def llm_status():
    if _llm is None:
        return jsonify({"connected": False, "provider": "none", "model": "",
                        "message": "No LLM configured."})
    try:
        if _llm_provider == "ollama":
            import requests as req
            req.get(f"{_llm.base_url}/api/tags", timeout=3).raise_for_status()
        return jsonify({"connected": True, "provider": _llm_provider,
                        "model": _llm_model,
                        "message": f"{_llm_provider} / {_llm_model} is reachable"})
    except Exception as e:
        return jsonify({"connected": False, "provider": _llm_provider,
                        "model": _llm_model, "message": f"LLM unreachable: {e}"})


# ── Collections ───────────────────────────────────────────────────────────────

@bp.get("/collections")
def list_collections():
    return jsonify({"collections": _store.stats(),
                    "total_chunks": _store.total_chunks()})


@bp.delete("/collections/<name>")
def clear_collection(name):
    valid = list(VectorStore.COLLECTIONS.values())
    if name not in valid:
        return _err(f"Unknown collection. Valid: {valid}", 404)
    _store.clear_collection(name)
    _cache.clear()
    return jsonify({"status": "cleared", "collection": name})


# ── Ingest ────────────────────────────────────────────────────────────────────

@bp.post("/ingest/file")
def ingest_file():
    if "file" not in request.files:
        return _err("No file provided", 400)
    f      = request.files["file"]
    suffix = Path(f.filename).suffix or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        f.save(tmp.name); tmp_path = tmp.name
    try:
        doc = load_file(tmp_path)
        doc.title = f.filename
        # Validate content
        if not cleaner.is_valid(doc.text):
            return _err("File content too short or invalid", 400)
        info = _ingest([doc], _store.collection_for("file"))
        _cache.clear()
        return jsonify({"status": "ok", "filename": f.filename, **info})
    except Exception as e:
        log.error(f"ingest_file_failed  error={e}")
        return _err(str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@bp.post("/ingest/confluence")
def ingest_confluence():
    data    = request.get_json() or {}
    missing = [k for k in ["space_key","url","username","api_token"] if not data.get(k)]
    if missing:
        return _err(f"Missing fields: {missing}", 400)
    try:
        docs = load_confluence(space_key=data["space_key"], url=data["url"],
                               username=data["username"], api_token=data["api_token"])
        info = _ingest(docs, _store.collection_for("confluence"))
        _cache.clear()
        return jsonify({"status": "ok", "space": data["space_key"], **info})
    except Exception as e:
        log.error(f"ingest_confluence_failed  error={e}")
        return _err(str(e))


@bp.post("/ingest/sharepoint")
def ingest_sharepoint():
    data = request.get_json() or {}
    try:
        docs = load_sharepoint(site_url=data.get("site_url"),
                               folder_path=data.get("folder_path", "/Shared Documents"),
                               client_id=data.get("client_id"),
                               client_secret=data.get("client_secret"))
        info = _ingest(docs, _store.collection_for("sharepoint"))
        _cache.clear()
        return jsonify({"status": "ok", **info})
    except Exception as e:
        log.error(f"ingest_sharepoint_failed  error={e}")
        return _err(str(e))


# ── Chat  (full pipeline) ─────────────────────────────────────────────────────

@bp.post("/chat")
def chat():
    data     = request.get_json() or {}
    question = (data.get("question") or "").strip()
    if not question:
        return _err("question is required", 400)

    # Safety: check the question itself
    safe, reason = content_filter.is_safe(question)
    if not safe:
        return jsonify({
            "answer": "I can't help with that request.",
            "chunks_used": 0, "llm_used": False,
            "sources": [], "safety_blocked": True,
        })

    top_k      = int(data.get("top_k", 6))
    collection = data.get("collection")
    use_rerank = data.get("rerank", True)
    use_cache  = data.get("use_cache", True)

    with Timer() as t:
        try:
            # 1. Embed query — catch connection errors early
            try:
                query_embedding = _embedder.embed_texts([question])[0]
            except Exception as emb_err:
                return _err(
                    f"Embedding failed: {emb_err}. "
                    "Is Ollama running? Try: ollama serve", 503
                )

            # 2. Cache lookup
            cached = _cache.get(question, query_embedding) if use_cache else None
            if cached:
                results = cached
            else:
                # 3. Retrieve
                results = _retriever.retrieve(
                    question, collection=collection, top_k=top_k
                )
                # 4. Secure filter (all sources allowed by default)
                results = SecureRetriever().filter(results)
                # 5. Rerank
                if use_rerank and results:
                    results = _reranker.rerank(question, results, top_k=top_k)
                # 6. Cache store
                if use_cache:
                    _cache.set(question, query_embedding, results)

            # 7. Build context
            context = "\n\n---\n\n".join(
                f"[{i}] {r.title} ({r.source}) | score:{r.score:.2f}\n{r.text}"
                for i, r in enumerate(results, 1)
            )

            # 8. Generate answer
            if _llm is not None:
                messages = build_rag_prompt(context, question)
                answer   = _llm.complete(messages)
                llm_used = True
            else:
                if results:
                    parts  = [f"[{i}] {r.title}\n{r.text[:400]}{'...' if len(r.text)>400 else ''}"
                              for i, r in enumerate(results, 1)]
                    answer = "⚠️ No LLM connected — raw retrieved chunks:\n\n" + "\n\n---\n\n".join(parts)
                else:
                    answer = "No relevant content found. Upload files or connect a source first."
                llm_used = False

            # 9. Content filter on answer
            answer = content_filter.safe_answer(answer)

            # 10. Bias check
            bias_report = _bias_checker.check(answer)

            # 11. Hallucination check
            hal_report = _hallucination.check(answer, context) if context else None

            # 12. Evaluate (async-ish — doesn't block response)
            eval_result = _evaluator.evaluate(
                question=question, answer=answer, context=context,
                latency_ms=t.elapsed_ms, chunks_used=len(results),
            ) if context else None

        except Exception as e:
            log.error(f"chat_failed  error={e}")
            return _err(str(e))

    return jsonify({
        "answer":      answer,
        "chunks_used": len(results),
        "llm_used":    llm_used,
        "latency_ms":  round(t.elapsed_ms, 1),
        "sources": [
            {"title": r.title, "source": r.source, "score": r.score}
            for r in results
        ],
        "hallucination": {
            "verdict":    hal_report.verdict if hal_report else "n/a",
            "confidence": hal_report.confidence if hal_report else None,
        } if hal_report else None,
        "bias": {
            "detected":  bias_report.has_bias,
            "severity":  bias_report.severity,
        },
        "eval": {
            "overall":    eval_result.overall_score if eval_result else None,
            "verdict":    eval_result.verdict if eval_result else None,
            "faithfulness": eval_result.faithfulness if eval_result else None,
        } if eval_result else None,
    })


# ── Feedback ──────────────────────────────────────────────────────────────────

@bp.post("/feedback")
def submit_feedback():
    data = request.get_json() or {}
    if not data.get("question") or not data.get("answer"):
        return _err("question and answer required", 400)
    entry = FeedbackEntry(
        id         = str(uuid.uuid4())[:8],
        question   = data["question"],
        answer     = data["answer"],
        rating     = int(data.get("rating", 0)),
        correction = data.get("correction", ""),
        comment    = data.get("comment", ""),
        sources_used = data.get("sources", []),
    )
    _feedback.add(entry)
    return jsonify({"status": "ok", "id": entry.id})


@bp.get("/feedback/stats")
def feedback_stats():
    return jsonify(_feedback.stats())


@bp.get("/feedback/export")
def feedback_export():
    pairs = _feedback.export_for_finetuning()
    return jsonify({"count": len(pairs), "pairs": pairs})


# ── Evaluation stats ──────────────────────────────────────────────────────────

@bp.get("/eval/latency")
def eval_latency():
    return jsonify(_evaluator.latency_stats())


# ── Cache stats ───────────────────────────────────────────────────────────────

@bp.get("/cache/stats")
def cache_stats():
    return jsonify(_cache.stats())


@bp.delete("/cache")
def cache_clear():
    _cache.clear()
    return jsonify({"status": "cleared"})


# ── Serve UI ──────────────────────────────────────────────────────────────────

# UI is served by main.py at app level (not blueprint level)
# to avoid route conflicts. See main.py → serve_index()
