"""
Smoke tests — run from project root:
  pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_helpers():
    from src.utils.helpers import generate_id, chunk_list
    assert len(generate_id("hello")) == 16
    assert generate_id("hello") == generate_id("hello")   # deterministic
    assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert chunk_list([], 2) == []


def test_chunker_basic():
    from src.ingestion.loader import Document
    from src.chunking.chunker import Chunker
    text  = "This is a sentence. " * 200
    doc   = Document(text=text, source="file", title="test.txt")
    chunks = Chunker(strategy="sentence", chunk_size=100, chunk_overlap=20).chunk(doc)
    assert len(chunks) >= 1
    assert all(c.text.strip() for c in chunks)
    assert all(c.source == "file" for c in chunks)


def test_chunker_recursive():
    from src.ingestion.loader import Document
    from src.chunking.chunker import Chunker
    text  = ("word " * 50 + "\n\n") * 10
    doc   = Document(text=text, source="file", title="test.md")
    chunks = Chunker(strategy="recursive", chunk_size=200, chunk_overlap=30).chunk(doc)
    assert len(chunks) > 1


def test_prompt_with_context():
    from src.prompts.prompt_templates import build_rag_prompt
    msgs = build_rag_prompt("Some context here", "What is X?")
    assert msgs[0]["role"] == "user"
    assert "CONTEXT" in msgs[0]["content"]
    assert "What is X?" in msgs[0]["content"]


def test_prompt_no_context():
    from src.prompts.prompt_templates import build_rag_prompt
    msgs = build_rag_prompt("", "What is X?")
    assert "Upload" in msgs[0]["content"] or "Connect" in msgs[0]["content"]


def test_load_txt(tmp_path):
    from src.ingestion.loader import load_file
    f = tmp_path / "sample.txt"
    f.write_text("Hello world\nSecond line")
    doc = load_file(f)
    assert "Hello world" in doc.text
    assert doc.source == "file"
    assert doc.title == "sample.txt"


def test_load_csv(tmp_path):
    from src.ingestion.loader import load_file
    f = tmp_path / "data.csv"
    f.write_text("name,age\nAlice,30\nBob,25")
    doc = load_file(f)
    assert "Alice" in doc.text
    assert "age" in doc.text


def test_vector_store_add_search(tmp_path):
    from src.ingestion.loader import Document
    from src.chunking.chunker import Chunker
    from src.vectordb.vector_store import VectorStore

    store  = VectorStore(persist_directory=str(tmp_path / "chroma"))
    doc    = Document(text="The capital of France is Paris. " * 20, source="file", title="geo.txt")
    chunks = Chunker(chunk_size=100, chunk_overlap=10).chunk(doc)

    # Use dummy embeddings (random floats) just to test storage/retrieval plumbing
    import random
    dim        = 768
    embeddings = [[random.random() for _ in range(dim)] for _ in chunks]
    store.add_chunks("uploaded_files", chunks, embeddings)

    assert store.stats()["uploaded_files"] == len(chunks)
    assert store.total_chunks() == len(chunks)

    # Search with a random vector — just verifies no crash and returns results
    query_vec = [random.random() for _ in range(dim)]
    results   = store.search("uploaded_files", query_vec, top_k=3)
    assert isinstance(results, list)
    store.clear_collection("uploaded_files")
    assert store.stats()["uploaded_files"] == 0
