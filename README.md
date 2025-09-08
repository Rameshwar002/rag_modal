
# 📘 RAG with ChromaDB (No GPU)

This project demonstrates a **lightweight Retrieval-Augmented Generation (RAG) pipeline** using:

* [Sentence Transformers](https://www.sbert.net/) for embeddings
* [ChromaDB](https://www.trychroma.com/) as a vector database

👉 Works on **CPU only** (no GPU required).

---

## 🚀 1. Installation

```bash
# Create virtual environment (recommended)
python -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install chromadb sentence-transformers pandas onnxruntime
```

⚠️ If you see warnings like
`WARNING: The script ... is not on PATH`,
you can ignore them or add the path to your system `PATH`.

---

## 📂 2. Project Structure

```
rag_project/
│── model.py
│── model2.py          
│── README.md          
│── chromadb_store/    # database files (auto-created)
```

---

## 📝 3. Usage Example

```python
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize DB (persistent storage)
client = chromadb.Client(Settings(persist_directory="./chromadb_store"))
collection = client.get_or_create_collection("documents")

# ✅ Add documents
docs = [
    "Python is a programming language.",
    "Llamas are domesticated South American camelids.",
    "ChromaDB is a vector database for AI applications."
]
embeddings = [embedder.encode(doc).tolist() for doc in docs]

collection.add(
    documents=docs,
    embeddings=embeddings,
    ids=[f"id_{i}" for i in range(len(docs))]
)

# 🔍 Query
query = "What is ChromaDB?"
results = collection.query(
    query_embeddings=[embedder.encode(query).tolist()],
    n_results=2
)

print("Query:", query)
print("Top Results:", results["documents"])
```

---

