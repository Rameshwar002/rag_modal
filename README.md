
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
pip install -r requirements.txt          
```

⚠️ If you see warnings like
`WARNING: The script ... is not on PATH`,
you can ignore them or add the path to your system `PATH`.

---