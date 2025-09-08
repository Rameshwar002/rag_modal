import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

client = PersistentClient(path="./chromadb_store")
collection = client.get_or_create_collection("documents")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def getEmbedding(text: str):
    """Generate embedding for a text chunk."""
    return embedder.encode(text).tolist()

def countTokens(text: str) -> int:
    """Approximate token count (by words)."""
    return len(text.split())

def chunkText(text, max_tokens=200):
    """Split text into smaller chunks (word-based)."""
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield " ".join(words[i:i+max_tokens])

def load_pdf(file_path: str):
    """Extract text from PDF (page by page)."""
    reader = PdfReader(file_path)
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            pages.append((page_num, text))
    return pages

def load_txt(file_path: str):
    """Load plain text file as one page."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [(1, f.read())]

def process_file(file_path: str):
    if file_path.endswith(".pdf"):
        pages = load_pdf(file_path)
    elif file_path.endswith(".txt"):
        pages = load_txt(file_path)
    else:
        print(f"Skipping unsupported file: {file_path}")
        return

    chunk_count = 0
    for page_num, text in pages:
        for i, chunk in enumerate(chunkText(text, max_tokens=80)):  # ~80 words per chunk
            collection.add(
                documents=[chunk],
                embeddings=[getEmbedding(chunk)],
                ids=[f"{os.path.basename(file_path)}_p{page_num}_c{i}"],
                metadatas=[{
                    "source": file_path,
                    "page": page_num,
                    "tokens": countTokens(chunk)
                }]
            )
            chunk_count += 1

    print(f"✅ Stored {chunk_count} chunks from {file_path} into ChromaDB")


if __name__ == "__main__":
    # Example files (replace with your own)
    # files = ["UpdatedWithFilesRead/TestDB.pdf"]
    files =["UpdatedWithFilesRead/TestData.pdf"]

    for file in files:
        process_file(file)

    # Example query
    # query_db("What are lists in Python?")
