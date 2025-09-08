# Install dependencies first:
# pip install sentence-transformers chromadb

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# -----------------------------
# STEP 1: Initialize Embedding Model
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")  

# -----------------------------
# STEP 2: Setup Vector Database (Chroma)
# -----------------------------
client = chromadb.Client(Settings(persist_directory="./chromadb_store"))
collection = client.get_or_create_collection(name="documents")

# -----------------------------
# STEP 3: Example Documents
# -----------------------------
documents = [
    "Data engineering is about designing and maintaining data pipelines.",
    "Data pipelines are used to collect, clean, and store data for analysis.",
    "Data engineers often work with tools like Apache Spark, Airflow, and Snowflake.",
    "Machine learning engineers build models, but data engineers make sure the data is ready.",
    "ETL stands for Extract, Transform, Load in data engineering."
]

# Clear collection before adding
collection.delete(ids=[str(i) for i in range(len(documents))])

# Add documents with embeddings
for i, doc in enumerate(documents):
    vector = embedder.encode(doc).tolist()
    collection.add(documents=[doc], embeddings=[vector], ids=[str(i)])

print("✅ Documents added to vector DB")

# -----------------------------
# STEP 4: Ask Questions (Semantic Search)
# -----------------------------
def ask_question(query, top_k=3):
    query_vector = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)

    print("\n🔎 Question:", query)
    print("📖 Retrieved Answers:")
    for rank, doc in enumerate(results["documents"][0], start=1):
        print(f"{rank}. {doc}")

# -----------------------------
# STEP 5: Try Queries
# -----------------------------
ask_question("What do data pipelines do?")
ask_question("What tools are used by data engineers?")
ask_question("Explain ETL in data engineering.")
