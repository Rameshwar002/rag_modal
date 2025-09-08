from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# -----------------------------
# 1. Test Embedding Model
# -----------------------------
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  
sentence = "Data engineering is about building data pipelines."
vector = embedder.encode(sentence)

print("✅ Embedding created!")

print("Vector shape:", vector.shape)  # Expect (384,)

# -----------------------------
# 2. Test ChromaDB
# -----------------------------
print("\nTesting ChromaDB...")
# persistent DB
client = chromadb.PersistentClient(path="./chromadb_store")
collection = client.get_or_create_collection("documents")


docs = ["Data engineering builds pipelines.",
        "Machine learning uses data prepared by data engineers."]
collection.add(documents=docs,
               ids=["1", "2"],
               embeddings=[embedder.encode(d).tolist() for d in docs])


query = "What do data engineers do?"
query_vector = embedder.encode(query).tolist()
results = collection.query(query_embeddings=[query_vector], n_results=2)

print("\n✅ Query test successful!")
print("Query:", query)
print("Results:", results["documents"][0])
