from chromadb import PersistentClient
from utils import getEmbedding


client = PersistentClient(path="./chromadb_store")
collection = client.get_or_create_collection("documents")

query = "What is Python?"
results = collection.query(
    query_embeddings=[getEmbedding(query)],
    n_results=2
)

seen = set()
print(f"\n Query: {query}\n")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"- {doc} (source: {meta['source']}, page: {meta.get('page','?')})")
    seen.add(doc)


# print("🔍 Query:", query, "\n")
# for res in results["documents"][0]:
#     if res not in seen:   # avoid repeats
#         print("-", res)
        