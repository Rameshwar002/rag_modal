from chromadb import PersistentClient
import pandas as pd

client = PersistentClient(path="./chromadb_store")
collection = client.get_or_create_collection("documents")

data = collection.get()

print("\n--- Documents ---")
print(data["documents"])

print("\n--- Metadata ---")
print(data["metadatas"])

if data["documents"]:
    df = pd.DataFrame({
        "id": data["ids"],
        "document": data["documents"],
        "metadata": data["metadatas"]
    })
    df.to_csv("chroma_export.csv", index=False)
    print("\n✅ Exported to chroma_export.csv")
