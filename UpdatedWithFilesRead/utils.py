from sentence_transformers import SentenceTransformer
import tiktoken

# loaded the emaded model you can change accordly 
embedder =SentenceTransformer("all-MiniLM-L6-v2")

def getEmbedding(text: str):
    return embedder.encode(text).tolist()

def countTokens(text: str, model: str ="gpt-3.5-turbo"):
    enc= tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def chunkText(text: str, max_tokens: int = 200):
    """Split text into smaller chunks based on token length."""
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = enc.encode(text)

    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunks.append(enc.decode(chunk_tokens))
    return chunks