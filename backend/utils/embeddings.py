"""
Embeddings Utility — Sentence Transformers + Pure NumPy Vector Store
Uses cosine similarity with numpy + pickle for persistence.
No C++ build tools required - works on Python 3.13+
"""
import os

# Force PyTorch and disable TensorFlow in transformers to avoid tf-keras Python 3.13 issues
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "vector_store")

_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Get or create the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts into normalized vectors."""
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embeddings, dtype="float32")


def build_and_save_vector_store(session_id: str, chunks: list[dict]) -> int:
    """Embed chunks, compute embeddings, and save to disk. Returns chunk count."""
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    
    store = {
        "embeddings": embeddings,
        "chunks": chunks,
    }
    
    store_path = os.path.join(VECTOR_STORE_DIR, f"{session_id}.pkl")
    with open(store_path, "wb") as f:
        pickle.dump(store, f)
    
    return len(chunks)


def similarity_search(session_id: str, query: str, top_k: int = 5) -> list[dict]:
    """Search for the most relevant chunks for a query using cosine similarity."""
    store_path = os.path.join(VECTOR_STORE_DIR, f"{session_id}.pkl")
    
    if not os.path.exists(store_path):
        return []
    
    with open(store_path, "rb") as f:
        store = pickle.load(f)
    
    embeddings = store["embeddings"]
    chunks = store["chunks"]
    
    query_embedding = embed_texts([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get top-k indices sorted by similarity
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.1:  # Minimum threshold
            chunk = chunks[idx].copy()
            chunk["score"] = float(similarities[idx])
            results.append(chunk)
    
    return results
