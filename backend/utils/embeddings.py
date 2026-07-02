"""
Embeddings Utility — Sentence Transformers + Pure NumPy Vector Store
Uses cosine similarity with numpy + pickle for persistence.
MongoDB fallback for Render/ephemeral deployments.
"""
import asyncio
import os

# Force PyTorch and disable TensorFlow in transformers to avoid tf-keras Python 3.13 issues
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Reduce memory and CPU footprint for serverless/Render deployments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils.session_store import save_vector_store, load_vector_store

VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "/tmp/vector_store")

_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Get or create the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def preload_embedding_model() -> None:
    """Warm up the embedding model on startup to avoid first-request delay."""
    get_embedding_model()


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts into normalized vectors."""
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    return np.array(embeddings, dtype="float32")


async def async_embed_texts(texts: list[str]) -> np.ndarray:
    """Non-blocking embedding."""
    return await asyncio.to_thread(embed_texts, texts)


def _load_store(session_id: str) -> dict | None:
    """Load vector store from disk or MongoDB."""
    store_path = os.path.join(VECTOR_STORE_DIR, f"{session_id}.pkl")
    if os.path.exists(store_path):
        with open(store_path, "rb") as f:
            return pickle.load(f)
    return load_vector_store(session_id)


def build_and_save_vector_store(session_id: str, chunks: list[dict]) -> int:
    """Embed chunks, compute embeddings, and save to disk + MongoDB. Returns chunk count."""
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

    save_vector_store(session_id, store)
    return len(chunks)


async def async_build_and_save_vector_store(session_id: str, chunks: list[dict]) -> int:
    """Non-blocking vector store build."""
    return await asyncio.to_thread(build_and_save_vector_store, session_id, chunks)


def similarity_search(session_id: str, query: str, top_k: int = 5) -> list[dict]:
    """Search for the most relevant chunks for a query using cosine similarity."""
    store = _load_store(session_id)
    if store is None:
        return []

    embeddings = store["embeddings"]
    chunks = store["chunks"]

    query_embedding = embed_texts([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if similarities[idx] > 0.1:
            chunk = chunks[idx].copy()
            chunk["score"] = float(similarities[idx])
            results.append(chunk)

    return results


async def async_similarity_search(session_id: str, query: str, top_k: int = 5) -> list[dict]:
    """Non-blocking similarity search."""
    return await asyncio.to_thread(similarity_search, session_id, query, top_k)
