"""
Embeddings Utility — Hybrid Vector Store (ChromaDB with NumPy fallback)
Uses ChromaDB for blazing-fast similarity search if available.
Automatically falls back to pure NumPy vector store if chromadb is not installed (e.g. local Windows without C++ tools).
Ensures data is always backed up on MongoDB.
"""
import asyncio
import os
import pickle

# Force PyTorch and disable TensorFlow in transformers to avoid tf-keras Python 3.13 issues
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Reduce memory and CPU footprint for serverless/Render deployments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from fastembed import TextEmbedding
from utils.session_store import save_vector_store, load_vector_store

try:
    import chromadb  # type: ignore
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "vector_store")

_embedding_model: TextEmbedding | None = None
_chroma_client = None


def cosine_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pure numpy implementation of cosine similarity to replace scikit-learn."""
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
    return np.dot(X, Y.T) / (np.dot(X_norm, Y_norm.T) + 1e-10)


def get_embedding_model() -> TextEmbedding:
    """Get or create the fastembed TextEmbedding model."""
    global _embedding_model
    if _embedding_model is None:
        # threads=1 strictly limits ONNX thread pooling to prevent Render OOM
        _embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5", threads=1)
    return _embedding_model


def get_chroma_client():
    """Get or create the persistent ChromaDB client if available."""
    global _chroma_client
    if not CHROMA_AVAILABLE:
        return None
    if _chroma_client is None:
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    return _chroma_client


def preload_embedding_model() -> None:
    """Warm up the embedding model and Chroma on startup to avoid first-request delay."""
    get_embedding_model()
    if CHROMA_AVAILABLE:
        get_chroma_client()


def embed_texts(texts: list[str]) -> list[list[float]] | np.ndarray:
    """Embed a list of texts into normalized vectors."""
    model = get_embedding_model()
    embeddings = list(model.embed(texts))
    if CHROMA_AVAILABLE:
        # Chroma requires list of list of floats, not numpy arrays
        return [e.tolist() for e in embeddings]
    else:
        return np.array(embeddings, dtype="float32")


async def async_embed_texts(texts: list[str]):
    """Non-blocking embedding."""
    return await asyncio.to_thread(embed_texts, texts)


def _load_store_chroma(session_id: str):
    """Load store using ChromaDB."""
    client = get_chroma_client()
    if client is None:
        return None
    
    # Check if collection exists locally
    try:
        collection = client.get_collection(name=session_id)
        if collection.count() > 0:
            return collection
    except Exception:
        pass
        
    # Collection not found locally or is empty, try loading from MongoDB
    store = load_vector_store(session_id)
    if not store:
        return None
        
    # Re-hydrate ChromaDB collection from MongoDB backup
    collection = client.get_or_create_collection(
        name=session_id,
        metadata={"hnsw:space": "cosine"}
    )
    chunks = store["chunks"]
    embeddings = store["embeddings"]
    
    # Convert embeddings to list of lists if stored as numpy in MongoDB (or vice versa)
    if isinstance(embeddings, np.ndarray):
        embeddings_list = embeddings.tolist()
    else:
        embeddings_list = embeddings
        
    ids = [f"id_{i}" for i in range(len(chunks))]
    metadatas = [{"chunk_id": c.get("chunk_id", ""), "page": c.get("page", 0)} for c in chunks]
    documents = [c.get("text", "") for c in chunks]
    
    collection.add(
        embeddings=embeddings_list,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    return collection


def _load_store_numpy(session_id: str) -> dict | None:
    """Load store using local Pickle or MongoDB backup."""
    store_path = os.path.join(VECTOR_STORE_DIR, f"{session_id}.pkl")
    if os.path.exists(store_path):
        try:
            with open(store_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
            
    store = load_vector_store(session_id)
    if store:
        # Save locally for faster access next time
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        try:
            with open(store_path, "wb") as f:
                pickle.dump(store, f)
        except Exception:
            pass
    return store


def build_and_save_vector_store(session_id: str, chunks: list[dict]) -> int:
    """Embed chunks, compute embeddings, and save to Vector Store + MongoDB backup."""
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    if CHROMA_AVAILABLE:
        # 1. Save to ChromaDB for fast search
        client = get_chroma_client()
        collection = client.get_or_create_collection(
            name=session_id,
            metadata={"hnsw:space": "cosine"}
        )
        ids = [f"id_{i}" for i in range(len(chunks))]
        metadatas = [{"chunk_id": c.get("chunk_id", ""), "page": c.get("page", 0)} for c in chunks]
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    else:
        # 1. Save to local pickle
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        store_path = os.path.join(VECTOR_STORE_DIR, f"{session_id}.pkl")
        store = {
            "embeddings": embeddings,
            "chunks": chunks,
        }
        with open(store_path, "wb") as f:
            pickle.dump(store, f)

    # 2. Save pure dict to MongoDB for Render fallback/cross-environment compatibility
    # Ensure embeddings saved to Mongo are converted to a list so it's JSON serializable/pickle-friendly across platforms
    if isinstance(embeddings, np.ndarray):
        embeddings_to_save = embeddings.tolist()
    else:
        embeddings_to_save = embeddings

    store_data = {
        "embeddings": embeddings_to_save,
        "chunks": chunks,
    }
    save_vector_store(session_id, store_data)
    return len(chunks)


async def async_build_and_save_vector_store(session_id: str, chunks: list[dict]) -> int:
    """Non-blocking vector store build."""
    return await asyncio.to_thread(build_and_save_vector_store, session_id, chunks)


def similarity_search(session_id: str, query: str, top_k: int = 5) -> list[dict]:
    """Search for the most relevant chunks for a query."""
    if CHROMA_AVAILABLE:
        collection = _load_store_chroma(session_id)
        if collection is None:
            return []

        query_embedding = embed_texts([query])
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        formatted_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            distances = results['distances'][0]
            
            for doc, meta, distance in zip(docs, metas, distances):
                score = 1.0 - distance
                if score > 0.1:
                    chunk = {
                        "text": doc,
                        "score": float(score),
                    }
                    if meta:
                        chunk.update(meta)
                    formatted_results.append(chunk)
        return formatted_results
    else:
        store = _load_store_numpy(session_id)
        if store is None:
            return []

        embeddings = np.array(store["embeddings"], dtype="float32")
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
