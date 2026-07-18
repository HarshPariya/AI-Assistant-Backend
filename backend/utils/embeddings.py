"""
Embeddings Utility — ChromaDB + FastEmbed Vector Store
Uses ChromaDB for blazing-fast similarity search with MongoDB fallback.
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
import chromadb  # type: ignore
from fastembed import TextEmbedding
from utils.session_store import save_vector_store, load_vector_store

VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "vector_store")

_embedding_model: TextEmbedding | None = None
_chroma_client: chromadb.PersistentClient | None = None


def get_embedding_model() -> TextEmbedding:
    """Get or create the fastembed TextEmbedding model."""
    global _embedding_model
    if _embedding_model is None:
        # threads=1 strictly limits ONNX thread pooling to prevent Render OOM
        _embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5", threads=1)
    return _embedding_model


def get_chroma_client() -> chromadb.PersistentClient:
    """Get or create the persistent ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    return _chroma_client


def preload_embedding_model() -> None:
    """Warm up the embedding model and Chroma on startup to avoid first-request delay."""
    get_embedding_model()
    get_chroma_client()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts into normalized vectors (list of floats for Chroma)."""
    model = get_embedding_model()
    embeddings = list(model.embed(texts))
    # Chroma requires list of list of floats, not numpy arrays
    return [e.tolist() for e in embeddings]


async def async_embed_texts(texts: list[str]) -> list[list[float]]:
    """Non-blocking embedding."""
    return await asyncio.to_thread(embed_texts, texts)


def _load_store(session_id: str):
    """Ensure ChromaDB has the collection for this session. 
    If not found locally (Render restart), restore from MongoDB backup."""
    client = get_chroma_client()
    
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
    
    ids = [f"id_{i}" for i in range(len(chunks))]
    metadatas = [{"chunk_id": c.get("chunk_id", ""), "page": c.get("page", 0)} for c in chunks]
    documents = [c.get("text", "") for c in chunks]
    
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    return collection


def build_and_save_vector_store(session_id: str, chunks: list[dict]) -> int:
    """Embed chunks, compute embeddings, and save to ChromaDB + MongoDB backup. Returns chunk count."""
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

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
    
    # 2. Save pure dict to MongoDB for Render ephemeral fallback
    store = {
        "embeddings": embeddings,
        "chunks": chunks,
    }
    save_vector_store(session_id, store)
    return len(chunks)


async def async_build_and_save_vector_store(session_id: str, chunks: list[dict]) -> int:
    """Non-blocking vector store build."""
    return await asyncio.to_thread(build_and_save_vector_store, session_id, chunks)


def similarity_search(session_id: str, query: str, top_k: int = 5) -> list[dict]:
    """Search for the most relevant chunks for a query using ChromaDB."""
    collection = _load_store(session_id)
    if collection is None:
        return []

    query_embedding = embed_texts([query])
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    # Format results to match previous return shape
    formatted_results = []
    
    if results['documents'] and len(results['documents'][0]) > 0:
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        distances = results['distances'][0]
        
        for doc, meta, distance in zip(docs, metas, distances):
            # ChromaDB cosine distance = 1 - cosine_similarity
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


async def async_similarity_search(session_id: str, query: str, top_k: int = 5) -> list[dict]:
    """Non-blocking similarity search."""
    return await asyncio.to_thread(similarity_search, session_id, query, top_k)
