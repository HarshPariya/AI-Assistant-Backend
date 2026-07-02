"""
Embeddings Utility — Refactored to use LangChain (FAISS + HuggingFaceEmbeddings)
"""
import asyncio
import os

# Force PyTorch and disable TensorFlow in transformers to avoid tf-keras Python 3.13 issues
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from utils.session_store import save_vector_store, load_vector_store

VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "vector_store")

_embeddings: HuggingFaceEmbeddings | None = None

def get_embeddings() -> HuggingFaceEmbeddings:
    """Get or create the LangChain HuggingFaceEmbeddings."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    return _embeddings


def preload_embedding_model() -> None:
    """Warm up the embedding model on startup."""
    get_embeddings()


def build_and_save_vector_store(session_id: str, chunks: list[dict]) -> int:
    """Embed chunks using FAISS and save to disk + MongoDB."""
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    embeddings = get_embeddings()
    
    texts = [c["text"] for c in chunks]
    metadatas = [{"page": c.get("page", 1)} for c in chunks]
    
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    
    # Save to disk
    store_path = os.path.join(VECTOR_STORE_DIR, session_id)
    vectorstore.save_local(store_path)
    
    # Save to MongoDB
    try:
        index_bytes = vectorstore.serialize_to_bytes()
        save_vector_store(session_id, {"faiss_bytes": index_bytes})
    except Exception as e:
        print(f"Failed to save FAISS to MongoDB: {e}")
        
    return len(chunks)


async def async_build_and_save_vector_store(session_id: str, chunks: list[dict]) -> int:
    """Non-blocking vector store build."""
    return await asyncio.to_thread(build_and_save_vector_store, session_id, chunks)


def similarity_search(session_id: str, query: str, top_k: int = 5) -> list[dict]:
    """Search using FAISS."""
    store_path = os.path.join(VECTOR_STORE_DIR, session_id)
    embeddings = get_embeddings()
    vectorstore = None
    
    if os.path.exists(store_path):
        vectorstore = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
    else:
        # Fallback to MongoDB
        data = load_vector_store(session_id)
        if data and "faiss_bytes" in data:
            vectorstore = FAISS.deserialize_from_bytes(data["faiss_bytes"], embeddings, allow_dangerous_deserialization=True)
            # Re-save locally for faster access next time
            os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
            vectorstore.save_local(store_path)
            
    if vectorstore is None:
        return []

    # FAISS with cosine distance (since we normalize embeddings, inner product/L2 is proportional to cosine)
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=top_k)
    
    results = []
    for doc, score in docs_and_scores:
        # In FAISS L2, smaller score is closer. 
        results.append({
            "text": doc.page_content,
            "page": doc.metadata.get("page", 1),
            "score": float(score) 
        })

    return results


async def async_similarity_search(session_id: str, query: str, top_k: int = 5) -> list[dict]:
    """Non-blocking similarity search."""
    return await asyncio.to_thread(similarity_search, session_id, query, top_k)
