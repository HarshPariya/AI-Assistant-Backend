"""
MongoDB-backed persistence for vector stores and in-memory session data.
Survives Render server restarts and ephemeral filesystems.
"""
import os
import pickle
from datetime import datetime
from typing import Any

from pymongo import MongoClient
from bson.binary import Binary

MONGODB_URL = os.getenv("MONGODB_URL")
_client: MongoClient | None = None


def _get_db():
    global _client
    if not MONGODB_URL:
        return None
    if _client is None:
        _client = MongoClient(
            MONGODB_URL,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            retryWrites=True,
        )
    return _client["ai_assistant"]


def is_available() -> bool:
    try:
        db = _get_db()
        if db is None:
            return False
        db.command("ping")
        return True
    except Exception:
        return False


def save_vector_store(session_id: str, store: dict) -> None:
    """Persist a vector store (embeddings + chunks) to MongoDB."""
    db = _get_db()
    if db is None:
        return
    data = Binary(pickle.dumps(store, protocol=pickle.HIGHEST_PROTOCOL))
    db["vector_stores"].replace_one(
        {"_id": session_id},
        {"_id": session_id, "data": data, "updated_at": datetime.utcnow().isoformat()},
        upsert=True,
    )


def load_vector_store(session_id: str) -> dict | None:
    """Load a vector store from MongoDB."""
    db = _get_db()
    if db is None:
        return None
    row = db["vector_stores"].find_one({"_id": session_id})
    if not row or "data" not in row:
        return None
    return pickle.loads(row["data"])


def save_session_data(session_id: str, session_type: str, data: Any) -> None:
    """Persist module session metadata (chat history, doc list, vision info)."""
    db = _get_db()
    if db is None:
        return
    db["module_sessions"].replace_one(
        {"_id": session_id},
        {
            "_id": session_id,
            "type": session_type,
            "data": data,
            "updated_at": datetime.utcnow().isoformat(),
        },
        upsert=True,
    )


def load_session_data(session_id: str) -> Any | None:
    """Load module session metadata from MongoDB."""
    db = _get_db()
    if db is None:
        return None
    row = db["module_sessions"].find_one({"_id": session_id})
    if not row:
        return None
    return row.get("data")


def save_vision_image(session_id: str, image_bytes: bytes, ext: str) -> None:
    """Store vision session image in MongoDB."""
    db = _get_db()
    if db is None:
        return
    db["vision_images"].replace_one(
        {"_id": session_id},
        {
            "_id": session_id,
            "data": Binary(image_bytes),
            "ext": ext,
            "updated_at": datetime.utcnow().isoformat(),
        },
        upsert=True,
    )


def load_vision_image(session_id: str) -> tuple[bytes, str] | None:
    """Load vision session image from MongoDB."""
    db = _get_db()
    if db is None:
        return None
    row = db["vision_images"].find_one({"_id": session_id})
    if not row or "data" not in row:
        return None
    return bytes(row["data"]), row.get("ext", ".jpg")
