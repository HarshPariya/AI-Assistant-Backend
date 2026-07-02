"""
History Module — Stores and retrieves per-user chat history using MongoDB.
"""
import os
import json
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pymongo import MongoClient

router = APIRouter()

# MongoDB setup
MONGODB_URL = os.getenv("MONGODB_URL")
client = None
db = None
collection = None

if MONGODB_URL:
    try:
        client = MongoClient(
            MONGODB_URL,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            retryWrites=True,
        )
        db = client["ai_assistant"]
        collection = db["conversations"]
    except Exception as e:
        print(f"⚠️  MongoDB connection failed: {e}")
        client = None
        db = None
        collection = None


def init_db():
    """Create indexes if using MongoDB."""
    if collection is not None:
        collection.create_index("user_id")
        collection.create_index([("user_id", 1), ("module", 1)])
        collection.create_index([("user_id", 1), ("updated_at", -1)])
        if db is not None:
            db["vector_stores"].create_index("updated_at")
            db["module_sessions"].create_index("type")
        print("✅ MongoDB indexes ready.")


# ─── Pydantic Models ─────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str
    id: Optional[str] = None


class SaveConversationRequest(BaseModel):
    user_id: str
    module: str = "general"
    title: Optional[str] = None
    messages: List[Message]
    session_id: Optional[str] = None  # if provided, upsert existing session


class ConversationSummary(BaseModel):
    id: str
    user_id: str
    module: str
    title: str
    message_count: int
    created_at: str
    updated_at: str


class ConversationDetail(BaseModel):
    id: str
    user_id: str
    module: str
    title: str
    messages: List[Message]
    created_at: str
    updated_at: str


# ─── Routes ──────────────────────────────────────────────────────────────────

@router.post("/save")
async def save_conversation(req: SaveConversationRequest):
    """Save or update a conversation for a user."""
    if collection is None:
        raise HTTPException(status_code=500, detail="MongoDB not configured")

    now = datetime.utcnow().isoformat()

    # Auto-generate title from first user message
    title = req.title
    if not title:
        first_user_msg = next((m for m in req.messages if m.role == "user"), None)
        if first_user_msg:
            title = first_user_msg.content[:60] + ("..." if len(first_user_msg.content) > 60 else "")
        else:
            title = "New Conversation"

    messages_list = [m.dict() for m in req.messages]

    session_id = req.session_id

    if session_id:
        # Try to update existing session
        existing = collection.find_one({"_id": session_id, "user_id": req.user_id})
        if existing:
            collection.update_one(
                {"_id": session_id},
                {"$set": {"messages": messages_list, "title": title, "updated_at": now}}
            )
            return {"id": session_id, "title": title, "action": "updated"}

    # Create new session
    new_id = str(uuid.uuid4())
    doc = {
        "_id": new_id,
        "user_id": req.user_id,
        "module": req.module,
        "title": title,
        "messages": messages_list,
        "created_at": now,
        "updated_at": now
    }
    collection.insert_one(doc)
    return {"id": new_id, "title": title, "action": "created"}


@router.get("/user/{user_id}")
async def get_user_history(user_id: str, module: Optional[str] = None, limit: int = 50):
    """Get all conversation summaries for a user."""
    if collection is None:
        raise HTTPException(status_code=500, detail="MongoDB not configured")

    query = {"user_id": user_id}
    if module:
        query["module"] = module

    cursor = collection.find(query).sort("updated_at", -1).limit(limit)
    
    return [
        {
            "id": row["_id"],
            "user_id": row["user_id"],
            "module": row["module"],
            "title": row["title"],
            "message_count": len(row.get("messages", [])),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        for row in cursor
    ]


@router.get("/user/{user_id}/latest")
async def get_latest_conversation(user_id: str, module: str):
    """Get the most recent conversation for a user in a specific module."""
    if collection is None:
        raise HTTPException(status_code=500, detail="MongoDB not configured")

    row = collection.find_one(
        {"user_id": user_id, "module": module},
        sort=[("updated_at", -1)],
    )

    if not row:
        return None

    return {
        "id": row["_id"],
        "user_id": row["user_id"],
        "module": row["module"],
        "title": row["title"],
        "messages": row.get("messages", []),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


@router.get("/session/{session_id}")
async def get_conversation(session_id: str):
    """Get a full conversation by session ID."""
    if collection is None:
        raise HTTPException(status_code=500, detail="MongoDB not configured")

    row = collection.find_one({"_id": session_id})

    if not row:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {
        "id": row["_id"],
        "user_id": row["user_id"],
        "module": row["module"],
        "title": row["title"],
        "messages": row.get("messages", []),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


@router.delete("/session/{session_id}")
async def delete_conversation(session_id: str, user_id: str):
    """Delete a conversation (only by its owner)."""
    if collection is None:
        raise HTTPException(status_code=500, detail="MongoDB not configured")

    result = collection.delete_one({"_id": session_id, "user_id": user_id})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found or not authorized")
    return {"deleted": True}

