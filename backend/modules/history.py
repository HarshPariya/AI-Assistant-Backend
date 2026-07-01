"""
History Module — Stores and retrieves per-user chat history using SQLite.
"""
import sqlite3
import json
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

DB_PATH = "history.db"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            module TEXT NOT NULL DEFAULT 'general',
            title TEXT,
            messages TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON conversations(user_id)")
    conn.commit()
    conn.close()


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
    conn = get_db()
    now = datetime.utcnow().isoformat()

    # Auto-generate title from first user message
    title = req.title
    if not title:
        first_user_msg = next((m for m in req.messages if m.role == "user"), None)
        if first_user_msg:
            title = first_user_msg.content[:60] + ("..." if len(first_user_msg.content) > 60 else "")
        else:
            title = "New Conversation"

    messages_json = json.dumps([m.dict() for m in req.messages])

    session_id = req.session_id

    if session_id:
        # Try to update existing session
        existing = conn.execute(
            "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
            (session_id, req.user_id)
        ).fetchone()
        if existing:
            conn.execute(
                """UPDATE conversations
                   SET messages = ?, title = ?, updated_at = ?
                   WHERE id = ? AND user_id = ?""",
                (messages_json, title, now, session_id, req.user_id)
            )
            conn.commit()
            conn.close()
            return {"id": session_id, "title": title, "action": "updated"}

    # Create new session
    new_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO conversations (id, user_id, module, title, messages, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (new_id, req.user_id, req.module, title, messages_json, now, now)
    )
    conn.commit()
    conn.close()
    return {"id": new_id, "title": title, "action": "created"}


@router.get("/user/{user_id}")
async def get_user_history(user_id: str, module: Optional[str] = None, limit: int = 50):
    """Get all conversation summaries for a user."""
    conn = get_db()
    if module:
        rows = conn.execute(
            """SELECT id, user_id, module, title, messages, created_at, updated_at
               FROM conversations WHERE user_id = ? AND module = ?
               ORDER BY updated_at DESC LIMIT ?""",
            (user_id, module, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT id, user_id, module, title, messages, created_at, updated_at
               FROM conversations WHERE user_id = ?
               ORDER BY updated_at DESC LIMIT ?""",
            (user_id, limit)
        ).fetchall()
    conn.close()

    return [
        {
            "id": row["id"],
            "user_id": row["user_id"],
            "module": row["module"],
            "title": row["title"],
            "message_count": len(json.loads(row["messages"])),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    ]


@router.get("/session/{session_id}")
async def get_conversation(session_id: str):
    """Get a full conversation by session ID."""
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM conversations WHERE id = ?", (session_id,)
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "module": row["module"],
        "title": row["title"],
        "messages": json.loads(row["messages"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


@router.delete("/session/{session_id}")
async def delete_conversation(session_id: str, user_id: str):
    """Delete a conversation (only by its owner)."""
    conn = get_db()
    result = conn.execute(
        "DELETE FROM conversations WHERE id = ? AND user_id = ?", (session_id, user_id)
    )
    conn.commit()
    conn.close()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Conversation not found or not authorized")
    return {"deleted": True}
