"""
Module 5: Image Caption & Visual Q&A
Upload images, generate captions, describe objects, Q&A via Groq Vision
"""
import os
import base64
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio
from fastapi.responses import StreamingResponse
from utils.llm import get_vision_model, chat_completion, async_chat_completion, chat_completion_stream
from utils.session_store import save_session_data, load_session_data, save_vision_image, load_vision_image

router = APIRouter()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploaded_files")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

SUPPORTED_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

# Session storage for image Q&A
vision_sessions: dict[str, dict] = {}


class VisionAnalysis(BaseModel):
    session_id: str
    caption: str
    objects: list[str]
    description: str
    mood: str
    colors: list[str]


class VisionQARequest(BaseModel):
    session_id: str
    question: str


class VisionQAResponse(BaseModel):
    answer: str
    session_id: str


def encode_image_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


import asyncio


def vision_chat(image_path: str, prompt: str, max_tokens: int = 700) -> str:
    """Call Groq vision model with an image and text prompt."""
    model = get_vision_model()

    ext = os.path.splitext(image_path)[1].lower()
    media_type_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"
    }
    media_type = media_type_map.get(ext, "image/jpeg")

    image_b64 = encode_image_to_base64(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_b64}"
                    }
                },
                {"type": "text", "text": prompt}
            ]
        }
    ]

    return chat_completion(
        messages=messages,
        model=model,
        temperature=0.4,
        max_tokens=max_tokens,
    )


async def async_vision_chat(image_path: str, prompt: str, max_tokens: int = 700) -> str:
    return await asyncio.to_thread(vision_chat, image_path, prompt, max_tokens)


@router.post("/analyze", response_model=VisionAnalysis)
async def analyze_image(file: UploadFile = File(...)):
    """Upload an image and get comprehensive visual analysis."""
    if file.content_type not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Use: JPEG, PNG, GIF, or WebP"
        )

    # Read and validate file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="Image too large. Maximum size is 10MB.")
    if len(content) < 100:
        raise HTTPException(status_code=400, detail="Image file is too small or corrupted.")

    session_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
    file_path = os.path.join(UPLOAD_DIR, f"vision_{session_id}{ext}")
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(content)

    # Store session info + persist image to MongoDB
    save_vision_image(session_id, content, ext)
    vision_sessions[session_id] = {
        "image_path": file_path,
        "filename": file.filename,
        "conversation": [],
    }
    save_session_data(session_id, "vision", {"filename": file.filename, "ext": ext})

    try:
        full_analysis = await async_vision_chat(
            file_path,
            """Analyze this image and respond with ONLY valid JSON (no markdown):
{
  "caption": "<one sentence caption>",
  "objects": ["<object1>", "<object2>"],
  "description": "<detailed 2-3 sentence description>",
  "mood": "<overall mood/atmosphere>",
  "colors": ["<dominant color 1>", "<color 2>"]
}""",
            max_tokens=500,
        )

        import json
        import re
        cleaned = full_analysis.strip()
        # Strip <think> blocks for reasoning models
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL).strip()
        # Strip markdown code blocks if present
        if "```" in cleaned:
            parts = cleaned.split("```")
            cleaned = parts[1] if len(parts) > 1 else cleaned
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            cleaned = cleaned[start:end]

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback if model doesn't return JSON (e.g. if vision is unsupported and it just returns text)
            raw_text = full_analysis.strip()
            # clean think tags for the description fallback
            raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
            data = {
                "caption": "Image analysis limited.",
                "objects": ["Various"],
                "description": raw_text if raw_text else "The model could not generate a valid JSON description.",
                "mood": "Neutral",
                "colors": ["Unknown"]
            }

        return VisionAnalysis(session_id=session_id, **data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")


async def _get_vision_session(session_id: str) -> dict | None:
    """Load vision session from memory or restore from MongoDB."""
    session = vision_sessions.get(session_id)
    if session:
        return session

    meta = load_session_data(session_id)
    if not meta:
        return None

    image_data = load_vision_image(session_id)
    if not image_data:
        return None

    image_bytes, ext = image_data
    file_path = os.path.join(UPLOAD_DIR, f"vision_{session_id}{ext}")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(image_bytes)

    session = {
        "image_path": file_path,
        "filename": meta.get("filename", "image"),
        "conversation": meta.get("conversation", []),
    }
    vision_sessions[session_id] = session
    return session


@router.post("/ask", response_model=VisionQAResponse)
async def ask_about_image(req: VisionQARequest):
    """Ask a question about a previously uploaded image."""
    session = await _get_vision_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Upload an image first.")

    image_path = session["image_path"]
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file not found. Please re-upload.")

    history_text = ""
    if session["conversation"]:
        history_text = "\n\nPrevious conversation:\n" + "\n".join(
            [f"Q: {c['q']}\nA: {c['a']}" for c in session["conversation"][-2:]]
        )

    try:
        answer = await async_vision_chat(
            image_path,
            f"Answer this question about the image: {req.question}{history_text}\n\nProvide a clear, concise answer.",
            max_tokens=500,
        )

        import re
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()

        session["conversation"].append({"q": req.question, "a": answer})
        save_session_data(req.session_id, "vision", {
            "filename": session.get("filename"),
            "ext": os.path.splitext(image_path)[1],
            "conversation": session["conversation"],
        })

        return VisionQAResponse(answer=answer, session_id=req.session_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def ask_about_image_stream(req: VisionQARequest):
    """Ask a question about a previously uploaded image and stream the response."""
    session = await _get_vision_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Upload an image first.")

    image_path = session["image_path"]
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file not found. Please re-upload.")

    history_text = ""
    if session["conversation"]:
        history_text = "\n\nPrevious conversation:\n" + "\n".join(
            [f"Q: {c['q']}\nA: {c['a']}" for c in session["conversation"][-2:]]
        )

    model = get_vision_model()
    ext = os.path.splitext(image_path)[1].lower()
    media_type_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"
    }
    media_type = media_type_map.get(ext, "image/jpeg")
    image_b64 = encode_image_to_base64(image_path)
    prompt = f"Answer this question about the image: {req.question}{history_text}\n\nProvide a clear, concise answer."

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_b64}"
                    }
                },
                {"type": "text", "text": prompt}
            ]
        }
    ]

    def stream_generator():
        full_answer = ""
        import re
        for chunk in chat_completion_stream(
            messages=messages,
            model=model,
            temperature=0.4,
            max_tokens=500,
        ):
            full_answer += chunk
            # Basic client-side <think> strip would be better, but we yield as is
            # except we don't know when a <think> tag ends in chunks easily.
            # We'll just yield chunks and let frontend handle it or it might just show it.
            # Actually, yielding chunks directly is fine, but to maintain exact compatibility:
            yield chunk

        cleaned_answer = re.sub(r'<think>.*?</think>', '', full_answer, flags=re.DOTALL).strip()
        
        session["conversation"].append({"q": req.question, "a": cleaned_answer})
        save_session_data(req.session_id, "vision", {
            "filename": session.get("filename"),
            "ext": os.path.splitext(image_path)[1],
            "conversation": session["conversation"],
        })

        import json
        metadata = {
            "session_id": req.session_id
        }
        yield f"\n__METADATA__::{json.dumps(metadata)}"

    return StreamingResponse(stream_generator(), media_type="text/plain")


@router.delete("/session/{session_id}")
async def clear_vision_session(session_id: str):
    """Clear vision session and delete image."""
    session = vision_sessions.pop(session_id, None)
    if session and os.path.exists(session.get("image_path", "")):
        os.remove(session["image_path"])
    return {"message": "Vision session cleared"}
