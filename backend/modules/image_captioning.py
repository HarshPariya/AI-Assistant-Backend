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
from utils.llm import get_groq_client, get_vision_model

router = APIRouter()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploaded_files")

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


def vision_chat(image_path: str, prompt: str) -> str:
    """Call Groq vision model with an image and text prompt."""
    client = get_groq_client()
    model = get_vision_model()
    
    # Detect media type
    ext = os.path.splitext(image_path)[1].lower()
    media_type_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
    media_type = media_type_map.get(ext, "image/jpeg")
    
    image_b64 = encode_image_to_base64(image_path)
    
    from utils.llm import chat_completion
    
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
        max_tokens=1024,
    )


@router.post("/analyze", response_model=VisionAnalysis)
async def analyze_image(file: UploadFile = File(...)):
    """Upload an image and get comprehensive visual analysis."""
    if file.content_type not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Use: {', '.join(SUPPORTED_TYPES)}"
        )
    
    session_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    file_path = os.path.join(UPLOAD_DIR, f"vision_{session_id}{ext}")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Store session info (keep the image for Q&A)
    vision_sessions[session_id] = {
        "image_path": file_path,
        "filename": file.filename,
        "conversation": [],
    }
    
    try:
        full_analysis = vision_chat(
            file_path,
            """Analyze this image comprehensively and respond with ONLY valid JSON:
{
  "caption": "<one sentence caption>",
  "objects": ["<object1>", "<object2>", ...],
  "description": "<detailed 2-3 sentence description>",
  "mood": "<overall mood/atmosphere>",
  "colors": ["<dominant color 1>", "<color 2>", ...]
}"""
        )
        
        # Parse response
        import json
        cleaned = full_analysis.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        
        data = json.loads(cleaned)
        return VisionAnalysis(session_id=session_id, **data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")


@router.post("/ask", response_model=VisionQAResponse)
async def ask_about_image(req: VisionQARequest):
    """Ask a question about a previously uploaded image."""
    session = vision_sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Upload an image first.")
    
    image_path = session["image_path"]
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file not found.")
    
    # Build conversation context
    history_text = ""
    if session["conversation"]:
        history_text = "\n\nPrevious conversation:\n" + "\n".join(
            [f"Q: {c['q']}\nA: {c['a']}" for c in session["conversation"][-3:]]
        )
    
    try:
        answer = vision_chat(
            image_path,
            f"Answer this question about the image: {req.question}{history_text}\n\nProvide a clear, detailed answer."
        )
        
        # Update conversation
        session["conversation"].append({"q": req.question, "a": answer})
        
        return VisionQAResponse(answer=answer, session_id=req.session_id)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def clear_vision_session(session_id: str):
    """Clear vision session and delete image."""
    session = vision_sessions.pop(session_id, None)
    if session and os.path.exists(session.get("image_path", "")):
        os.remove(session["image_path"])
    return {"message": "Vision session cleared"}
