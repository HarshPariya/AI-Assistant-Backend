"""
Module: General Chat & Image Generation
Provides a ChatGPT-like interface, text-to-image capabilities, and handles PDF/Image uploads.
"""
import json
import base64
import os
import re
import tempfile
from urllib.parse import quote, unquote
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from utils.llm import get_model, get_vision_model, async_chat_completion
from utils.pdf_loader import extract_text_with_pages
from typing import Optional, List

router = APIRouter()

# 10 MB file size limit
MAX_FILE_SIZE = 10 * 1024 * 1024

IMAGE_REQUEST_PATTERN = re.compile(
    r"\b(generate|create|draw|make|design|show|render)\b.{0,40}\b(image|picture|photo|illustration|art|portrait|logo)\b",
    re.IGNORECASE,
)


def _is_image_generation_request(text: str) -> bool:
    return bool(IMAGE_REQUEST_PATTERN.search(text or ""))


def _build_image_response(user_text: str) -> str:
    """Build a reliable pollinations.ai markdown image from the user's prompt."""
    prompt = user_text.strip()
    for prefix in ("generate an image of", "create an image of", "draw an image of",
                     "make an image of", "generate a picture of", "create a picture of",
                     "draw a picture of", "generate image of", "create image of"):
        if prompt.lower().startswith(prefix):
            prompt = prompt[len(prefix):].strip()
            break
    encoded = quote(prompt, safe="")
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true"
    alt = prompt[:80] or "Generated image"
    return f"Here is your image:\n\n![{alt}]({url})\n\nEnjoy!"


def _fix_pollinations_urls(text: str) -> str:
    """Re-encode pollinations image URLs so markdown images load correctly."""

    def _fix_url(match: re.Match) -> str:
        prefix, url = match.group(1), match.group(2)
        prompt_match = re.search(r"prompt/([^?]+)", url)
        if not prompt_match:
            return match.group(0)
        prompt = unquote(prompt_match.group(1))
        encoded = quote(prompt, safe="")
        fixed = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true"
        return f"{prefix}{fixed})"

    return re.sub(r"(!\[[^\]]*\]\()([^)]*pollinations\.ai[^)]+)\)", _fix_url, text)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


class ChatResponse(BaseModel):
    answer: str


SYSTEM_PROMPT = """You are a highly capable, friendly AI assistant.
You can answer any general questions, help with coding, writing, and analysis.

If the user uploads a document (PDF) or Image, you will be provided with the text extracted from it or the image itself. Analyze it and answer the user's question based on it.

IMPORTANT - IMAGE GENERATION:
If the user asks you to "generate an image", "draw", "create a picture", or anything similar, you MUST do exactly the following:
1. Come up with a detailed, highly descriptive prompt for the image.
2. Return a markdown image link using this format:
   ![Image Description](https://image.pollinations.ai/prompt/YOUR_URL_ENCODED_PROMPT?width=1024&height=1024&nologo=true)
3. Provide a brief friendly message below the image.

Example:
User: Generate an image of a cyberpunk city
Assistant: Here is your image:
![Cyberpunk City](https://image.pollinations.ai/prompt/A%20futuristic%20cyberpunk%20city%20at%20night%20with%20neon%20lights%20and%20flying%20cars?width=1024&height=1024&nologo=true)
Enjoy the futuristic vibes!

Always URL encode the prompt in the URL. Never tell the user you cannot generate images."""


@router.post("/ask", response_model=ChatResponse)
async def general_chat(
    messages: str = Form(...),
    file: Optional[UploadFile] = File(None),
):
    """General chat endpoint with image generation and multimodal file capability."""

    # 1. Parse Messages
    try:
        raw_msgs = json.loads(messages)
        parsed_messages = [ChatMessage(**m) for m in raw_msgs]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid messages JSON")

    if not parsed_messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    last_msg = parsed_messages[-1]

    # Fast path: image generation requests (no LLM round-trip, reliable URLs)
    if not file and _is_image_generation_request(last_msg.content):
        return ChatResponse(answer=_build_image_response(last_msg.content))

    # 2. Build Chat Context
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    model_to_use = get_model()  # default text model

    # Process history (except the very last message)
    for msg in parsed_messages[:-1]:
        api_messages.append({"role": msg.role, "content": msg.content})

    # last_msg already set above

    # 3. Handle File Attachment
    if file and file.filename:
        content_type = file.content_type or ""

        # Validate file size
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")

        # --- PDF Handling ---
        if content_type == "application/pdf" or file.filename.lower().endswith(".pdf"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    temp_path = tmp.name
                    tmp.write(file_content)

                extracted = extract_text_with_pages(temp_path)
                pdf_text = "\n".join([f"--- Page {p['page']} ---\n{p['text']}" for p in extracted])

                if os.path.exists(temp_path):
                    os.remove(temp_path)

                # Limit PDF context to avoid token overflow
                enhanced_prompt = f"{last_msg.content}\n\n[ATTACHED PDF DOCUMENT TEXT]:\n{pdf_text[:20000]}"
                api_messages.append({"role": last_msg.role, "content": enhanced_prompt})

            except Exception as e:
                import traceback
                with open("backend_error.log", "a") as f:
                    f.write(f"\n--- PDF ERROR ---\n")
                    traceback.print_exc(file=f)
                raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

        # --- Image Handling ---
        elif content_type.startswith("image/"):
            try:
                # Validate minimum image size (must be >2 pixels)
                if len(file_content) < 100:
                    raise HTTPException(status_code=400, detail="Image file is too small or corrupted.")

                base64_image = base64.b64encode(file_content).decode("utf-8")

                # Switch to vision model
                model_to_use = get_vision_model()

                # Groq vision payload format
                user_text = last_msg.content or "Describe this image in detail."
                api_messages.append({
                    "role": last_msg.role,
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{content_type};base64,{base64_image}"
                            }
                        }
                    ]
                })
            except HTTPException:
                raise
            except Exception as e:
                import traceback
                with open("backend_error.log", "a") as f:
                    f.write(f"\n--- IMAGE ERROR ---\n")
                    traceback.print_exc(file=f)
                raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF or Image.")
    else:
        # No file, just standard text
        api_messages.append({"role": last_msg.role, "content": last_msg.content})

    # 4. Call Groq — lower max_tokens for text, higher for vision analysis
    is_vision = model_to_use == get_vision_model()
    max_tok = 1024 if not is_vision else 800

    try:
        answer = await async_chat_completion(
            messages=api_messages,
            model=model_to_use,
            temperature=0.7,
            max_tokens=max_tok,
        )
        return ChatResponse(answer=_fix_pollinations_urls(answer))

    except Exception as e:
        import traceback
        try:
            with open("backend_error.log", "a") as f:
                f.write(f"\n--- ERROR ---\n")
                traceback.print_exc(file=f)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))
