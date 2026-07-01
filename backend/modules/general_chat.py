"""
Module: General Chat & Image Generation
Provides a ChatGPT-like interface, text-to-image capabilities, and handles PDF/Image uploads.
"""
import json
import base64
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from utils.llm import get_groq_client, get_model, get_vision_model, chat_completion
from utils.pdf_loader import extract_text_with_pages
from typing import Optional, List

router = APIRouter()


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

    # 2. Build Chat Context
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    model_to_use = get_model() # default text model: llama-3.3-70b-versatile

    # Process history (except the very last message)
    for msg in parsed_messages[:-1]:
        api_messages.append({"role": msg.role, "content": msg.content})

    # The last message is the current one we need to potentially attach the file to
    last_msg = parsed_messages[-1]

    # 3. Handle File Attachment
    if file:
        content_type = file.content_type
        
        # --- PDF Handling ---
        if content_type == "application/pdf":
            try:
                import tempfile
                import shutil
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    temp_path = tmp.name
                    shutil.copyfileobj(file.file, tmp)
                
                # Extract text
                extracted = extract_text_with_pages(temp_path)
                pdf_text = "\n".join([f"--- Page {p['page']} ---\n{p['text']}" for p in extracted])
                
                # Cleanup
                import os
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                # Inject PDF text into the user's prompt
                enhanced_prompt = f"{last_msg.content}\n\n[ATTACHED PDF DOCUMENT TEXT]:\n{pdf_text[:30000]}" # limit to avoid blowing up context
                api_messages.append({"role": last_msg.role, "content": enhanced_prompt})

            except Exception as e:
                import traceback
                with open("backend_error.log", "a") as f:
                    f.write(f"\n--- PDF ERROR ---\n")
                    traceback.print_exc(file=f)
                raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

        # --- Image Handling ---
        elif content_type and content_type.startswith("image/"):
            try:
                # Groq Vision requires base64
                image_bytes = await file.read()
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                
                # Switch to vision model
                model_to_use = get_vision_model()

                # Groq vision payload format
                api_messages.append({
                    "role": last_msg.role,
                    "content": [
                        {"type": "text", "text": last_msg.content},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{content_type};base64,{base64_image}"
                            }
                        }
                    ]
                })
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

    # 4. Call Groq with retry logic
    try:
        answer = chat_completion(
            messages=api_messages,
            model=model_to_use,
            temperature=0.7,
            max_tokens=1024,  # Reduced for speed - still plenty for most answers
        )
        return ChatResponse(answer=answer)
        
    except Exception as e:
        import traceback, os as _os
        try:
            with open("backend_error.log", "a") as f:
                f.write(f"\n--- ERROR ---\n")
                traceback.print_exc(file=f)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))
