"""
PDF Text Extraction Utility
Uses pypdf for pure-Python robust text extraction with page tracking,
avoiding C-extension crashes in serverless deployments.
Includes OCR fallback for image-only PDFs using Groq Vision API.
"""
import os
import io
import base64
from pathlib import Path
from pypdf import PdfReader
from pypdf.errors import PdfStreamError
from PIL import Image

def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    """Uses Groq Vision API to extract text from an image."""
    try:
        from groq import Groq
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.thumbnail((1024, 1024))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
        response = client.chat.completions.create(
            model='qwen/qwen3.6-27b',
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'Extract all text from this image exactly as written. If there is no text, briefly describe the image. Return only the extracted text or description.'},
                        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64_str}'}}
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=1024
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""


def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF file."""
    try:
        reader = PdfReader(file_path)
    except Exception as e:
        print(f"PdfReader init failed for {file_path}: {e}")
        return ""
    text = ""
    for page in reader.pages:
        try:
            t = page.extract_text()
            if t:
                text += t + "\n"
            else:
                # OCR Fallback for the page
                for image_file_object in page.images:
                    ocr_text = extract_text_from_image_bytes(image_file_object.data)
                    if ocr_text:
                        text += ocr_text + "\n"
        except Exception:
            continue
    
    if not text.strip():
        return "[System Note: This PDF contains no extractable text. It might be a scanned document or image-based PDF. Inform the user that you cannot read its contents and ask them if they have a text-based version.]"
    return text.strip()


def extract_text_with_pages(file_path: str) -> list[dict]:
    """Extract text per page with page numbers for citation."""
    try:
        reader = PdfReader(file_path)
    except Exception as e:
        print(f"PdfReader init failed for {file_path}: {e}")
        return []
    pages = []
    
    # Process max 10 pages for OCR to avoid massive rate limits on Groq
    ocr_count = 0
    
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
            if not text or not text.strip():
                # Attempt OCR if no text found on page
                if ocr_count < 3:
                    for img_obj in page.images:
                        ocr_res = extract_text_from_image_bytes(img_obj.data)
                        if ocr_res:
                            text = (text or "") + "\n" + ocr_res
                    if text and text.strip():
                        ocr_count += 1
                        
            if text and text.strip():
                pages.append({
                    "page": i + 1,
                    "text": text.strip(),
                    "char_count": len(text.strip()),
                })
        except Exception:
            continue
            
    if not pages:
        pages.append({
            "page": 1,
            "text": "[System Note: This PDF contains no extractable text. It might be a scanned document or image-based PDF. Inform the user that you cannot read its contents and ask them if they have a text-based version.]",
            "char_count": 200,
        })
    return pages


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def chunk_pages(pages: list[dict], chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    """Chunk page-tracked content for RAG with citation support."""
    chunks = []
    for page_data in pages:
        text = page_data["text"]
        page_num = page_data["page"]
        sub_chunks = chunk_text(text, chunk_size, overlap)
        for i, chunk in enumerate(sub_chunks):
            chunks.append({
                "text": chunk,
                "page": page_num,
                "chunk_id": f"page{page_num}_chunk{i}",
            })
    return chunks
