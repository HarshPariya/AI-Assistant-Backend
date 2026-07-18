"""
PDF Text Extraction Utility
Uses pypdf for pure-Python robust text extraction with page tracking,
avoiding C-extension crashes in serverless deployments.
"""
import os
from pathlib import Path
from pypdf import PdfReader


def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    
    if not text.strip():
        return "[System Note: This PDF contains no extractable text. It might be a scanned document or image-based PDF. Inform the user that you cannot read its contents and ask them if they have a text-based version.]"
    return text.strip()


def extract_text_with_pages(file_path: str) -> list[dict]:
    """Extract text per page with page numbers for citation."""
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            text = text.strip()
            if text:
                pages.append({
                    "page": i + 1,
                    "text": text,
                    "char_count": len(text),
                })
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
