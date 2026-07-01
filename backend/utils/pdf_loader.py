"""
PDF Text Extraction Utility
Uses PyMuPDF for robust text extraction with page tracking
"""
import os
from pathlib import Path
import fitz  # PyMuPDF


def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF file."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()


def extract_text_with_pages(file_path: str) -> list[dict]:
    """Extract text per page with page numbers for citation."""
    doc = fitz.open(file_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({
                "page": i + 1,
                "text": text,
                "char_count": len(text),
            })
    doc.close()
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
