"""
Module 4: Multi-PDF Research Assistant
Upload multiple PDFs, summarize, compare, generate study notes
"""
import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
from utils.llm import system_user_chat, chat_completion, get_model
from utils.pdf_loader import extract_text_from_pdf, extract_text_with_pages, chunk_pages
from utils.embeddings import build_and_save_vector_store, similarity_search

router = APIRouter()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploaded_files")

# Session storage: session_id -> list of doc info
research_sessions: dict[str, list[dict]] = {}


class ResearchSession(BaseModel):
    session_id: str
    documents: list[dict]
    total_chunks: int


class ResearchActionRequest(BaseModel):
    session_id: str
    action: str  # summarize_all | compare | study_notes | key_takeaways | ask
    query: Optional[str] = None


class ResearchResponse(BaseModel):
    result: str
    action: str
    session_id: str


RESEARCH_SYSTEM = """You are an expert research assistant and academic analyst.
You help researchers understand, compare, and synthesize information from multiple documents.
Be thorough, well-structured, and academically rigorous."""


@router.post("/upload", response_model=ResearchSession)
async def upload_research_pdfs(files: list[UploadFile] = File(...)):
    """Upload multiple PDFs for research analysis."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 PDFs allowed.")
    
    session_id = str(uuid.uuid4())
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    all_chunks = []
    documents = []
    
    for i, file in enumerate(files):
        if not file.filename.lower().endswith(".pdf"):
            continue
        
        file_path = os.path.join(UPLOAD_DIR, f"research_{session_id}_{i}.pdf")
        content = await file.read()
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        try:
            pages = extract_text_with_pages(file_path)
            full_text = extract_text_from_pdf(file_path)
            
            doc_chunks = chunk_pages(pages)
            # Tag chunks with document info
            for chunk in doc_chunks:
                chunk["doc_id"] = i
                chunk["doc_name"] = file.filename
            
            all_chunks.extend(doc_chunks)
            documents.append({
                "doc_id": i,
                "filename": file.filename,
                "page_count": len(pages),
                "word_count": len(full_text.split()),
                "chunk_count": len(doc_chunks),
            })
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    if not all_chunks:
        raise HTTPException(status_code=400, detail="No readable text found in uploaded PDFs.")
    
    total_chunks = build_and_save_vector_store(session_id, all_chunks)
    research_sessions[session_id] = documents
    
    return ResearchSession(
        session_id=session_id,
        documents=documents,
        total_chunks=total_chunks,
    )


@router.post("/action", response_model=ResearchResponse)
async def perform_research_action(req: ResearchActionRequest):
    """Perform a research action (summarize, compare, notes, takeaways, ask)."""
    documents = research_sessions.get(req.session_id)
    if not documents:
        raise HTTPException(status_code=404, detail="Session not found. Upload PDFs first.")
    
    doc_names = [d["filename"] for d in documents]
    
    if req.action == "ask" and req.query:
        # RAG-based question answering
        chunks = similarity_search(req.session_id, req.query, top_k=6)
        context = "\n\n".join([f"[{c['doc_name']} - Page {c['page']}]: {c['text']}" for c in chunks])
        
        result = system_user_chat(
            system_prompt=RESEARCH_SYSTEM,
            user_message=f"Documents: {', '.join(doc_names)}\n\nContext:\n{context}\n\nQuestion: {req.query}",
            temperature=0.3,
            max_tokens=1500,
        )
    
    elif req.action == "summarize_all":
        # Get broad sample from each document
        all_context = []
        for doc in documents:
            chunks = similarity_search(req.session_id, f"main topics overview introduction conclusion {doc['filename']}", top_k=3)
            for c in chunks:
                if c.get("doc_id") == doc["doc_id"]:
                    all_context.append(f"[{doc['filename']}]: {c['text']}")
        
        context = "\n\n".join(all_context[:15])
        result = system_user_chat(
            system_prompt=RESEARCH_SYSTEM,
            user_message=f"Summarize these research documents:\nDocuments: {', '.join(doc_names)}\n\nContent samples:\n{context}\n\nProvide a structured summary for each document (2-3 paragraphs each) followed by an overall synthesis.",
            temperature=0.4,
            max_tokens=2500,
        )
    
    elif req.action == "compare":
        chunks = similarity_search(req.session_id, "methodology findings results conclusions comparison", top_k=8)
        context = "\n\n".join([f"[{c.get('doc_name','Doc')} - Page {c['page']}]: {c['text']}" for c in chunks])
        result = system_user_chat(
            system_prompt=RESEARCH_SYSTEM,
            user_message=f"Compare these documents: {', '.join(doc_names)}\n\nContent:\n{context}\n\nCreate a detailed comparison covering: methodology, findings, conclusions, similarities, and differences. Use a structured format with headers.",
            temperature=0.4,
            max_tokens=2500,
        )
    
    elif req.action == "study_notes":
        chunks = similarity_search(req.session_id, "key concepts definitions important terms methodology", top_k=10)
        context = "\n\n".join([f"[{c.get('doc_name','Doc')} - Page {c['page']}]: {c['text']}" for c in chunks])
        result = system_user_chat(
            system_prompt=RESEARCH_SYSTEM,
            user_message=f"Create comprehensive study notes from these documents: {', '.join(doc_names)}\n\nContent:\n{context}\n\nFormat as: Key Concepts, Important Definitions, Main Arguments, Critical Points, Questions for Further Study.",
            temperature=0.4,
            max_tokens=2500,
        )
    
    elif req.action == "key_takeaways":
        chunks = similarity_search(req.session_id, "conclusion findings results implications", top_k=8)
        context = "\n\n".join([f"[{c.get('doc_name','Doc')} - Page {c['page']}]: {c['text']}" for c in chunks])
        result = system_user_chat(
            system_prompt=RESEARCH_SYSTEM,
            user_message=f"Extract key takeaways from: {', '.join(doc_names)}\n\nContent:\n{context}\n\nList the 10-15 most important takeaways in bullet points, organized by theme/document.",
            temperature=0.3,
            max_tokens=1500,
        )
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}")
    
    return ResearchResponse(
        result=result,
        action=req.action,
        session_id=req.session_id,
    )


@router.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get info about a research session."""
    docs = research_sessions.get(session_id, [])
    return {"session_id": session_id, "documents": docs}
