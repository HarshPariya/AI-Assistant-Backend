"""
Module 3: PDF Chatbot (RAG)
Upload PDFs, build vector store, chat with source citations
"""
import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from utils.llm import chat_completion, get_model
from utils.pdf_loader import extract_text_with_pages, chunk_pages
from utils.embeddings import build_and_save_vector_store, similarity_search

router = APIRouter()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploaded_files")

# In-memory session chat history
chat_sessions: dict[str, list[dict]] = {}


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    chunk_count: int
    message: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str


RAG_SYSTEM = """You are a helpful AI assistant that answers questions based on the provided document context.
Always be accurate and cite the page numbers when referencing content.
If the answer is not in the context, say so clearly rather than making things up.
Be concise but thorough."""


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and create a vector store for RAG."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    session_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"chat_{session_id}.pdf")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    try:
        pages = extract_text_with_pages(file_path)
        if not pages:
            raise HTTPException(status_code=400, detail="No text found in the PDF.")
        
        chunks = chunk_pages(pages)
        chunk_count = build_and_save_vector_store(session_id, chunks)
        
        # Initialize chat session
        chat_sessions[session_id] = []
        
        return UploadResponse(
            session_id=session_id,
            filename=file.filename,
            chunk_count=chunk_count,
            message=f"Successfully processed '{file.filename}' into {chunk_count} searchable chunks.",
        )
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@router.post("/ask", response_model=ChatResponse)
async def ask_question(req: ChatRequest):
    """Ask a question about the uploaded PDF using RAG."""
    try:
        relevant_chunks = similarity_search(req.session_id, req.message, top_k=5)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail="Session expired due to server sleep. Please re-upload your document."
        )
    
    if not relevant_chunks:
        raise HTTPException(
            status_code=404,
            detail="No documents found for this session. Please upload a PDF first."
        )
    
    # Build context from chunks
    context_parts = []
    sources = []
    seen_pages = set()
    
    for chunk in relevant_chunks:
        context_parts.append(f"[Page {chunk['page']}]: {chunk['text']}")
        if chunk['page'] not in seen_pages:
            sources.append({"page": chunk['page'], "score": round(chunk.get('score', 0), 3)})
            seen_pages.add(chunk['page'])
    
    context = "\n\n".join(context_parts)
    
    # Build conversation history
    history = chat_sessions.get(req.session_id, [])
    
    messages = [
        {"role": "system", "content": RAG_SYSTEM},
    ]
    
    # Add last 4 exchanges for context
    for exchange in history[-4:]:
        messages.append({"role": "user", "content": exchange["question"]})
        messages.append({"role": "assistant", "content": exchange["answer"]})
    
    messages.append({
        "role": "user",
        "content": f"Context from the document:\n{context}\n\nQuestion: {req.message}"
    })
    
    from utils.llm import get_groq_client
    client = get_groq_client()
    response = client.chat.completions.create(
        model=get_model(),
        messages=messages,
        temperature=0.3,
        max_tokens=1500,
    )
    answer = response.choices[0].message.content
    
    # Update session history
    if req.session_id not in chat_sessions:
        chat_sessions[req.session_id] = []
    chat_sessions[req.session_id].append({
        "question": req.message,
        "answer": answer,
    })
    
    return ChatResponse(
        answer=answer,
        sources=sorted(sources, key=lambda x: x["page"]),
        session_id=req.session_id,
    )


@router.get("/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session."""
    history = chat_sessions.get(session_id, [])
    return {"session_id": session_id, "history": history}


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a chat session."""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return {"message": "Session cleared"}
