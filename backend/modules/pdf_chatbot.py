"""
Module 3: PDF Chatbot (RAG)
Upload PDFs, build vector store, chat with source citations
"""
import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from utils.llm import async_chat_completion, async_system_user_chat, get_model
from utils.pdf_loader import extract_text_with_pages, chunk_pages
from utils.embeddings import async_build_and_save_vector_store, async_similarity_search
from utils.session_store import save_session_data, load_session_data

router = APIRouter()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploaded_files")
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB

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
Always answer the user's Question directly using the document context below.
Cite page numbers when referencing content.
If the answer is not in the context, say you could not find it in the document.
Never ask the user to provide a question — a question is always included after "Question:"."""


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and create a vector store for RAG."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Read and validate file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 15MB.")

    session_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"chat_{session_id}.pdf")
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(content)

    try:
        pages = extract_text_with_pages(file_path)
        if not pages:
            raise HTTPException(status_code=400, detail="No text found in the PDF.")

        chunks = chunk_pages(pages)
        chunk_count = await async_build_and_save_vector_store(session_id, chunks)

        # Initialize chat session
        chat_sessions[session_id] = []
        save_session_data(session_id, "chatbot", {
            "filename": file.filename,
            "chunk_count": chunk_count,
            "history": [],
        })

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
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Please enter a question.")

    if req.session_id not in chat_sessions:
        stored = load_session_data(req.session_id)
        if isinstance(stored, dict):
            chat_sessions[req.session_id] = stored.get("history", [])
        elif isinstance(stored, list):
            chat_sessions[req.session_id] = stored
        else:
            chat_sessions[req.session_id] = []

    relevant_chunks = await async_similarity_search(req.session_id, req.message, top_k=4)

    if not relevant_chunks:
        # Try restoring session metadata from MongoDB
        meta = load_session_data(req.session_id)
        if meta:
            raise HTTPException(
                status_code=400,
                detail="Session expired due to server restart. Please re-upload your document."
            )
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
        if chunk["page"] not in seen_pages:
            sources.append({"page": chunk["page"], "score": round(chunk.get("score", 0), 3)})
            seen_pages.add(chunk["page"])

    context = "\n\n".join(context_parts)

    # Build conversation history (last 3 exchanges only for speed)
    history = chat_sessions.get(req.session_id, [])

    from utils.llm import get_chat_model
    from langchain_core.prompts import ChatPromptTemplate
    
    lc_messages = [("system", RAG_SYSTEM)]

    for exchange in history[-3:]:
        lc_messages.append(("user", exchange["question"]))
        lc_messages.append(("assistant", exchange["answer"]))

    lc_messages.append(("user", f"Context from the document:\n{context[:6000]}\n\nQuestion: {req.message}"))

    prompt = ChatPromptTemplate.from_messages(lc_messages)
    chat_model = get_chat_model(temperature=0.3, max_tokens=800)
    
    chain = prompt | chat_model
    result = await chain.ainvoke({})
    answer = result.content

    # Update session history
    if req.session_id not in chat_sessions:
        chat_sessions[req.session_id] = []
    chat_sessions[req.session_id].append({
        "question": req.message,
        "answer": answer,
    })

    stored = load_session_data(req.session_id)
    meta = stored if isinstance(stored, dict) else {}
    meta["history"] = chat_sessions[req.session_id]
    save_session_data(req.session_id, "chatbot", meta)

    return ChatResponse(
        answer=answer,
        sources=sorted(sources, key=lambda x: x["page"]),
        session_id=req.session_id,
    )


@router.get("/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session."""
    history = chat_sessions.get(session_id)
    if history is None:
        stored = load_session_data(session_id)
        if isinstance(stored, dict):
            history = stored.get("history", [])
        elif isinstance(stored, list):
            history = stored
        else:
            history = []
        chat_sessions[session_id] = history
    return {"session_id": session_id, "history": history}


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a chat session."""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return {"message": "Session cleared"}
