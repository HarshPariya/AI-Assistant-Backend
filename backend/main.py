"""
AI Career & Research Assistant — FastAPI Backend
"""
import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Force PyTorch and disable TensorFlow in transformers to avoid tf-keras Python 3.13 issues
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables FIRST before importing modules
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
load_dotenv()

# Create required directories
os.makedirs("uploaded_files", exist_ok=True)
os.makedirs("vector_store", exist_ok=True)

from modules.resume_reviewer import router as resume_router
from modules.interview_assistant import router as interview_router
from modules.pdf_chatbot import router as chatbot_router
from modules.research_assistant import router as research_router
from modules.image_captioning import router as vision_router
from modules.general_chat import router as general_chat_router
from modules.history import router as history_router, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: warm up Groq client, embeddings, and MongoDB on boot."""
    try:
        init_db()
        from utils.llm import get_groq_client
        get_groq_client()
        from utils.embeddings import preload_embedding_model
        await asyncio.to_thread(preload_embedding_model)
        from utils.session_store import is_available
        if is_available():
            print("✅ MongoDB connected.")
        else:
            print("⚠️  MongoDB not available — history and session persistence disabled.")
        print("✅ Groq client and embeddings initialized.")
    except Exception as e:
        print(f"⚠️  Startup warning: {e}")
    yield
    # Cleanup (if needed) goes here


app = FastAPI(
    title="AI Career & Research Assistant API",
    description="Multi-Modal GenAI Platform with Groq LLM",
    version="1.1.0",
    lifespan=lifespan,
)

# CORS - allow Next.js frontend and deployed URLs
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://ai-assistant-gamma-sable.vercel.app",
]

# Also allow any env-configured frontend URL
frontend_url = os.getenv("FRONTEND_URL", "")
if frontend_url and frontend_url not in allowed_origins:
    allowed_origins.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register module routers
app.include_router(resume_router, prefix="/resume", tags=["Resume Reviewer"])
app.include_router(interview_router, prefix="/interview", tags=["Interview Assistant"])
app.include_router(chatbot_router, prefix="/chat", tags=["PDF Chatbot"])
app.include_router(research_router, prefix="/research", tags=["Research Assistant"])
app.include_router(vision_router, prefix="/vision", tags=["Image Captioning"])
app.include_router(general_chat_router, prefix="/general", tags=["General Chat"])
app.include_router(history_router, prefix="/history", tags=["Chat History"])


@app.get("/health")
async def health_check():
    from utils.session_store import is_available
    return {
        "status": "ok",
        "message": "AI Career & Research Assistant API is running",
        "version": "1.2.0",
        "mongodb": is_available(),
    }


@app.get("/")
async def root():
    return {"message": "Welcome to AI Career & Research Assistant API"}
