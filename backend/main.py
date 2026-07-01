"""
AI Career & Research Assistant — FastAPI Backend
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Force PyTorch and disable TensorFlow in transformers to avoid tf-keras Python 3.13 issues
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load environment variables
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

app = FastAPI(
    title="AI Career & Research Assistant API",
    description="Multi-Modal GenAI Platform with Groq LLM",
    version="1.0.0",
    on_startup=[init_db],
)

# CORS - allow Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
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
    return {
        "status": "ok",
        "message": "AI Career & Research Assistant API is running",
        "version": "1.0.0",
    }


@app.get("/")
async def root():
    return {"message": "Welcome to AI Career & Research Assistant API"}
