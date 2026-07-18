# AI Career & Research Assistant (Backend)

A powerful, multi-modal generative AI platform built entirely from scratch with FastAPI and Groq. This backend acts as the intelligent core for the AI Career & Research Assistant, processing PDFs, analyzing images, and managing high-speed conversational AI pipelines.

**Live Backend API:** [https://ai-assistant-backend-01.onrender.com](https://ai-assistant-backend-01.onrender.com)  
**Frontend Application:** [https://ai-assistant-gamma-sable.vercel.app](https://ai-assistant-gamma-sable.vercel.app)

---

## 🚀 Features & Modules

| # | Module | Description |
|---|--------|-------------|
| 1 | **General AI Chat** | Blazing-fast conversational agent using raw Groq streaming calls. |
| 2 | **Resume Reviewer** | Parses uploaded PDF resumes, performs ATS scoring, and returns structured JSON feedback. |
| 3 | **Interview Assistant** | Generates tailored mock interview questions and interactively grades user answers. |
| 4 | **PDF Chatbot (RAG)** | Uploads up to 3 PDFs (max 2MB each), chunks and embeds them, then answers context-aware questions via RAG. |
| 5 | **Research Assistant** | Supports multi-PDF cross-document analysis with the same RAG pipeline. |
| 6 | **Image Q&A (Vision)** | Uses Groq's multimodal vision model (`qwen/qwen3.6-27b`) to analyze uploaded images and answer questions about them. |
| 7 | **Chat History** | Persistent conversation history linked to user sessions via MongoDB. |
| 8 | **Voice** | Voice input/output module for hands-free interaction. |

---

## 🏗️ Architecture & Technology Stack

Every line of this backend was custom-written to ensure maximum performance and minimal overhead.

### Core Framework
- **FastAPI + Python** — Chosen for its extreme speed and native async/await capabilities, allowing the server to handle multiple simultaneous PDF uploads, LLM streams, and database queries without blocking.
- **Uvicorn (ASGI)** — Production-grade ASGI server for the FastAPI app.

### AI / LLM Engine
- **Groq Python SDK** — Official SDK for ultra-fast inference using Groq's LPU™ hardware.
- **Text Model:** `llama-3.1-8b-instant` for all general chat, PDF Q&A, resume review, and interview tasks.
- **Vision Model:** `qwen/qwen3.6-27b` for multimodal image analysis.
- **❌ NO LangChain** — We explicitly do not use LangChain. By writing custom wrapper functions (`utils/llm.py`), we achieved much faster response times, greater prompt control, and a significantly leaner architecture.

### Embeddings & Vector Store (Hybrid Architecture)
Instead of relying on expensive paid vector databases like Pinecone, we built a fully custom Retrieval-Augmented Generation (RAG) system:

- **FastEmbed (`BAAI/bge-small-en-v1.5`)** — Local ONNX-based embedding model. Runs entirely on CPU with `threads=1` to minimize memory usage on Render's free tier.
- **ChromaDB (Production)** — High-performance vector database with HNSW indexing for blazing-fast similarity search. Used automatically when installed (Linux/Render).
- **NumPy Fallback (Local Dev)** — If ChromaDB is not available (e.g., Windows without C++ Build Tools), the system automatically falls back to a pure NumPy cosine similarity engine. This ensures the backend works everywhere.
- **MongoDB Persistence** — Every vector store is simultaneously backed up to MongoDB. If the Render server restarts and wipes its ephemeral disk, the ChromaDB collection is instantly re-hydrated from MongoDB on the next request — zero data loss.

### Database
- **MongoDB Atlas (PyMongo)** — Stores all chat history, extracted vector embeddings, session metadata, and vision session images. Survives Render server restarts.

### Document Processing
- **pypdf** — Lightweight, pure-Python library for extracting text from PDFs (resumes, research papers).

### Memory Optimizations (Render Free Tier)
The following optimizations were made to keep the backend within Render's 512MB free-tier memory limit:
- Removed `scikit-learn` — Replaced with a pure NumPy cosine similarity implementation (saving ~100MB RAM).
- Removed `PyMuPDF` — Was installed but unused; `pypdf` was already handling all PDF extraction.
- FastEmbed `threads=1` — Restricts ONNX Runtime's thread pool from spawning multiple threads (dramatically reduces memory overhead).
- FastFail on API errors — The LLM client is configured to immediately stop retrying on `400`/`401`/`403`/`404` errors, preventing runaway retries that waste memory and CPU.

---

## 📁 Project Structure

```
backend/
├── main.py                  # FastAPI app, lifespan startup, CORS, route registration
├── modules/
│   ├── general_chat.py      # General conversational AI
│   ├── resume_reviewer.py   # Resume parsing & ATS scoring
│   ├── interview_assistant.py # Mock interview generator
│   ├── pdf_chatbot.py       # PDF upload + RAG Q&A
│   ├── research_assistant.py # Multi-PDF research analysis
│   ├── image_captioning.py  # Image Q&A via vision model
│   ├── history.py           # Chat history management
│   └── voice.py             # Voice input/output
└── utils/
    ├── llm.py               # Groq SDK wrapper with retry & streaming logic
    ├── embeddings.py        # Hybrid ChromaDB/NumPy vector store engine
    ├── pdf_loader.py        # PDF text extraction & chunking
    ├── session_store.py     # MongoDB persistence layer
    ├── json_helpers.py      # Structured JSON output helpers
    └── tools.py             # Shared utility functions
```

---

## ⚙️ Local Setup & Installation

### Prerequisites
- Python 3.10+
- A [Groq API Key](https://console.groq.com/)
- A [MongoDB Atlas](https://www.mongodb.com/products/platform/atlas-database) connection string

### 1. Clone the Repository
```bash
git clone https://github.com/HarshPariya/AI-Assistant-Backend.git
cd AI-Assistant-Backend
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note for Windows users:** `chromadb` requires [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) to install. If you don't have them, the backend will automatically fall back to the NumPy vector store — no action required.

### 4. Environment Variables
Create a `.env` file in the root directory (never commit this file):
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
GROQ_VISION_MODEL=qwen/qwen3.6-27b
MONGODB_URL=your_mongodb_connection_string
FRONTEND_URL=http://localhost:3000
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
```

### 5. Run the Application
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Run System Checks
A comprehensive test script is included to verify all modules, APIs, and connections:
```bash
python test_all.py
```
This will test:
- Environment variables ✅
- Groq Text API ✅
- Groq Vision API ✅
- PDF Loading & Chunking ✅
- Embedding Model ✅
- MongoDB Connection ✅
- All 8 module imports ✅
- End-to-end General Chat ✅
- End-to-end PDF RAG Pipeline ✅

---

## 🔒 Security

- The `.env` file is listed in `.gitignore` and is **never committed** to this repository.
- API keys are loaded exclusively via `python-dotenv` at runtime.
- PDF uploads are limited to **3 files** and **2MB per file** to prevent abuse.

---

## 🚢 Deployment (Render)

This backend is deployed on [Render.com](https://render.com) as a Web Service.

**Key Render settings:**
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Environment Variables:** Set all variables from the `.env` section above in Render's dashboard.

Since Render's free tier uses an **ephemeral filesystem** (disk is wiped on every restart), all vector store data is automatically backed up to MongoDB. On restart, the ChromaDB collection is rebuilt from MongoDB on the next user request — completely transparently.

---

## 📊 System Optimizations Summary

| Optimization | Impact |
|---|---|
| Removed `scikit-learn` | Saves ~100MB RAM |
| Removed `PyMuPDF` | Reduces install size & build time |
| FastEmbed `threads=1` | Prevents ONNX RAM spike on startup |
| ChromaDB → NumPy fallback | Works everywhere, no C++ tools needed locally |
| MongoDB vector backup | Zero data loss across Render restarts |
| Fast-fail on 4xx errors | Prevents runaway retries wasting CPU/RAM |
| Disabled OCR fallback in PDF loader | Eliminates heavy image processing timeouts |

---

## 👨‍💻 Developer

Developed by **Harsh Pariya**.  
This backend was meticulously engineered from scratch to prioritize speed, stability, and efficiency. By avoiding heavy wrapper libraries like LangChain, building a hybrid RAG pipeline (ChromaDB + NumPy), and implementing aggressive memory optimizations, the system demonstrates deep, low-level integration with modern AI technologies — all running on a free-tier cloud instance.
