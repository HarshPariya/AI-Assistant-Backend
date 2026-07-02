# AI Career & Research Assistant (Backend)

A powerful, multi-modal generative AI platform built entirely from scratch with FastAPI and Groq. This backend acts as the intelligent core for the AI Career & Research Assistant, processing PDFs, analyzing images, and managing high-speed conversational AI pipelines.

**Live Backend API URL:** [https://ai-assistant-backend-01.onrender.com](https://ai-assistant-backend-01.onrender.com)  
**Frontend Application:** [https://ai-assistant-gamma-sable.vercel.app](https://ai-assistant-gamma-sable.vercel.app)

---

## 🏗️ How It Was Built & Technology Breakdown

Every single line of this backend was custom-written to ensure maximum performance and minimal overhead. Here is exactly what is used and why:

- **Core Framework (FastAPI & Python):** We chose FastAPI for its extreme speed and native asynchronous capabilities (`async`/`await`), allowing the server to handle multiple PDF uploads and LLM streams simultaneously without blocking.
- **LLM Engine (Groq Python SDK):** We use the official `groq` Python SDK to communicate with Llama 3 models (and Llama Vision for images). 
- **NO LangChain:** **We explicitly DO NOT use LangChain.** While LangChain is popular, it adds unnecessary bloat, abstraction, and latency. By writing custom wrapper functions (in `utils/llm.py`), we achieved much faster response times, greater control over prompts, and a leaner architecture.
- **Embeddings & Vector Store:** Instead of relying on paid vector databases like Pinecone, we built a fully custom, lightweight Retrieval-Augmented Generation (RAG) system:
  - Text chunking is done manually.
  - Embeddings are generated locally on the server using `sentence-transformers` (`all-MiniLM-L6-v2`) via Hugging Face.
  - Vector similarity search is calculated in milliseconds using standard matrix multiplication via `NumPy` and `scikit-learn` (`cosine_similarity`).
- **Database (MongoDB Atlas):** We use PyMongo to connect to MongoDB Atlas. This stores all user chat history, extracted vector embeddings, and session data, so nothing is lost when the Render server spins down or restarts.
- **Document Processing:** We use `PyMuPDF` (`fitz`) because it is the absolute fastest and most reliable library for extracting text from dense PDF files (like resumes and research papers).

---

## 🚀 Features & Modules

1. **General AI Chat:** Blazing-fast conversational agent using raw Groq calls.
2. **Resume Reviewer:** Parses PDF text and strictly coerces the LLM to output structured JSON for ATS scoring and feedback.
3. **Interview Assistant:** Generates tailored mock interview questions and grades user answers interactively.
4. **PDF Chatbot (Custom RAG):** Chunks PDFs, embeds them locally, saves them to MongoDB, and performs ultra-fast cosine similarity searches to answer context-aware questions.
5. **Research Assistant:** Supports multi-PDF uploads for cross-document analysis.
6. **Image Q&A (Vision):** Uses Groq's multimodal vision models to analyze uploaded images.
7. **Chat History & Persistence:** Seamless history management linked to NextAuth emails.

---

## ⚙️ Local Setup & Installation

### 1. Prerequisites
- Python 3.10+
- A [Groq API Key](https://console.groq.com/)
- A [MongoDB Atlas](https://www.mongodb.com/products/platform/atlas-database) Connection String

### 2. Clone the Repository
```bash
git clone https://github.com/HarshPariya/AI-Assistant-Backend.git
cd AI-Assistant-Backend
```

### 3. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-70b-8192
GROQ_VISION_MODEL=llama-3.2-11b-vision-preview
MONGODB_URL=your_mongodb_connection_string
FRONTEND_URL=http://localhost:3000
```

### 6. Run the Application
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## 👨‍💻 About The Developer

Developed by **Harsh Pariya**.
This backend was meticulously engineered from scratch to prioritize speed and efficiency. By avoiding heavy wrapper libraries like LangChain and building a custom NumPy-based RAG pipeline, the system demonstrates deep, low-level integration with modern AI technologies.
