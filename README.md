# AI Career & Research Assistant (Backend)

A powerful, multi-modal generative AI platform built with FastAPI and Groq. This backend powers the AI Career & Research Assistant, offering a suite of intelligent tools for career development, document analysis, and conversational AI.

**Live Backend API URL:** [https://ai-assistant-backend-01.onrender.com](https://ai-assistant-backend-01.onrender.com)  
**Frontend Application:** [https://ai-assistant-gamma-sable.vercel.app](https://ai-assistant-gamma-sable.vercel.app)

---

## 🚀 Features & Modules

1. **General AI Chat:** A blazing-fast conversational agent powered by Groq LLMs (Llama 3 / Mixtral).
2. **Resume Reviewer:** Upload PDF resumes. The system parses the text and uses an LLM to evaluate it, providing an ATS score, strengths, weaknesses, and actionable improvements.
3. **Interview Assistant:** Generates tailored mock interview questions based on job role and experience level. Evaluates user answers with constructive feedback.
4. **PDF Chatbot (RAG):** Upload any PDF document. The backend chunks the text, embeds it using Sentence Transformers, stores it in a custom MongoDB vector store, and allows users to ask questions with context-aware answers.
5. **Research Assistant:** Supports multi-PDF uploads for cross-document analysis. Users can query across multiple research papers simultaneously.
6. **Image Q&A (Vision):** Upload images and ask questions about them using Groq's multimodal vision models.
7. **Chat History & Persistence:** All conversations, session data, and vector stores are persisted in **MongoDB Atlas**, allowing seamless resumption of tasks across sessions.

---

## 🛠️ Technology Stack

- **Framework:** FastAPI (Python 3.10+)
- **LLM Engine:** [Groq](https://groq.com/) API (Llama 3, Mixtral)
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector Search:** NumPy & Scikit-learn (Cosine Similarity)
- **Database:** MongoDB Atlas (PyMongo)
- **Document Processing:** PyMuPDF (`fitz`), Pillow (Images)
- **Deployment:** Render

---

## ⚙️ Local Setup & Installation

### 1. Prerequisites
- Python 3.10 or higher
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
Create a `.env` file in the root directory and add the following:
```env
# Groq Setup
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-70b-8192
GROQ_VISION_MODEL=llama-3.2-11b-vision-preview

# MongoDB
MONGODB_URL=your_mongodb_connection_string

# Frontend Connection (CORS)
FRONTEND_URL=http://localhost:3000
```

### 6. Run the Application
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
The API will be available at `http://localhost:8000`. You can view the interactive Swagger documentation at `http://localhost:8000/docs`.

---

## 📂 Project Structure
```text
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── modules/                # Feature-specific API routers
│   ├── general_chat.py
│   ├── image_captioning.py
│   ├── interview_assistant.py
│   ├── pdf_chatbot.py
│   ├── research_assistant.py
│   ├── resume_reviewer.py
│   └── history.py          # MongoDB history integration
└── utils/                  # Shared utilities
    ├── embeddings.py       # Sentence Transformers & Vector Search
    ├── llm.py              # Groq client wrapper
    ├── pdf_parser.py       # Document extraction logic
    └── session_store.py    # MongoDB persistence functions
```

---

## 👨‍💻 About The Developer

Developed by Harsh Pariya.
This robust backend architecture was engineered to demonstrate high-performance AI integration using FastAPI, Groq's ultra-fast inference, and efficient Retrieval-Augmented Generation (RAG) pipelines.

Enjoy building the future of AI!
