# AI Career & Research Assistant (Backend)

A high-performance, asynchronous FastAPI backend powering the AI Career & Research Assistant platform. This robust architecture handles complex Generative AI interactions, streaming, RAG (Retrieval-Augmented Generation), and multi-modal file processing (PDFs, Images, and Audio).

**Live Backend API URL:** [https://ai-assistant-backend-01.onrender.com](https://ai-assistant-backend-01.onrender.com)  
**Live Frontend Application:** [https://ai-assistant-gamma-sable.vercel.app](https://ai-assistant-gamma-sable.vercel.app)

---

## 🔗 Repository Links
- **Backend Repository:** [https://github.com/HarshPariya/AI-Assistant-Backend](https://github.com/HarshPariya/AI-Assistant-Backend)
- **Frontend Repository:** [https://github.com/HarshPariya/AI-Assistant-Frontend](https://github.com/HarshPariya/AI-Assistant-Frontend)

---

## 🏗️ How It Was Built & Technology Breakdown

This backend was built to be lightning-fast and highly scalable, utilizing modern asynchronous Python practices.

- **Core Framework (FastAPI):** Chosen for its exceptional speed, asynchronous capabilities, and automatic Swagger documentation generation.
- **AI Integration (Groq & OpenAI SDK):** We utilize the blazing-fast Groq Inference API via the standard `groq` SDK for all core language model interactions (using models like Llama 3). 
- **NO LangChain:** To guarantee maximum performance, direct streaming control, and avoid heavy framework overhead, **we intentionally do NOT use LangChain**. All memory, tools, and agent logic are handled via custom, optimized code.
- **Vector Embeddings (pypdf & Sentence-Transformers):** We use `pypdf` for pure-Python PDF extraction to ensure serverless compatibility. Embeddings are generated locally using the `sentence-transformers` library (optimized to run on single CPU threads in cloud environments).
- **Voice AI (Whisper):** Integrates the `openai` SDK to utilize the Whisper-1 model for highly accurate speech-to-text transcriptions.
- **Web Search API (DuckDuckGo/Tavily):** Uses real-time web search integrations for our Agentic AI modules.
- **Database (MongoDB):** We use PyMongo to securely persist all conversation histories, vector stores, and sessions, ensuring seamless continuation of user activities across devices.

---

## 🚀 Key API Modules

1. `/general` - **General AI Chat:** Supports text, image inputs, and agentic tool-calling (web search). Streams responses chunk-by-chunk using Server-Sent Events (SSE).
2. `/voice` - **Voice AI (Speech-to-Text):** Handles multi-part audio file uploads and returns perfect text transcripts via Whisper.
3. `/interview` - **Interview Assistant:** Generates dynamic interview questions based on job roles and uses AI to rigorously evaluate user answers, acting as a mock interviewer.
4. `/resume` - **Resume Reviewer:** Parses user-uploaded resumes and provides a strict ATS compatibility breakdown and suggestions.
5. `/chat` - **PDF Chatbot (Custom RAG):** Processes uploaded PDFs, creates chunked vector embeddings, and enables users to chat with their documents.
6. `/research` - **Research Assistant:** Handles multiple PDF uploads concurrently, allowing for deep, cross-document comparison, summarization, and key takeaway extraction.
7. `/vision` - **Image Q&A:** A Vision AI endpoint that interprets images alongside user prompts.
8. `/history` - **Chat History:** Securely fetches and manages past conversation sessions stored in MongoDB.

---

## ⚙️ Local Setup & Installation

### 1. Prerequisites
- Python 3.10 or higher
- A [Groq](https://console.groq.com/) API Key
- An [OpenAI](https://platform.openai.com/) API Key (For Whisper Speech-to-Text)
- A [MongoDB Atlas](https://www.mongodb.com/products/platform/atlas-database) URI

### 2. Clone the Repository
```bash
git clone https://github.com/HarshPariya/AI-Assistant-Backend.git
cd AI-Assistant-Backend
```

### 3. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 4. Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=your_openai_key_here
MONGODB_URL=mongodb+srv://user:pass@cluster...
FRONTEND_URL=http://localhost:3000
```

### 5. Run the Server
```bash
npm run dev
# OR manually:
python -m uvicorn main:app --reload --port 8000
```

---

## 👨‍💻 About The Developer

Developed by **Harsh Pariya**.
This backend was built to demonstrate the power of pure, optimized Python microservices when combined with ultra-fast LLM inference providers like Groq.
