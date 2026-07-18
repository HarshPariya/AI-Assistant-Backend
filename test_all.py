"""
Comprehensive test script to check all modules, API key, PDF handling, and image processing.
"""
import sys
import os
import json
import time
import asyncio
import base64
import struct
import zlib
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Suppress all warnings (includes pypdf parsing warnings)
warnings.filterwarnings("ignore")

# Redirect stderr to suppress C-level pypdf warnings ("incorrect startxref pointer")
_devnull = open(os.devnull, "w")
_old_stderr = sys.stderr
sys.stderr = _devnull

# Load env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

print("=" * 60)
print("AI Career & Research Assistant - Full System Check")
print("=" * 60)

# === 1. Check Environment Variables ===
print("\n[1] Checking Environment Variables...")
api_key = os.getenv("GROQ_API_KEY", "")
groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
groq_vision = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")
mongodb_url = os.getenv("MONGODB_URL", "")

print(f"  GROQ_API_KEY:        {'✅ Set (' + api_key[:12] + '...)' if api_key and api_key != 'your_groq_api_key_here' else '❌ NOT SET'}")
print(f"  GROQ_MODEL:          {groq_model}")
print(f"  GROQ_VISION_MODEL:   {groq_vision}")
print(f"  MONGODB_URL:         {'✅ Set' if mongodb_url else '⚠️  Not set (optional)'}")

# === 2. Test Groq API Key ===
print("\n[2] Testing Groq API Key...")
try:
    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=groq_model,
        messages=[{"role": "user", "content": "Say 'API OK' in 2 words only."}],
        max_tokens=10,
        temperature=0,
    )
    result = response.choices[0].message.content.strip()
    print(f"  Groq Text API:       ✅ Working! Response: '{result}'")
except Exception as e:
    print(f"  Groq Text API:       ❌ FAILED: {e}")

# === 3. Test Vision API ===
print("\n[3] Testing Groq Vision API (qwen3.6-27b)...")
try:
    def _make_test_png(w=20, h=20, rgb=(255, 50, 50)):
        def chunk(name, data):
            c = name + data
            return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
        sig = b'\x89PNG\r\n\x1a\n'
        ihdr = chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
        row = b'\x00' + bytes(rgb) * w
        raw_data = row * h
        idat = chunk(b'IDAT', zlib.compress(raw_data))
        iend = chunk(b'IEND', b'')
        return sig + ihdr + idat + iend

    img_bytes = _make_test_png(50, 50, (255, 50, 50))
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    response = client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this image? Answer in 3 words."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]
        }],
        max_tokens=20,
        temperature=0,
    )
    result = response.choices[0].message.content.strip()
    print(f"  Groq Vision API:     ✅ Working! Response: '{result}'")
except Exception as e:
    print(f"  Groq Vision API:     ❌ FAILED: {e}")

# === 4. Test PDF Loading ===
print("\n[4] Testing PDF Loading...")
try:
    from backend.utils.pdf_loader import extract_text_from_pdf, extract_text_with_pages, chunk_pages
    test_pdf = str(Path(__file__).parent / "test_dummy.pdf")
    if os.path.exists(test_pdf):
        text = extract_text_from_pdf(test_pdf)
        pages = extract_text_with_pages(test_pdf)
        chunks = chunk_pages(pages)
        print(f"  PDF Text Extract:    ✅ OK ({len(text)} chars, {len(pages)} pages, {len(chunks)} chunks)")
    else:
        print(f"  PDF Test:            ⚠️  No test_dummy.pdf found, skipping")
except Exception as e:
    print(f"  PDF Loading:         ❌ FAILED: {e}")

# === 5. Test Embeddings ===
print("\n[5] Testing Embedding Model...")
try:
    from backend.utils.embeddings import get_embedding_model, embed_texts, CHROMA_AVAILABLE
    model = get_embedding_model()
    embeddings = embed_texts(["Hello world", "Test sentence for embeddings"])
    mode_str = "ChromaDB" if CHROMA_AVAILABLE else "NumPy Fallback"
    num_embeddings = len(embeddings)
    dim_embeddings = len(embeddings[0]) if isinstance(embeddings, list) else embeddings.shape[1]
    print(f"  Embedding Model:     ✅ OK (shape: ({num_embeddings}, {dim_embeddings}), mode: {mode_str})")
except Exception as e:
    print(f"  Embedding Model:     ❌ FAILED: {e}")

# === 6. Test MongoDB Connection ===
print("\n[6] Testing MongoDB Connection...")
try:
    from backend.utils.session_store import is_available
    avail = is_available()
    print(f"  MongoDB:             {'✅ Connected' if avail else '⚠️  Not available (sessions will use local storage)'}")
except Exception as e:
    print(f"  MongoDB:             ❌ FAILED: {e}")

# === 7. Test JSON Helpers ===
print("\n[7] Testing JSON Helpers (Interview/Resume modules)...")
try:
    from backend.utils.json_helpers import async_json_system_user_chat
    print(f"  JSON Helpers:        ✅ Imported OK")
except Exception as e:
    print(f"  JSON Helpers:        ❌ FAILED: {e}")

# === 8. Test Module Imports ===
print("\n[8] Testing All Module Imports...")
modules = [
    ("Resume Reviewer", "modules.resume_reviewer"),
    ("Interview Assistant", "modules.interview_assistant"),
    ("PDF Chatbot", "modules.pdf_chatbot"),
    ("Research Assistant", "modules.research_assistant"),
    ("Image Captioning", "modules.image_captioning"),
    ("General Chat", "modules.general_chat"),
    ("History", "modules.history"),
    ("Voice", "modules.voice"),
]
for name, mod_path in modules:
    try:
        __import__(mod_path)
        print(f"  {name:<25} ✅ Import OK")
    except Exception as e:
        print(f"  {name:<25} ❌ FAILED: {e}")

# === 9. Full E2E: General Chat ===
print("\n[9] Testing General Chat (E2E)...")
try:
    from backend.utils.llm import async_chat_completion, get_model

    async def test_general():
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2? Answer with just the number."},
        ]
        answer = await async_chat_completion(msgs, model=get_model(), max_tokens=10)
        return answer

    result = asyncio.run(test_general())
    print(f"  General Chat E2E:    ✅ Working! Answer: '{result.strip()}'")
except Exception as e:
    print(f"  General Chat E2E:    ❌ FAILED: {e}")

# === 10. Full E2E: PDF Chatbot (Upload + Ask) ===
print("\n[10] Testing PDF Chatbot (Upload + Ask E2E)...")
try:
    test_pdf = str(Path(__file__).parent / "test_dummy.pdf")
    if os.path.exists(test_pdf):
        from backend.utils.pdf_loader import extract_text_with_pages, chunk_pages
        from backend.utils.embeddings import build_and_save_vector_store, similarity_search, CHROMA_AVAILABLE

        import uuid

        pages = extract_text_with_pages(test_pdf)
        chunks = chunk_pages(pages)
        session_id = str(uuid.uuid4())
        count = build_and_save_vector_store(session_id, chunks)
        results = similarity_search(session_id, "What is this document about?", top_k=3)
        mode_str = "ChromaDB" if CHROMA_AVAILABLE else "NumPy Fallback"
        print(f"  PDF RAG Pipeline:    ✅ OK ({count} chunks indexed, {len(results)} results found, mode: {mode_str})")

        # Cleanup
        import glob
        import shutil
        for f in glob.glob(f"vector_store/{session_id}*"):
            if os.path.isdir(f):
                shutil.rmtree(f, ignore_errors=True)
            else:
                os.remove(f)
    else:
        print(f"  PDF RAG Pipeline:    ⚠️  No test_dummy.pdf found, skipping")
except Exception as e:
    print(f"  PDF RAG Pipeline:    ❌ FAILED: {e}")

# === 11. UPLOAD_DIR checks ===
print("\n[11] Checking Directory Setup...")
for d in ["uploaded_files", "vector_store"]:
    full_path = Path(__file__).parent / d
    exists = full_path.exists()
    writable = os.access(str(full_path), os.W_OK) if exists else False
    print(f"  {d:<20} {'✅ Exists & Writable' if exists and writable else '⚠️ Missing or not writable - will be created on demand'}")

print("\n" + "=" * 60)
print("System Check Complete!")
print("=" * 60)
