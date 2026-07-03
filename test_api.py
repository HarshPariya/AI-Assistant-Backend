"""
API endpoint live test — tests all module endpoints against the running server.
"""
import requests
import time
import sys
import json
import io
import struct
import zlib
import base64

BASE = "http://localhost:8000"

def wait_for_server(max_wait=30):
    for i in range(max_wait):
        try:
            r = requests.get(f"{BASE}/", timeout=2)
            if r.status_code == 200:
                print("✅ Backend server is UP!")
                return True
        except Exception:
            pass
        time.sleep(1)
    print("❌ Backend server not responding after 30s")
    return False


def make_png(w=10, h=10, rgb=(255, 50, 50)):
    """Create a minimal valid PNG image in-memory."""
    def chunk(name, data):
        c = name + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    sig = b'\x89PNG\r\n\x1a\n'
    ihdr = chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
    row = b'\x00' + bytes(rgb) * w
    raw = row * h
    idat = chunk(b'IDAT', zlib.compress(raw))
    iend = chunk(b'IEND', b'')
    return sig + ihdr + idat + iend


if not wait_for_server():
    sys.exit(1)

print()
print("=" * 60)
print("Live API Endpoint Tests")
print("=" * 60)
print()

results = []

# ── 1. Health ─────────────────────────────────────────────────────────────
try:
    r = requests.get(f"{BASE}/health", timeout=10)
    data = r.json()
    status = f"status={data.get('status')} mongodb={data.get('mongodb')}"
    ok = r.status_code == 200
    print(f"[1] GET /health: {'✅' if ok else '❌'} {r.status_code} - {status}")
    results.append(("Health", ok))
except Exception as e:
    print(f"[1] GET /health: ❌ ERROR: {e}")
    results.append(("Health", False))

# ── 2. Interview Roles ────────────────────────────────────────────────────
try:
    r = requests.get(f"{BASE}/interview/roles", timeout=10)
    roles = r.json().get("roles", [])
    ok = r.status_code == 200 and len(roles) > 0
    print(f"[2] GET /interview/roles: {'✅' if ok else '❌'} {r.status_code} - {len(roles)} roles")
    results.append(("Interview Roles", ok))
except Exception as e:
    print(f"[2] GET /interview/roles: ❌ ERROR: {e}")
    results.append(("Interview Roles", False))

# ── 3. Generate Interview Questions ───────────────────────────────────────
try:
    r = requests.post(f"{BASE}/interview/generate", json={
        "role": "Software Engineer",
        "experience_level": "mid",
        "num_questions": 3,
        "focus_areas": []
    }, timeout=60)
    data = r.json()
    total = data.get("total", 0)
    ok = r.status_code == 200 and total > 0
    print(f"[3] POST /interview/generate: {'✅' if ok else '❌'} {r.status_code} - {total} questions generated")
    results.append(("Interview Generate", ok))
except Exception as e:
    print(f"[3] POST /interview/generate: ❌ ERROR: {e}")
    results.append(("Interview Generate", False))

# ── 4. General Chat (text only) ───────────────────────────────────────────
try:
    r = requests.post(f"{BASE}/general/ask",
        data={"messages": json.dumps([{"role": "user", "content": "Reply with exactly: GENERAL_OK"}])},
        timeout=30
    )
    answer = r.json().get("answer", "")
    ok = r.status_code == 200 and bool(answer)
    print(f"[4] POST /general/ask (text): {'✅' if ok else '❌'} {r.status_code} - '{answer[:40]}'")
    results.append(("General Chat Text", ok))
except Exception as e:
    print(f"[4] POST /general/ask: ❌ ERROR: {e}")
    results.append(("General Chat Text", False))

# ── 5. General Chat with Image ────────────────────────────────────────────
try:
    png_bytes = make_png(50, 50, (255, 0, 0))
    r = requests.post(f"{BASE}/general/ask",
        data={"messages": json.dumps([{"role": "user", "content": "What color is this image? Answer: color name only."}])},
        files={"file": ("test.png", io.BytesIO(png_bytes), "image/png")},
        timeout=40
    )
    answer = r.json().get("answer", "")
    ok = r.status_code == 200 and bool(answer)
    print(f"[5] POST /general/ask (image): {'✅' if ok else '❌'} {r.status_code} - '{answer[:50]}'")
    results.append(("General Chat Image", ok))
except Exception as e:
    print(f"[5] POST /general/ask (image): ❌ ERROR: {e}")
    results.append(("General Chat Image", False))

# ── 6. PDF Chatbot Upload ─────────────────────────────────────────────────
session_id = None
try:
    pdf_path = "test_dummy.pdf"
    with open(pdf_path, "rb") as f:
        r = requests.post(f"{BASE}/chat/upload",
            files={"file": ("test.pdf", f, "application/pdf")},
            timeout=60
        )
    data = r.json()
    session_id = data.get("session_id")
    chunk_count = data.get("chunk_count", 0)
    ok = r.status_code == 200 and bool(session_id)
    print(f"[6] POST /chat/upload: {'✅' if ok else '❌'} {r.status_code} - session={session_id[:8] if session_id else 'none'}... chunks={chunk_count}")
    results.append(("PDF Upload", ok))
except Exception as e:
    print(f"[6] POST /chat/upload: ❌ ERROR: {e}")
    results.append(("PDF Upload", False))

# ── 7. PDF Chatbot Ask ────────────────────────────────────────────────────
if session_id:
    try:
        r = requests.post(f"{BASE}/chat/ask",
            json={"session_id": session_id, "message": "What is this document about?"},
            timeout=40
        )
        answer = r.json().get("answer", "")
        ok = r.status_code == 200 and bool(answer)
        print(f"[7] POST /chat/ask: {'✅' if ok else '❌'} {r.status_code} - '{answer[:60]}'")
        results.append(("PDF Chat Ask", ok))
    except Exception as e:
        print(f"[7] POST /chat/ask: ❌ ERROR: {e}")
        results.append(("PDF Chat Ask", False))
else:
    print("[7] POST /chat/ask: ⏭️  Skipped (no session_id)")
    results.append(("PDF Chat Ask", False))

# ── 8. Vision Analyze ────────────────────────────────────────────────────
vision_session_id = None
try:
    png_bytes = make_png(50, 50, (50, 200, 50))
    r = requests.post(f"{BASE}/vision/analyze",
        files={"file": ("green_img.png", io.BytesIO(png_bytes), "image/png")},
        timeout=40
    )
    data = r.json()
    vision_session_id = data.get("session_id")
    caption = data.get("caption", "")
    ok = r.status_code == 200 and bool(vision_session_id) and bool(caption)
    print(f"[8] POST /vision/analyze: {'✅' if ok else '❌'} {r.status_code} - caption='{caption[:50]}'")
    results.append(("Vision Analyze", ok))
except Exception as e:
    print(f"[8] POST /vision/analyze: ❌ ERROR: {e}")
    results.append(("Vision Analyze", False))

# ── 9. Vision Ask ─────────────────────────────────────────────────────────
if vision_session_id:
    try:
        r = requests.post(f"{BASE}/vision/ask",
            json={"session_id": vision_session_id, "question": "What do you see?"},
            timeout=40
        )
        answer = r.json().get("answer", "")
        ok = r.status_code == 200 and bool(answer)
        print(f"[9] POST /vision/ask: {'✅' if ok else '❌'} {r.status_code} - '{answer[:60]}'")
        results.append(("Vision Ask", ok))
    except Exception as e:
        print(f"[9] POST /vision/ask: ❌ ERROR: {e}")
        results.append(("Vision Ask", False))
else:
    print("[9] POST /vision/ask: ⏭️  Skipped (no vision session)")
    results.append(("Vision Ask", False))

# ── 10. Resume Analyze ────────────────────────────────────────────────────
try:
    pdf_path = "test_dummy.pdf"
    with open(pdf_path, "rb") as f:
        r = requests.post(f"{BASE}/resume/analyze",
            files={"file": ("resume.pdf", f, "application/pdf")},
            timeout=60
        )
    ok = r.status_code == 200
    if ok:
        data = r.json()
        score = data.get("ats_score", "?")
        print(f"[10] POST /resume/analyze: ✅ {r.status_code} - ATS score={score}/100")
    else:
        detail = r.json().get("detail", r.text[:80])
        print(f"[10] POST /resume/analyze: ❌ {r.status_code} - {detail}")
    results.append(("Resume Analyze", ok))
except Exception as e:
    print(f"[10] POST /resume/analyze: ❌ ERROR: {e}")
    results.append(("Resume Analyze", False))

# ── 11. Research Upload ───────────────────────────────────────────────────
research_session = None
try:
    pdf_path = "test_dummy.pdf"
    with open(pdf_path, "rb") as f:
        r = requests.post(f"{BASE}/research/upload",
            files=[("files", ("test.pdf", f, "application/pdf"))],
            timeout=60
        )
    data = r.json()
    research_session = data.get("session_id")
    docs = data.get("documents", [])
    ok = r.status_code == 200 and bool(research_session)
    print(f"[11] POST /research/upload: {'✅' if ok else '❌'} {r.status_code} - {len(docs)} docs, session={research_session[:8] if research_session else 'none'}...")
    results.append(("Research Upload", ok))
except Exception as e:
    print(f"[11] POST /research/upload: ❌ ERROR: {e}")
    results.append(("Research Upload", False))

# ── 12. Research Action ──────────────────────────────────────────────────
if research_session:
    try:
        r = requests.post(f"{BASE}/research/action",
            json={"session_id": research_session, "action": "key_takeaways"},
            timeout=60
        )
        result = r.json().get("result", "")
        ok = r.status_code == 200 and bool(result)
        print(f"[12] POST /research/action: {'✅' if ok else '❌'} {r.status_code} - result='{result[:60]}'")
        results.append(("Research Action", ok))
    except Exception as e:
        print(f"[12] POST /research/action: ❌ ERROR: {e}")
        results.append(("Research Action", False))
else:
    print("[12] POST /research/action: ⏭️  Skipped")
    results.append(("Research Action", False))

# ── 13. General Chat PDF Upload ───────────────────────────────────────────
try:
    pdf_path = "test_dummy.pdf"
    with open(pdf_path, "rb") as f:
        r = requests.post(f"{BASE}/general/ask",
            data={"messages": json.dumps([{"role": "user", "content": "Summarize this PDF in one sentence."}])},
            files={"file": ("doc.pdf", f, "application/pdf")},
            timeout=60
        )
    answer = r.json().get("answer", "")
    ok = r.status_code == 200 and bool(answer)
    print(f"[13] POST /general/ask (PDF): {'✅' if ok else '❌'} {r.status_code} - '{answer[:60]}'")
    results.append(("General Chat PDF", ok))
except Exception as e:
    print(f"[13] POST /general/ask (PDF): ❌ ERROR: {e}")
    results.append(("General Chat PDF", False))

print()
print("=" * 60)
print("Results Summary:")
passed = sum(1 for _, ok in results if ok)
total = len(results)
for name, ok in results:
    print(f"  {'✅' if ok else '❌'} {name}")
print()
print(f"Passed: {passed}/{total}")
print("=" * 60)
