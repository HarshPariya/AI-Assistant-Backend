"""
Module 1: Resume Reviewer
ATS scoring, skill gap analysis, job role matching
"""
import json
import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from utils.pdf_loader import extract_text_from_pdf
from utils.json_helpers import async_json_system_user_chat

router = APIRouter()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploaded_files")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


class ResumeAnalysis(BaseModel):
    ats_score: int
    ats_breakdown: dict
    strengths: list[str]
    weaknesses: list[str]
    missing_skills: list[str]
    improvements: list[str]
    job_roles: list[dict]
    overall_feedback: str


RESUME_SYSTEM_PROMPT = """You are an expert ATS (Applicant Tracking System) and career coach AI.
Analyze the provided resume text and return a comprehensive, structured JSON analysis.
Be specific, actionable, and encouraging but honest.

Return ONLY valid JSON with this exact structure:
{
  "ats_score": <integer 0-100>,
  "ats_breakdown": {
    "formatting": <0-20>,
    "keywords": <0-25>,
    "experience": <0-25>,
    "skills": <0-20>,
    "education": <0-10>
  },
  "strengths": [<list of 3-5 specific strengths>],
  "weaknesses": [<list of 3-5 specific weaknesses>],
  "missing_skills": [<list of 5-8 skills that should be added>],
  "improvements": [<list of 5-7 actionable improvement suggestions>],
  "job_roles": [
    {"role": "<job title>", "match_score": <0-100>, "reason": "<brief reason>"}
  ],
  "overall_feedback": "<1-2 paragraph comprehensive feedback>"
}"""


@router.post("/analyze", response_model=ResumeAnalysis)
async def analyze_resume(file: UploadFile = File(...)):
    """Upload a PDF resume and get comprehensive ATS analysis."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")

    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"resume_{file_id}.pdf")
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(content)

    try:
        resume_text = extract_text_from_pdf(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract PDF text: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    if len(resume_text.strip()) < 100:
        raise HTTPException(status_code=400, detail="PDF appears to be empty or unreadable.")

    try:
        analysis = await async_json_system_user_chat(
            system_prompt=RESUME_SYSTEM_PROMPT,
            user_message=f"Analyze this resume:\n\n{resume_text[:5000]}",
            temperature=0.3,
            max_tokens=2500,
            max_retries=3,
        )
        return ResumeAnalysis(**analysis)

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Failed to parse AI response. Please try again.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
