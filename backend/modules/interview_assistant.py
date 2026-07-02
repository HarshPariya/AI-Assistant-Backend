"""
Module 2: Interview Assistant
Role-based interview question generation and mock interview mode
"""
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from utils.llm import async_system_user_chat

router = APIRouter()

ROLES = [
    "AI/ML Engineer",
    "Data Scientist",
    "Software Engineer",
    "Data Analyst",
    "Backend Engineer",
    "Frontend Engineer",
    "Full Stack Engineer",
    "DevOps Engineer",
    "Product Manager",
    "Cybersecurity Engineer",
]


class GenerateRequest(BaseModel):
    role: str
    experience_level: str = "mid"  # junior, mid, senior
    num_questions: int = 10
    focus_areas: list[str] = []


class QuestionResponse(BaseModel):
    questions: list[dict]
    role: str
    total: int


class MockInterviewRequest(BaseModel):
    role: str
    question: str
    user_answer: str
    conversation_history: list[dict] = []


class MockInterviewResponse(BaseModel):
    evaluation: str
    score: int
    follow_up_question: str
    tips: list[str]
    model_answer_hint: str


INTERVIEW_SYSTEM = """You are an expert technical interviewer with 15+ years of experience at top tech companies.
Generate interview questions and evaluate answers professionally.
Return ONLY valid JSON as specified."""

EVAL_SYSTEM = """You are a senior technical interviewer evaluating a candidate's answer.
Provide constructive, specific feedback. Be fair but rigorous.
Return ONLY valid JSON."""


@router.get("/roles")
async def get_roles():
    """Get available roles for interview generation."""
    return {"roles": ROLES}


@router.post("/generate", response_model=QuestionResponse)
async def generate_questions(req: GenerateRequest):
    """Generate interview questions for a specific role."""
    # Cap questions at 10 for speed
    num_q = min(req.num_questions, 10)
    focus = f" Focus on: {', '.join(req.focus_areas)}." if req.focus_areas else ""

    prompt = f"""Generate {num_q} interview questions for a {req.experience_level}-level {req.role} position.{focus}

Return JSON:
{{
  "questions": [
    {{
      "id": 1,
      "question": "<question text>",
      "category": "<Technical|Behavioral|System Design|Problem Solving>",
      "difficulty": "<Easy|Medium|Hard>",
      "expected_time": "<2-3 minutes>",
      "hint": "<brief hint for the interviewer>",
      "model_answer": "<a detailed exemplary answer (1-2 paragraphs) a strong candidate would give>"
    }}
  ]
}}"""

    try:
        response = await async_system_user_chat(
            system_prompt=INTERVIEW_SYSTEM,
            user_message=prompt,
            temperature=0.7,
            max_tokens=1500,
        )
        cleaned = response.strip()
        if cleaned.startswith("```"):
            parts = cleaned.split("```")
            cleaned = parts[1] if len(parts) > 1 else cleaned
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        data = json.loads(cleaned)
        return QuestionResponse(
            questions=data["questions"],
            role=req.role,
            total=len(data["questions"]),
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response as JSON. Please try again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mock-evaluate", response_model=MockInterviewResponse)
async def evaluate_mock_answer(req: MockInterviewRequest):
    """Evaluate a candidate's mock interview answer and provide a follow-up question."""
    history_text = ""
    if req.conversation_history:
        history_text = "\n\nConversation so far:\n" + "\n".join(
            [f"Q: {h.get('question', '')}\nA: {h.get('answer', '')}"
             for h in req.conversation_history[-2:]]  # Reduced to last 2 for speed
        )

    prompt = f"""Role: {req.role}
Question asked: {req.question}
Candidate's answer: {req.user_answer}{history_text}

Evaluate and return JSON:
{{
  "evaluation": "<detailed constructive feedback>",
  "score": <0-100>,
  "follow_up_question": "<relevant follow-up question>",
  "tips": ["<tip1>", "<tip2>", "<tip3>"],
  "model_answer_hint": "<brief hint on what a strong answer would include>"
}}"""

    try:
        response = await async_system_user_chat(
            system_prompt=EVAL_SYSTEM,
            user_message=prompt,
            temperature=0.4,
            max_tokens=700,
        )
        cleaned = response.strip()
        if cleaned.startswith("```"):
            parts = cleaned.split("```")
            cleaned = parts[1] if len(parts) > 1 else cleaned
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        data = json.loads(cleaned)
        return MockInterviewResponse(**data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response. Please try again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/answer")
async def generate_model_answer(role: str, question: str):
    """Generate a model answer for an interview question."""
    prompt = f"Role: {role}\nQuestion: {question}\n\nProvide a comprehensive, structured model answer (1-2 paragraphs) that a senior candidate would give. Include specific examples and technical details."

    try:
        response = await async_system_user_chat(
            system_prompt="You are an expert technical interviewer. Provide exemplary interview answers.",
            user_message=prompt,
            temperature=0.5,
            max_tokens=500,
        )
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
