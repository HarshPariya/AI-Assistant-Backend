"""
Module 2: Interview Assistant
Role-based interview question generation and mock interview mode
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from utils.llm import async_system_user_chat, get_chat_model
from langchain_core.prompts import ChatPromptTemplate

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


class InterviewQuestion(BaseModel):
    id: int
    question: str
    category: str
    difficulty: str
    expected_time: str
    hint: str
    model_answer: str


class QuestionListResponse(BaseModel):
    questions: list[InterviewQuestion]


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
Generate interview questions and evaluate answers professionally."""

EVAL_SYSTEM = """You are a senior technical interviewer evaluating a candidate's answer.
Provide constructive, specific feedback. Be fair but rigorous."""


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

    user_msg = f"Generate {num_q} interview questions for a {req.experience_level}-level {req.role} position.{focus}"

    try:
        chat_model = get_chat_model(temperature=0.7, max_tokens=3000)
        structured_llm = chat_model.with_structured_output(QuestionListResponse)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", INTERVIEW_SYSTEM),
            ("user", "{user_msg}")
        ])
        
        chain = prompt | structured_llm
        result = await chain.ainvoke({"user_msg": user_msg})
        
        # Convert Pydantic models to dicts for backward compatibility with frontend
        questions_dicts = [q.model_dump() for q in result.questions]
        
        if not questions_dicts:
            raise ValueError("No questions returned by the model")
            
        return QuestionResponse(
            questions=questions_dicts,
            role=req.role,
            total=len(questions_dicts),
        )
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

    user_msg = f"""Role: {req.role}
Question asked: {req.question}
Candidate's answer: {req.user_answer}{history_text}"""

    try:
        chat_model = get_chat_model(temperature=0.4, max_tokens=900)
        structured_llm = chat_model.with_structured_output(MockInterviewResponse)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", EVAL_SYSTEM),
            ("user", "{user_msg}")
        ])
        
        chain = prompt | structured_llm
        result = await chain.ainvoke({"user_msg": user_msg})
        return result
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
