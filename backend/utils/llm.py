"""
Groq LLM Client — Refactored to use LangChain
Optimized for speed: lower max_tokens defaults, async-safe retries
"""
import asyncio
import os
import time
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from dotenv import load_dotenv

load_dotenv()

# Active vision model
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL_DEFAULT = "llama-3.1-8b-instant"


def get_model() -> str:
    """Get the text model name."""
    model = os.getenv("GROQ_MODEL", TEXT_MODEL_DEFAULT)
    if model == "llama-3.3-70b-versatile":
        return TEXT_MODEL_DEFAULT
    return model


def get_vision_model() -> str:
    """Get the vision model name."""
    return VISION_MODEL


def get_chat_model(model: str | None = None, temperature: float = 0.7, max_tokens: int = 1024) -> ChatGroq:
    """Get a LangChain ChatGroq instance."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY not set. Please add your Groq API key to the .env file."
        )
        
    return ChatGroq(
        model_name=model or get_model(),
        temperature=temperature,
        max_tokens=max_tokens,
        groq_api_key=api_key,
        max_retries=3
    )


def _convert_to_langchain_messages(messages: list[dict]) -> list[BaseMessage]:
    lc_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))
    return lc_messages


def chat_completion(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """
    Run a chat completion using LangChain ChatGroq.
    Maintains backward compatibility with the old dict-based message interface.
    """
    chat = get_chat_model(model=model, temperature=temperature, max_tokens=max_tokens)
    
    try:
        # Pass directly if Vision format (complex content list), otherwise convert
        # LangChain ChatGroq can handle vision format if passed as HumanMessage
        if any(isinstance(m.get("content"), list) for m in messages):
            # Vision payload
            lc_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    lc_messages.append(SystemMessage(content=msg["content"]))
                else:
                    lc_messages.append(HumanMessage(content=msg["content"]))
            response = chat.invoke(lc_messages)
        else:
            lc_messages = _convert_to_langchain_messages(messages)
            response = chat.invoke(lc_messages)
            
        return response.content
    except Exception as e:
        print(f"LangChain Groq Error: {str(e)}")
        raise e


def system_user_chat(
    system_prompt: str,
    user_message: str,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """Convenience wrapper for system + user message pattern."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    return chat_completion(messages, model=model, temperature=temperature, max_tokens=max_tokens)


async def async_chat_completion(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """Non-blocking wrapper for async endpoints."""
    # LangChain has natively async methods (ainvoke)
    chat = get_chat_model(model=model, temperature=temperature, max_tokens=max_tokens)
    
    try:
        if any(isinstance(m.get("content"), list) for m in messages):
            lc_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    lc_messages.append(SystemMessage(content=msg["content"]))
                else:
                    lc_messages.append(HumanMessage(content=msg["content"]))
            response = await chat.ainvoke(lc_messages)
        else:
            lc_messages = _convert_to_langchain_messages(messages)
            response = await chat.ainvoke(lc_messages)
            
        return response.content
    except Exception as e:
        raise e


async def async_system_user_chat(
    system_prompt: str,
    user_message: str,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """Non-blocking wrapper for system + user message pattern."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    return await async_chat_completion(messages, model=model, temperature=temperature, max_tokens=max_tokens)
