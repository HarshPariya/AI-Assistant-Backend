"""
Groq LLM Client — Centralized LLM access for all modules
Optimized for speed: lower max_tokens defaults, async-safe retries
"""
import asyncio
import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client: Groq | None = None

# Active vision model (llama-4-scout replaces decommissioned llama-3.2-90b-vision-preview)
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL_DEFAULT = "llama-3.1-8b-instant"


def get_groq_client() -> Groq:
    """Get or create a Groq client instance (singleton)."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            raise ValueError(
                "GROQ_API_KEY not set. Please add your Groq API key to the .env file. "
                "Get a free key at https://console.groq.com"
            )
        _client = Groq(api_key=api_key)
    return _client


def get_model() -> str:
    """Get the text model name."""
    model = os.getenv("GROQ_MODEL", TEXT_MODEL_DEFAULT)
    # Hard override for the rate-limited 70b model in case Render still has it in env variables
    if model == "llama-3.3-70b-versatile":
        return TEXT_MODEL_DEFAULT
    return model


def get_vision_model() -> str:
    """Get the vision model name. Always returns the active non-decommissioned model."""
    # Force the working model — bypass any stale env vars
    return VISION_MODEL


def chat_completion(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,  # Reduced default for speed (was 4096)
) -> str:
    """
    Run a chat completion with Groq with robust retry logic.
    Uses linear backoff to avoid blocking the event loop excessively.
    """
    client = get_groq_client()
    selected_model = model or get_model()

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=selected_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            # Don't retry on client errors (4xx) — only on rate limits / server errors
            if "400" in str(e) or "401" in str(e) or "403" in str(e):
                print(f"Groq API Client Error (no retry): {str(e)}")
                raise

            if attempt < max_retries - 1:
                # Short linear backoff: 1s, 2s — don't wait too long
                sleep_time = attempt + 1
                print(f"Groq API Error (attempt {attempt + 1}): {str(e)}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                print(f"Groq API Error after {max_retries} attempts: {str(e)}")
                raise last_error


def system_user_chat(
    system_prompt: str,
    user_message: str,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,  # Reduced default for speed
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
    """Non-blocking wrapper — runs sync Groq call in a thread pool."""
    return await asyncio.to_thread(
        chat_completion, messages, model, temperature, max_tokens
    )


async def async_system_user_chat(
    system_prompt: str,
    user_message: str,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """Non-blocking wrapper for system + user message pattern."""
    return await asyncio.to_thread(
        system_user_chat, system_prompt, user_message, model, temperature, max_tokens
    )
