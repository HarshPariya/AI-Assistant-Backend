"""Shared JSON extraction helpers for LLM responses."""
import json
import re
from utils.llm import async_system_user_chat


def extract_json_object(text: str) -> dict:
    """Parse JSON from an LLM response, tolerating markdown fences and extra prose."""
    if not text or not text.strip():
        raise json.JSONDecodeError("Empty response", text or "", 0)

    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 1 else cleaned
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            raise
        return json.loads(match.group())


async def async_json_system_user_chat(
    system_prompt: str,
    user_message: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 2500,
    max_retries: int = 3,
) -> dict:
    """Call Groq and parse a JSON object, retrying on empty or invalid responses."""
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = await async_system_user_chat(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return extract_json_object(response)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            if attempt < max_retries - 1:
                continue
            raise

    raise last_error or ValueError("Failed to get JSON response from model")
