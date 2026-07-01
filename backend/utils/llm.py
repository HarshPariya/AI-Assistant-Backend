"""
Groq LLM Client — Centralized LLM access for all modules
"""
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client: Groq | None = None


def get_groq_client() -> Groq:
    """Get or create a Groq client instance."""
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
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def get_vision_model() -> str:
    # Hardcode to the working model to bypass cached environment variables in the terminal
    return "meta-llama/llama-4-scout-17b-16e-instruct"


import time

def chat_completion(messages: list[dict], model: str | None = None, temperature: float = 0.7, max_tokens: int = 4096) -> str:
    """Run a chat completion with Groq with robust retry logic."""
    client = get_groq_client()
    selected_model = model or get_model()
    
    max_retries = 3
    base_delay = 2
    
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
            if attempt == max_retries - 1:
                print(f"Groq API Error after {max_retries} attempts: {str(e)}")
                raise
            
            # Exponential backoff
            sleep_time = base_delay * (2 ** attempt)
            print(f"Groq API Error: {str(e)}. Retrying in {sleep_time}s...")
            time.sleep(sleep_time)


def system_user_chat(system_prompt: str, user_message: str, model: str | None = None, temperature: float = 0.7, max_tokens: int = 4096) -> str:
    """Convenience wrapper for system + user message pattern."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    return chat_completion(messages, model=model, temperature=temperature, max_tokens=max_tokens)
