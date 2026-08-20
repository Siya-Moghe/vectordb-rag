from __future__ import annotations
import os

_CLIENT = None
MODEL_NAME = "gemini-3.6-flash"  

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context. If the context doesn't contain the answer, say so "
    "clearly instead of guessing."
)


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set. "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above."
    )


def generate(question: str, context_chunks: list[str], system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    client = _get_client()
    prompt = build_prompt(question, context_chunks)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={"system_instruction": system_prompt},
    )
    return response.text