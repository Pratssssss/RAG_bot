from __future__ import annotations

import json
from urllib import request, error

from ragbot.config import Settings


SYSTEM_PROMPT = """You are a grounded document Q&A assistant.
Answer only from the provided context.
If the answer is not supported by the context, say: "I don't know based on the provided documents."
Include citations in square brackets using the source labels."""


def generate_answer(question: str, chunks: list[dict], settings: Settings) -> str:
    prompt = build_prompt(question, chunks)
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 768},
    }
    data = json.dumps(payload).encode("utf-8")
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Ollama generation failed: HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError("Could not reach Ollama. Make sure Ollama is installed and running.") from exc
    return body.get("message", {}).get("content", "").strip()


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = []
    for index, chunk in enumerate(chunks, start=1):
        context.append(f"[Source {index}: {chunk['citation']}]\n{chunk['text']}")
    return f"""Context:
{chr(10).join(context)}

Question: {question}

Answer concisely with citations. Use only the context."""
