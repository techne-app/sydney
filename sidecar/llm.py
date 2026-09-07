"""Gemma, served by llama-server.

Gemma is not loaded in-process. It has to be served: only llama-server's
--jinja applies the model's own chat template and returns tool calls as a
structured `tool_calls` field. llama-cpp-python hands back the raw
`<|tool_call>call:...` text, which is why this project once carried a
hand-written parser for Gemma's syntax.
"""
import os
from typing import Any, Dict, List

import httpx

LLAMA_SERVER_URL = os.environ.get("TECHNE_LLAMA_SERVER", "http://127.0.0.1:8081")
LLAMA_TIMEOUT = 300.0


def chat_completion(messages: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
    """Call Gemma (OpenAI-compatible /v1/chat/completions). Returns the message.

    Thinking is disabled. Gemma bills chain-of-thought against the same
    max_tokens as the answer, and will spend the whole budget reasoning and emit
    nothing — `finish_reason: "length"`, empty content. That produced "Sorry, I
    didn't catch that" for follow-up questions and left the reranker, at
    max_tokens=64, silently falling back to cosine order.

    Neither routing nor reranking needs it, and it costs ~500 tokens at 28 tok/s
    — roughly 18 seconds of latency for nothing.

    Callers can pass chat_template_kwargs to override.
    """
    payload = {
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": messages,
        **kwargs,
    }
    with httpx.Client(timeout=LLAMA_TIMEOUT) as client:
        response = client.post(f"{LLAMA_SERVER_URL}/v1/chat/completions", json=payload)
        response.raise_for_status()
    return response.json()["choices"][0]["message"]


def llama_server_ready() -> bool:
    try:
        with httpx.Client(timeout=5.0) as client:
            return client.get(f"{LLAMA_SERVER_URL}/health").status_code == 200
    except Exception:
        return False
