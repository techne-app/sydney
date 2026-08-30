from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from llama_cpp import Llama
import os
import re
import json

from tools import TOOLS, TOOL_NAMES  # single source of truth for tool schemas

app = FastAPI()

# Allow requests from Tauri/localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
import sys
BASE_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
MODEL_NAME = "Qwen3-4B-Q4_K_M.gguf"
# Production: bundled inside .app at Contents/Resources/models/
RESOURCES_MODEL = os.path.join(BASE_DIR, "..", "Resources", "models", MODEL_NAME)
# Dev: uv run main.py from sidecar/, model is at sidecar/models/
LOCAL_MODEL = os.path.join(BASE_DIR, "models", MODEL_NAME)
MODEL_PATH = RESOURCES_MODEL if os.path.exists(RESOURCES_MODEL) else LOCAL_MODEL

print(f"Loading model from {MODEL_PATH}...")
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,   # -1 = offload all layers to Metal GPU
    n_ctx=4096,        # context window
    verbose=False,
)
print("Model loaded.")


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    reply: str


@app.post("/chat")
def chat(request: ChatRequest):
    # Convert messages to the format llama-cpp expects
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Qwen3: disable thinking mode via /no_think in the system prompt
    # This prevents <think>...</think> blocks and keeps responses clean/fast
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = "/no_think\n" + messages[0]["content"]
    else:
        messages = [{"role": "system", "content": "/no_think"}] + messages

    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )

    reply = response["choices"][0]["message"]["content"]
    reply = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL).strip()

    return ChatResponse(reply=reply)


# ---------------------------------------------------------------------------
# /route — LLM-native intent routing (replaces the manual IntentDetector).
# Feeds the conversation + tool schemas to Qwen and returns EITHER a tool call
# (action) OR a plain reply (chat). See docs/specs/intent-shift.md.
# ---------------------------------------------------------------------------

class RouteRequest(BaseModel):
    messages: List[Message]
    # Pinned-thread context, when the user has a thread open. Keys mirror
    # ThreadCardData / the extension's _buildContextString().
    pinned_thread: Optional[Dict[str, Any]] = None


def _build_route_system_prompt(pinned: Optional[Dict[str, Any]]) -> str:
    """Routing instruction + pinned-thread context.

    The pinned block and demonstrative guidance are ported from the extension's
    _buildContextString() (src/utils/intentDetector.ts) and actionOnly.ts.
    """
    instruction = (
        "/no_think\n"
        "You are the assistant for a Hacker News explorer app. "
        "Use a tool only when the user wants an action (search for discussions, "
        "or summarize the thread they have pinned). Otherwise, reply "
        "conversationally in plain text."
    )
    if pinned:
        context = (
            "\n\nContext: User has pinned this thread:\n"
            f"- Title: \"{pinned.get('story_title', '')}\"\n"
            f"- Theme: \"{pinned.get('theme', '')}\"\n"
            f"- Category: \"{pinned.get('category', '')}\"\n"
            f"- Comments: {pinned.get('comment_count', '')}\n"
            f"- Summary: \"{pinned.get('summary', '')}\"\n"
            "When the user says \"this thread\", \"this discussion\", \"this\", "
            "\"here\", or asks what it's about / for the key points, they mean "
            "this pinned thread."
        )
    else:
        context = "\n\nContext: No thread currently pinned."
    return instruction + context


def _extract_tool_calls(content: str) -> List[Dict[str, Any]]:
    """Parse Qwen's <tool_call>{json}</tool_call> markers out of the reply text.

    llama-cpp-python has no parser for Qwen's tool-call format, so we do it
    ourselves. Robust to multiple blocks, surrounding whitespace, and malformed
    JSON (bad blocks are skipped, not fatal).
    """
    calls: List[Dict[str, Any]] = []
    for block in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL):
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and parsed.get("name"):
            calls.append(parsed)
    return calls


def _as_tool_call(name: str, arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    return {"type": "tool_call", "name": name, "arguments": arguments or {}}


@app.post("/route")
def route(request: RouteRequest):
    messages = [{"role": "system", "content": _build_route_system_prompt(request.pinned_thread)}]
    messages += [{"role": m.role, "content": m.content} for m in request.messages]

    # Native Qwen template (no chat_format override) — it supports `tools`.
    response = llm.create_chat_completion(
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.1,   # low = consistent routing / structured output
        max_tokens=512,
    )
    msg = response["choices"][0]["message"]

    # 1. Belt-and-suspenders: if a library version parsed tool_calls, trust it.
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function", {})
        if fn.get("name") in TOOL_NAMES:
            return _as_tool_call(fn["name"], fn.get("arguments"))

    content = re.sub(r"<think>.*?</think>", "", msg.get("content") or "", flags=re.DOTALL).strip()

    # 2. Parse Qwen's <tool_call> markers ourselves; take the first known tool.
    for call in _extract_tool_calls(content):
        if call["name"] in TOOL_NAMES:
            return _as_tool_call(call["name"], call.get("arguments"))

    # 3. No tool call -> conversational reply (strip any stray markers).
    reply = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL).strip()
    return {"type": "chat", "reply": reply}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
