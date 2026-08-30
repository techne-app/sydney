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
MODEL_NAME = "google_gemma-4-26B-A4B-it-Q3_K_M.gguf"  # was Qwen3-4B-Q4_K_M.gguf
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
        "You are the assistant for a Hacker News explorer app. "
        "Use a tool only when the user wants an action (search for discussions, "
        "or summarize the thread they have pinned). Otherwise, reply "
        "conversationally in plain text."
    )
    if pinned:
        # Deliberately no Summary line. The extension's _buildContextString()
        # included one, but handing the model the summary means "what is this
        # about?" gets answered from context instead of calling the tool —
        # Gemma did exactly that, and saying "you MUST call the tool" did not
        # stop it. Title/theme/category are enough to resolve "this"; the
        # summary the user actually sees comes from the frontend's own copy of
        # pinnedThread via ThreadSummaryTool, never from here.
        context = (
            "\n\nContext: User has pinned this thread:\n"
            f"- Title: \"{pinned.get('story_title', '')}\"\n"
            f"- Theme: \"{pinned.get('theme', '')}\"\n"
            f"- Category: \"{pinned.get('category', '')}\"\n"
            f"- Comments: {pinned.get('comment_count', '')}\n"
            "When the user says \"this thread\", \"this discussion\", \"this\", "
            "\"here\", or asks what it's about / for the key points, they mean "
            "this pinned thread, and you must call summarize_pinned_thread."
        )
    else:
        context = "\n\nContext: No thread currently pinned."
    return instruction + context


# Gemma 4 emits tool calls in its own non-JSON syntax, straight from the chat
# template baked into the GGUF:
#     <|tool_call>call:search_threads{keyword_filter:<|"|>rust<|"|>}<tool_call|>
# Note the delimiters are NOT a matched pair, and string values are wrapped in
# <|"|> rather than quotes. The closing marker is optional here so a reply cut
# off by max_tokens still yields the call.
_GEMMA_CALL_RE = re.compile(
    r"<\|tool_call>call:([A-Za-z_]\w*)\s*\{(.*?)\}(?:<tool_call\|>|\s*$)", re.DOTALL
)
# key:<|"|>string<|"|>  or  key:bare_token  (numbers, booleans)
_GEMMA_ARG_RE = re.compile(r"(\w+)\s*:\s*(?:<\|\"\|>(.*?)<\|\"\|>|([^,}]+))", re.DOTALL)

# Gemma's thinking channel, which must not reach the user as a chat reply.
_GEMMA_THOUGHT_RE = re.compile(r"<\|channel>thought.*?<channel\|>", re.DOTALL)


def _parse_gemma_args(body: str) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    for key, quoted, bare in _GEMMA_ARG_RE.findall(body):
        if quoted:
            args[key] = quoted
        elif bare.strip():
            raw = bare.strip()
            try:
                args[key] = json.loads(raw)  # numbers, true/false, null
            except json.JSONDecodeError:
                args[key] = raw
    return args


def _extract_tool_calls(content: str) -> List[Dict[str, Any]]:
    """Parse tool-call markers out of the reply text.

    llama-cpp-python doesn't parse either model's tool-call format for us, so we
    do it here. Handles Gemma 4's <|tool_call>call:name{...} syntax and Qwen's
    <tool_call>{json}</tool_call>, so swapping the model back doesn't break
    routing. Malformed blocks are skipped rather than fatal — /route falls
    through to chat, which is the safer failure.
    """
    calls: List[Dict[str, Any]] = []

    for name, body in _GEMMA_CALL_RE.findall(content):
        calls.append({"name": name, "arguments": _parse_gemma_args(body)})

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

    # Native model template (no chat_format override) — it supports `tools`.
    # The generic chatml-function-calling handler is deliberately avoided: it
    # forces a tool call on every message, including "hello".
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

    content = msg.get("content") or ""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)      # Qwen
    content = _GEMMA_THOUGHT_RE.sub("", content).strip()                       # Gemma

    # 2. Parse the model's tool-call markers ourselves; take the first known tool.
    for call in _extract_tool_calls(content):
        if call["name"] in TOOL_NAMES:
            return _as_tool_call(call["name"], call.get("arguments"))

    # 3. No tool call -> conversational reply (strip any stray markers).
    reply = _GEMMA_CALL_RE.sub("", content)
    reply = re.sub(r"<tool_call>.*?</tool_call>", "", reply, flags=re.DOTALL).strip()
    return {"type": "chat", "reply": reply}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
