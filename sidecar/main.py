from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from llama_cpp import Llama
import os
import re
import json

import numpy as np

import corpus
from tools import TOOLS, TOOL_NAMES  # single source of truth for tool schemas

app = FastAPI()

# Allow requests from Tauri/localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup
import sys
BASE_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))


def resolve_model(name: str) -> str:
    """Locate a GGUF in the bundled .app or in the dev checkout.

    Production: bundled inside .app at Contents/Resources/models/ — tauri.conf.json
    maps the whole sidecar/models/ directory there, so any GGUF dropped in it is
    picked up with no config change.
    Dev: main.py runs from sidecar/, so the models sit alongside it.
    """
    bundled = os.path.join(BASE_DIR, "..", "Resources", "models", name)
    local = os.path.join(BASE_DIR, "models", name)
    return bundled if os.path.exists(bundled) else local


MODEL_NAME = "google_gemma-4-26B-A4B-it-Q3_K_M.gguf"  # was Qwen3-4B-Q4_K_M.gguf
MODEL_PATH = resolve_model(MODEL_NAME)
# The embedding model MUST be the one that produced the stored vectors —
# nomic-embed-text-v1.5, 768-dim (techne-pipeline/scripts/start_embedding_server.sh).
# A query embedded by any other model lands in a different vector space, and
# cosine against it returns noise rather than weak matches.
EMBED_MODEL_NAME = "nomic-embed-text-v1.5.Q8_0.gguf"
EMBED_MODEL_PATH = resolve_model(EMBED_MODEL_NAME)

print(f"Loading model from {MODEL_PATH}...")
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,   # -1 = offload all layers to Metal GPU
    n_ctx=8192,        # 100 rerank candidates run ~3k tokens; 4096 was too tight
    verbose=False,
)
print("Model loaded.")

# CPU-only on purpose: the pipeline runs its embedder with -ngl 0 to keep the
# Metal context free for the chat model, and that matters more here with 12GB of
# Gemma already on the GPU. Embedding one short query per search is cheap on CPU.
print(f"Loading embedding model from {EMBED_MODEL_PATH}...")
embedder = Llama(
    model_path=EMBED_MODEL_PATH,
    embedding=True,
    n_gpu_layers=0,
    verbose=False,
)
print("Embedding model loaded.")

# Start syncing the 30-day thread corpus in the background. Non-blocking: this
# runs while Gemma loads, so the ~44s download costs the user nothing.
corpus.start()


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


class SearchRequest(BaseModel):
    query: str
    limit: int = 3


# How many candidates the embedding stage hands to the reranker. Mirrors the
# pipeline's feed generation, which cosines to 100 and then has the LLM pick ~6;
# we ask for 3.
RERANK_CANDIDATES = 100


def _build_rerank_prompt(query: str, candidates, limit: int) -> str:
    listing = "\n".join(
        f"{i + 1}. {meta.get('story_title', '')} — {meta.get('theme', '')}"
        for i, (meta, _) in enumerate(candidates)
    )
    return (
        f"A user searched Hacker News for: \"{query}\"\n\n"
        f"Below are {len(candidates)} candidate discussions, already filtered by "
        "semantic similarity. Pick the "
        f"{limit} most genuinely relevant to what the user is asking for.\n\n"
        f"{listing}\n\n"
        f"Reply with ONLY the {limit} numbers, best first, separated by commas. "
        "No explanation, no other text."
    )


def _parse_rerank_picks(text: str, count: int, limit: int):
    """Pull the chosen indices out of the model's reply.

    Deliberately just scrapes integers rather than asking for JSON or a tool
    call — Gemma's structured output is quirky enough (see _extract_tool_calls),
    and a bare list of numbers is the one format it is hard to get wrong. Any
    out-of-range or duplicate number is dropped; if nothing usable comes back
    the caller falls back to plain cosine order.
    """
    picks = []
    for token in re.findall(r"\d+", text):
        index = int(token) - 1
        if 0 <= index < count and index not in picks:
            picks.append(index)
        if len(picks) == limit:
            break
    return picks


@app.post("/search")
def search(request: SearchRequest):
    if not corpus.is_ready():
        return {"results": [], "error": "Thread data is still loading, try again shortly.",
                "corpus": corpus.status()}

    query = request.query.strip()
    if not query:
        return {"results": [], "error": "Empty query."}

    # "search_query: " is required, not decorative: every theme was stored with
    # the "search_document: " prefix, and nomic places queries and documents
    # differently. The wrong prefix degrades relevance silently — no error.
    embedding = embedder.create_embedding(f"search_query: {query}")
    vector = np.asarray(embedding["data"][0]["embedding"], dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    # Pull well past RERANK_CANDIDATES, then keep only the best-scoring thread
    # per story. A row is one comment thread, not one story, so a popular post
    # contributes dozens of near-identical rows — without this, a query like
    # "self hosting email" returns three threads from the same discussion.
    # (Note this is per-story on purpose; the corpus itself is still keyed by
    # thread_id, which is what makes delta merges correct.)
    ranked = corpus.search(vector, k=RERANK_CANDIDATES * 4)
    seen_stories = set()
    candidates = []
    for meta, score in ranked:
        story_id = meta.get("story_id")
        if story_id in seen_stories:
            continue
        seen_stories.add(story_id)
        candidates.append((meta, score))
        if len(candidates) == RERANK_CANDIDATES:
            break

    if not candidates:
        return {"results": [], "error": "No matching discussions found."}

    # Rerank the shortlist with Gemma. If it returns nothing usable, fall through
    # to cosine order rather than failing the search.
    picks = []
    try:
        completion = llm.create_chat_completion(
            messages=[{"role": "user",
                       "content": _build_rerank_prompt(query, candidates, request.limit)}],
            temperature=0.1,
            max_tokens=64,
        )
        reply = completion["choices"][0]["message"].get("content") or ""
        reply = _GEMMA_THOUGHT_RE.sub("", reply)
        picks = _parse_rerank_picks(reply, len(candidates), request.limit)
    except Exception as error:
        print(f"[search] rerank failed, using cosine order: {error}")

    if not picks:
        picks = list(range(min(request.limit, len(candidates))))

    results = []
    for index in picks:
        meta, score = candidates[index]
        results.append({**meta, "score": score})

    return {"results": results}


@app.get("/health")
def health():
    return {"status": "ok", "corpus": corpus.status()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
