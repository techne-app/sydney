from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from llama_cpp import Llama
import httpx
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


# Gemma is served by llama-server, not loaded in-process. It has to be: only
# llama-server's --jinja applies the model's own chat template and returns tool
# calls as a structured `tool_calls` field. llama-cpp-python hands back the raw
# `<|tool_call>call:...` text, which is why this file used to carry a
# hand-written parser for Gemma's syntax.
#
# Note this cannot be a partial move — 12GB in-process plus 12GB in llama-server
# does not fit in 24GB of RAM.
LLAMA_SERVER_URL = os.environ.get("TECHNE_LLAMA_SERVER", "http://127.0.0.1:8081")
LLAMA_TIMEOUT = 300.0

# nomic stays in-process. Only the chat model needs --jinja (tool calling), and
# nomic is 139MB on CPU embedding one short query per search — running a second
# server for that would add a process and a failure mode for no gain.
#
# It MUST be the model that produced the stored vectors — nomic-embed-text-v1.5,
# 768-dim (techne-pipeline/scripts/start_embedding_server.sh). A query embedded
# by any other model lands in a different vector space, and cosine against it
# returns noise rather than weak matches.
EMBED_MODEL_NAME = "nomic-embed-text-v1.5.Q8_0.gguf"
EMBED_MODEL_PATH = resolve_model(EMBED_MODEL_NAME)

print(f"Loading embedding model from {EMBED_MODEL_PATH}...")
embedder = Llama(
    model_path=EMBED_MODEL_PATH,
    embedding=True,
    n_gpu_layers=0,   # CPU: keeps the Metal context free for llama-server's Gemma
    verbose=False,
)
print("Embedding model loaded.")


def chat_completion(messages: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
    """Call Gemma on llama-server (OpenAI-compatible /v1/chat/completions).

    Replaces llm.create_chat_completion(). Returns the raw `message` object, so
    callers read `tool_calls` and `content` the same way they did before.

    Thinking is disabled. Gemma's chain-of-thought is billed against the same
    max_tokens as the answer, and it will happily spend the whole budget
    reasoning and emit nothing — `finish_reason: "length"`, empty content. That
    produced "Sorry, I didn't catch that" for any follow-up question, and left
    the reranker (max_tokens=64) silently falling back to cosine order.

    Neither call needs it: routing is a classification, reranking is picking
    three numbers. It also costs ~500 tokens at 28 tok/s — around 18 seconds of
    latency for nothing.

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

# Start syncing the 30-day thread corpus in the background. Non-blocking: this
# runs while Gemma loads, so the ~44s download costs the user nothing.
corpus.start()


class Message(BaseModel):
    role: str
    content: str
    # Tool turns, replayed by the frontend so the model can tell its own past
    # tool output from prose it wrote. Without these it reads previous search
    # results as its own writing and answers the next search by imitating the
    # format — inventing threads and links instead of calling the tool.
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

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


def _as_tool_call(name: str, arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    return {"type": "tool_call", "name": name, "arguments": arguments or {}}


def _to_chat_message(m: Message) -> Dict[str, Any]:
    """Rebuild a message for the chat template, keeping tool fields when present."""
    out: Dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        # An assistant turn that called a tool carries the call, not prose.
        out["content"] = m.content or None
        out["tool_calls"] = m.tool_calls
    if m.tool_call_id:
        out["tool_call_id"] = m.tool_call_id
    if m.name:
        out["name"] = m.name
    return out


@app.post("/route")
def route(request: RouteRequest):
    messages = [{"role": "system", "content": _build_route_system_prompt(request.pinned_thread)}]
    messages += [_to_chat_message(m) for m in request.messages]

    # llama-server --jinja applies Gemma's own chat template and returns tool
    # calls as a structured `tool_calls` field, so there is nothing to parse.
    msg = chat_completion(
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.1,   # low = consistent routing / structured output
        max_tokens=512,
    )

    def _usable(name: str) -> bool:
        """Reject calls that cannot succeed given the current context.

        Summarizing a pinned thread with nothing pinned has no valid outcome —
        the executor would only report that there is no thread. The model does
        reach for it on phrasings like "what was the second one about?", so
        enforce the precondition here rather than trusting the prompt.

        Note: search_threads is deliberately NOT guarded for a missing keyword.
        Suppressing it and replying "what would you like me to search for?" put
        that question in the history, which made the model drop the argument
        again on the next turn — a loop that never recovered.
        """
        if name == "summarize_pinned_thread" and not request.pinned_thread:
            print("[route] ignoring summarize_pinned_thread — nothing is pinned")
            return False
        return name in TOOL_NAMES

    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function", {})
        if _usable(fn.get("name")):
            return _as_tool_call(fn["name"], fn.get("arguments"))

    # No tool call -> conversational reply. Gemma's reasoning arrives in a
    # separate `reasoning_content` field, so `content` needs no cleaning.
    reply = (msg.get("content") or "").strip()

    # A suppressed tool call leaves content empty — the whole reply WAS the
    # call. Ask again without tools so the model has to answer in words.
    if not reply:
        retry = chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )
        reply = (retry.get("content") or "").strip()

    # The retry can come back empty too. /route is now the only path to the
    # model — there's no /chat fallback behind it — so an empty reply would
    # leave the user looking at nothing at all.
    if not reply:
        reply = "Sorry — I didn't catch that. Could you rephrase?"

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
    call — a bare list of numbers is the one format that is hard to get wrong,
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
        completion = chat_completion(
            messages=[{"role": "user",
                       "content": _build_rerank_prompt(query, candidates, request.limit)}],
            temperature=0.1,
            max_tokens=64,
        )
        picks = _parse_rerank_picks(completion.get("content") or "", len(candidates), request.limit)
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
    # llama_server is reported separately because the sidecar starts fine
    # without it and only fails when a request needs Gemma. Surfacing it here
    # makes "the model server isn't up" diagnosable instead of a 500 later.
    return {
        "status": "ok",
        "llama_server": llama_server_ready(),
        "corpus": corpus.status(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
