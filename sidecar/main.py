from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import os
import re

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

import agent
import search
from llm import chat_completion, llama_server_ready

# nomic loads here; Gemma is served by llama-server (see llm.py).
search.load_embedder()

# Open the session database and build the agent's runner.
agent.start()

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


@app.post("/search")
def search_endpoint(request: SearchRequest):
    return search.run_search(request.query, request.limit)


# ---------------------------------------------------------------------------
# /sessions/{id}/messages — the agent. One endpoint serves every conversation;
# the session id in the path selects which history to load. Runs the tool loop,
# so unlike /route it can call a tool, read the result, and call another before
# answering.
# ---------------------------------------------------------------------------

class AgentMessage(BaseModel):
    message: str


@app.post("/sessions/{session_id}/messages")
async def send_message(session_id: str, body: AgentMessage):
    if not llama_server_ready():
        return {"reply": "", "error": "The model server is still starting up."}
    try:
        return await agent.send(session_id, body.message)
    except Exception as error:
        print(f"[agent] failed: {error}")
        return {"reply": "", "error": str(error)}


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
