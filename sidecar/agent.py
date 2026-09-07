"""The ADK agent — one loop that can call tools repeatedly before answering.

The problem this solves: /route gives the model exactly one tool call per turn
and then stops. When a question needs something the model doesn't have — the
next page of results, what a thread actually says — it cannot ask, so it
invents. Every hallucination this project has hit came from that gap.

Here the model calls a tool, sees the result, and decides again. ADK's Runner
drives that loop; the tools below are plain Python functions, and ADK builds
their schemas from the signature and docstring. Nothing hand-written.
"""
import os
from typing import Any, Dict

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

import search
from llm import LLAMA_SERVER_URL

APP_NAME = "techne"

# One user per machine — this is a desktop app, not a service. ADK requires a
# user_id, so it is a constant; conversations, not users, are the dimension
# that varies.
USER_ID = "local"

# Beside the corpus cache. Contents/Resources is read-only in a signed .app, so
# anything written at runtime lives under Application Support.
DB_DIR = os.path.expanduser("~/Library/Application Support/com.technesystems.techne")
DB_PATH = os.path.join(DB_DIR, "sessions.db")

INSTRUCTION = (
    "You help a user explore Hacker News. Use search_threads to find "
    "discussions on a topic. Answer from what the tools return — do not invent "
    "discussions, titles, or links. If you do not have enough information to "
    "answer, say so plainly rather than guessing."
)


def search_threads(keyword_filter: str) -> Dict[str, Any]:
    """Search Hacker News discussion threads by topic.

    Args:
        keyword_filter: The topic or keywords to search discussions for.
    """
    # ADK reads this function's name, type hints and docstring to build the
    # tool schema it sends the model — the same information sidecar/tools.py
    # spells out by hand for /route.
    print(f"[agent] search_threads(keyword_filter={keyword_filter!r})")
    return search.run_search(keyword_filter, limit=3)


_agent = LlmAgent(
    model=LiteLlm(
        model="openai/gemma",
        api_base=f"{LLAMA_SERVER_URL}/v1",
        api_key="not-needed",
        # Gemma bills chain-of-thought against max_tokens and will spend the
        # whole budget thinking, returning empty content. Disabled the same way
        # llm.chat_completion does — see the note there.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ),
    name=APP_NAME,
    instruction=INSTRUCTION,
    tools=[search_threads],
)

_session_service: DatabaseSessionService | None = None
_runner: Runner | None = None


def start() -> None:
    """Open the session database and build the runner.

    Sessions persist rather than living in memory: reopening the app and asking
    a follow-up on an old conversation has to still work, and an in-memory store
    loses that on every restart.
    """
    global _session_service, _runner
    os.makedirs(DB_DIR, exist_ok=True)
    _session_service = DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{DB_PATH}")
    _runner = Runner(agent=_agent, app_name=APP_NAME, session_service=_session_service)
    print(f"[agent] sessions at {DB_PATH}")


async def send(session_id: str, message: str) -> Dict[str, Any]:
    """Run one turn. Returns the final reply plus the tool calls it made.

    session_id is the frontend's conversation id, so a Dexie conversation and an
    ADK session are the same thing seen from two sides — the UI store holds
    rendered prose, the agent store holds the structure it reasons over.
    """
    if _runner is None or _session_service is None:
        raise RuntimeError("agent not started — call start() first")

    session = await _session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    if session is None:
        session = await _session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )

    reply_parts: list[str] = []
    tool_calls: list[Dict[str, Any]] = []

    async for event in _runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=message)]),
    ):
        if not (event.content and event.content.parts):
            continue
        for part in event.content.parts:
            call = getattr(part, "function_call", None)
            if call:
                tool_calls.append({"name": call.name, "arguments": dict(call.args or {})})
            elif getattr(part, "text", None):
                reply_parts.append(part.text)

    # The model emits text at several points in the loop — a preamble before a
    # tool call, then the real answer. The last non-empty part is the answer;
    # the earlier ones are it thinking out loud about what to fetch.
    reply = next((t.strip() for t in reversed(reply_parts) if t.strip()), "")
    return {"reply": reply, "tool_calls": tool_calls}
