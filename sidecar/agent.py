"""The ADK agent — one loop that can call tools repeatedly before answering.

The problem this solves: routing used to give the model exactly one tool call
per turn and then stop. When a question needed something the model didn't have
— the next page of results, what a thread actually says — it couldn't ask, so
it invented. Every hallucination this project hit came from that gap.

Here the model calls a tool, sees the result, and decides again. ADK's Runner
drives that loop; the tools below are plain Python functions, and ADK builds
their schemas from the signature and docstring. Nothing hand-written.
"""
import os
from typing import Any, Dict, Optional

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

import corpus
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

BASE_INSTRUCTION = (
    "You help a user explore Hacker News. Use search_threads to find "
    "discussions on a topic, and get_thread to read one in detail when the "
    "user asks about a specific result. Answer from what the tools return — do "
    "not invent discussions, titles, or links. If you do not have enough "
    "information to answer, say so plainly rather than guessing. Do not show "
    "thread ids to the user; they are for your own tool calls."
    "\n\nA message may begin with a note about a discussion the user has open "
    "in front of them. When they say \"this thread\", \"this discussion\", "
    "\"this\", or \"this one\", they mean that one — not an item from a list "
    "of search results. Answer from the note; there is no need to search."
)

# The thread the user has dragged into the chat travels WITH the message it was
# sent alongside, rather than living in session state.
#
# State was the wrong home for it, and failed in a way worth recording. State
# feeds the system prompt, which has no position in time — as far as the model
# can tell it has always said this — while a thread just read with get_thread
# sits in the newest message. Asked to explain "this thread", the model picked
# the recent one. That is a correct reading of what it could see, and no amount
# of instruction wording outranked it (measured: it still chose the wrong thread
# with an explicit "the user opened it just now" line in the prompt).
#
# An attachment belongs to the turn it was attached to. Putting it there makes
# the pin the most recent thing in the conversation, which is what the user
# means by "this", and it makes history self-describing: an old turn keeps its
# attachment, so unpinning cannot make the model disown an earlier answer as
# invented — which is what deleting the state used to cause.
def _with_attachment(message: str, thread: Optional[Dict[str, Any]]) -> str:
    if not thread:
        return message
    return (
        "[The user has this discussion open in front of them:\n"
        f"\"{thread.get('story_title', '')}\" — {thread.get('theme', '')}.\n"
        f"What people said: {thread.get('summary', '')}]\n\n"
        f"{message}"
    )


def search_threads(keyword_filter: str) -> Dict[str, Any]:
    """Search Hacker News discussion threads by topic.

    Args:
        keyword_filter: The topic or keywords to search discussions for.
    """
    # ADK reads this function's name, type hints and docstring to build the
    # tool schema it sends the model. There is no hand-written schema to keep
    # in sync — the function is the tool.
    print(f"[agent] search_threads(keyword_filter={keyword_filter!r})")
    return search.run_search(keyword_filter, limit=3)


def get_thread(thread_id: int) -> Dict[str, Any]:
    """Get the full summary of one Hacker News discussion thread.

    Use this when the user asks about a specific discussion you have already
    found — for example "what was the second one about?" or "tell me more about
    that Rust thread". Search results carry only a title and a one-line description;
    this returns what was actually discussed, plus the link to it.

    Args:
        thread_id: The id of the thread, taken from an earlier search result.
    """
    print(f"[agent] get_thread(thread_id={thread_id!r})")
    thread = corpus.get(thread_id)
    if thread is None:
        return {
            "error": (
                "No thread with that id is in the last 30 days of data. "
                "Search for the topic instead."
            )
        }
    return corpus.public_view(thread, detail=True)


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
    instruction=BASE_INSTRUCTION,
    tools=[search_threads, get_thread],
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


async def send(
    session_id: str,
    message: str,
    pinned_thread: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one turn.

    session_id is the frontend's conversation id, so a Dexie conversation and an
    ADK session are the same thing seen from two sides — the UI store holds
    rendered prose, the agent store holds the structure it reasons over.

    Returns the reply, the tool calls made, and the raw results those tools
    produced. The frontend renders links from `results` rather than from the
    model's prose: the model retypes a 50-character URL every time it writes a
    list, and one wrong character is a dead link nobody notices until it's
    clicked.
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

    print(f"[agent] attached={'yes: ' + str(pinned_thread.get('story_title')) if pinned_thread else 'none'}")

    reply_parts: list[str] = []
    tool_calls: list[Dict[str, Any]] = []
    results: list[Dict[str, Any]] = []

    async for event in _runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=_with_attachment(message, pinned_thread))],
        ),
    ):
        if not (event.content and event.content.parts):
            continue
        for part in event.content.parts:
            call = getattr(part, "function_call", None)
            response = getattr(part, "function_response", None)
            if call:
                tool_calls.append({"name": call.name, "arguments": dict(call.args or {})})
            elif response:
                # Keep the structured output so the UI can render it directly.
                # search_threads returns {"results": [...]}; get_thread returns a
                # single thread. Both need capturing, or a turn that reads one
                # discussion leaves the user with prose and no way to open it.
                payload = response.response
                if not isinstance(payload, dict):
                    continue
                if isinstance(payload.get("results"), list):
                    results.extend(payload["results"])
                elif payload.get("thread_id"):
                    # The link comes from the corpus, not from the payload the
                    # model saw — see corpus.public_view. Looking it up here is
                    # also what guarantees it is right: a model retyping a
                    # 50-character URL gets one character wrong eventually, and
                    # a dead link is only noticed when somebody clicks it.
                    row = corpus.get(payload["thread_id"])
                    if row:
                        results.append({
                            "thread_id": row.get("thread_id"),
                            "title": row.get("story_title"),
                            "link": row.get("anchor"),
                        })
            elif getattr(part, "text", None):
                reply_parts.append(part.text)

    # The model emits text at several points in the loop — a preamble before a
    # tool call, then the real answer. The last non-empty part is the answer;
    # the earlier ones are it narrating what it is about to fetch.
    reply = next((t.strip() for t in reversed(reply_parts) if t.strip()), "")
    return {"reply": reply, "tool_calls": tool_calls, "results": results}
