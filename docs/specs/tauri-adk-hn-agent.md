# Techne HN Agent: Tauri + ADK + llama-server

## Vision

A desktop agent that lets you chat with all of Hacker News. Ask questions, discover trends, and analyze discussions — all locally, no cloud LLM required.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Tauri Shell (Rust)                             │
│  - Spawns llama-server + Python sidecar         │
│  - Manages process lifecycle                    │
│  - Serves React frontend                        │
└────────┬──────────────────────────┬─────────────┘
         │                          │
         ▼                          ▼
┌──────────────────┐    ┌───────────────────────────┐
│  llama-server    │    │  Python Sidecar (FastAPI)  │
│  (Port 41820)    │    │  (Port 41821)              │
│                  │    │                             │
│  - GGUF model    │◄───│  - ADK Agent               │
│  - --jinja       │    │  - LiteLLM → llama-server  │
│  - Tool calling  │    │  - Tool registry           │
│  - OpenAI API    │    │  - Session management      │
└──────────────────┘    └──────────┬────────────────┘
                                  │
                    ┌─────────┬───┴───┬─────────┐
                    ▼         ▼       ▼         ▼
                ┌───────┐ ┌──────┐ ┌───────┐ ┌──────┐
                │Search │ │Thread│ │Trend- │ │User  │
                │Tool   │ │Tool  │ │ing    │ │Tool  │
                └───┬───┘ └──┬───┘ └───┬───┘ └──┬───┘
                    │        │         │        │
                    └────────┴────┬────┴────────┘
                                 ▼
                  ┌─────────────────────────┐
                  │  Techne Backend         │
                  │  (Azure Functions)      │
                  │  - All HN threads       │
                  │  - All thread tags      │
                  │  - Search API           │
                  └─────────────────────────┘
```

## Components

### 1. Tauri Shell (Rust)

**Role**: Process orchestrator. Spawns and manages child processes, serves the UI.

**Current state**: Minimal — one command (`open_external_url`), no sidecar management.

**Changes needed**:

```
src-tauri/
├── src/
│   ├── main.rs              # (existing)
│   ├── lib.rs               # Add process spawning
│   └── sidecar.rs           # NEW: llama-server + Python lifecycle
├── Cargo.toml               # Add tauri-plugin-shell
└── tauri.conf.json           # Register sidecar binaries
```

**Key responsibilities**:
- Spawn `llama-server` on startup with the bundled GGUF model
- Wait for llama-server health check (`GET /health`) before spawning Python sidecar
- Spawn Python sidecar (`uvicorn sidecar.main:app`)
- Wait for sidecar health check (`GET /health`) before showing the UI
- Kill both processes on app quit
- Expose `restart_agent` command to frontend for recovery

**Startup sequence**:
```
App launches
  → Spawn llama-server (port 41820, --jinja flag for tool calling)
  → Poll GET http://localhost:41820/health until ready
  → Spawn Python sidecar (port 41821)
  → Poll GET http://localhost:41821/health until ready
  → Frontend loads, connects to sidecar
```

### 2. llama-server

**Role**: LLM inference engine. Serves an OpenAI-compatible API with tool calling support.

**Replaces**: In-browser WebLLM and the current `llama-cpp-python` in-process model loading in `sidecar/main.py`.

**Why separate from Python**: Decouples model loading from the agent. The Python sidecar can restart without reloading the model (~10s+ cold start). Also enables future model hot-swapping.

**Launch command**:
```bash
llama-server \
  -m models/Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 41820 \
  --jinja \
  -ngl -1 \
  -c 16384
```

**Model choice**: Start with Hermes 2 Pro 8B (Q4_K_M) for battle-tested tool calling. Can swap to Functionary v3.2 or Gemma 4 later — just change the GGUF file and restart.

**Flags**:
- `--jinja`: Required. Enables the model's chat template for proper tool calling.
- `-ngl -1`: Offload all layers to Metal GPU.
- `-c 16384`: Context window. 16K gives room for system prompt (~500 tokens) + tool definitions (~500) + search results (~3K) + thread comments (~5K) + conversation history + model reasoning. The memory overhead is ~2GB above the base model on Apple Silicon.

### 3. Python Sidecar (FastAPI + ADK)

**Role**: The agent brain. Receives user messages, reasons about them, calls tools, returns responses.

**Replaces**: All of `src/utils/intentDetector.ts`, `src/utils/tools/`, `src/prompts/`, `src/utils/webLLMClient.ts`, and the current `sidecar/main.py`.

```
sidecar/
├── main.py                  # FastAPI app, /chat, /chat/stream, /health
├── agent.py                 # create_agent() — assembles agent from tools
├── agents/                  # Sub-agent definitions (empty in v1, seam for growth)
│   └── __init__.py
├── tools/
│   ├── __init__.py          # Exports all tool functions
│   ├── search.py            # search_threads — keyword/tag/date search
│   ├── thread.py            # get_thread — full thread with comments
│   ├── trending.py          # get_trending — trending threads/tags
│   └── user.py              # get_user_threads — threads by HN user
├── state.py                 # State key constants + namespace helpers
├── prompts/
│   └── system.py            # System prompt for the HN agent
└── pyproject.toml
```

#### main.py — API Surface (pseudocode)

The actual ADK API uses a `Runner` that takes a session service, not direct agent calls. This pseudocode shows the intent — exact API calls will differ during implementation.

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService  # DatabaseSessionService in Phase 2
from google.genai import types
from agent import create_agent

app = FastAPI()
session_service = InMemorySessionService()  # swap to DatabaseSessionService later
agent = create_agent()
runner = Runner(agent=agent, app_name="techne", session_service=session_service)

USER_ID = "local"  # single-user desktop app

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    session = await session_service.get_session(
        app_name="techne", user_id=USER_ID, session_id=request.session_id
    ) or await session_service.create_session(
        app_name="techne", user_id=USER_ID
    )

    content = types.Content(
        role="user", parts=[types.Part(text=request.message)]
    )

    async def generate():
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session.id, new_message=content
        ):
            # Emit tool-call events so frontend can show "Searching...", "Reading thread..."
            if event.actions and event.actions.tool_calls:
                for tc in event.actions.tool_calls:
                    yield sse_event("tool_call", {
                        "tool": tc.function.name,
                        "args": tc.function.arguments
                    })
            # Emit text chunks for streaming response
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        yield sse_event("text", {"content": part.text})

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/health")
def health():
    return {"status": "ok"}
```

**SSE event types** the frontend should handle:
- `tool_call` — agent is invoking a tool (display status: "Searching...", "Reading thread...")
- `tool_result` — tool returned data (optionally render thread cards)
- `text` — agent is generating response text (stream to UI)
- `agent_transfer` — (v1.5+) control moved to a sub-agent

#### agent.py — ADK Agent Setup

```python
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from tools import search_threads, get_thread, get_trending, get_user_threads
from prompts.system import SYSTEM_PROMPT

def create_agent() -> Agent:
    model = LiteLlm(
        model="openai/hermes-2-pro",
        api_base="http://localhost:41820/v1",
        api_key="not-needed"
    )

    return Agent(
        name="hn_agent",
        model=model,
        instruction=SYSTEM_PROMPT,
        tools=[
            search_threads,
            get_thread,
            get_trending,
            get_user_threads,
        ]
    )
```

#### System Prompt

```python
SYSTEM_PROMPT = """You are Techne, an AI assistant that helps users explore
and understand Hacker News discussions.

You have access to the complete archive of Hacker News threads, all tagged
by topic and theme. Use your tools to find relevant discussions, analyze
trends, and synthesize insights.

When a user asks a question:
1. Think about which tools will help you answer it.
2. Search for relevant threads. Use multiple searches if needed.
3. Fetch full thread details when you need to read the actual discussion.
4. Synthesize your findings into a clear, direct answer.

Always cite specific HN threads when referencing discussions.
Prefer concrete data over vague summaries.
If you're not sure, search more before answering.
"""
```

### 4. Tools

Each tool is a decorated Python function. ADK handles the schema generation and the tool-calling loop.

All tools use a shared `httpx.Client` with a 15-second timeout (Azure Functions cold starts can take 5-10s). Every tool checks the response status and returns structured errors the agent can reason about.

#### Common pattern

```python
import httpx

BACKEND_URL = "https://techne-pipeline-func-prod.azurewebsites.net"
client = httpx.Client(timeout=15.0)

def _call_backend(endpoint: str, payload: dict) -> dict:
    """POST to backend, return JSON or structured error."""
    response = client.post(f"{BACKEND_URL}{endpoint}", json=payload)
    if response.status_code != 200:
        return {"error": f"Backend returned {response.status_code}", "endpoint": endpoint}
    return response.json()
```

#### search.py

```python
from google.adk.tools import tool

@tool
def search_threads(
    query: str,
    tags: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    sort_by: str = "relevance",
    limit: int = 20,
    offset: int = 0
) -> dict:
    """Search Hacker News threads by keyword, tags, and date range.

    Args:
        query: Search query (keywords or natural language).
        tags: Optional list of tags to filter by (e.g. ["ai", "rust"]).
        date_from: Start date in YYYY-MM-DD format.
        date_to: End date in YYYY-MM-DD format.
        sort_by: Sort order — "relevance", "date", or "comments".
        limit: Max number of results (default 20, max 50).
        offset: Pagination offset (default 0).

    Returns:
        Dict with 'threads' list and 'total_count'.
    """
    return _call_backend("/api/agent/search", {
        "query": query,
        "tags": tags,
        "date_from": date_from,
        "date_to": date_to,
        "sort_by": sort_by,
        "limit": min(limit, 50),
        "offset": offset,
    })
```

#### thread.py

```python
@tool
def get_thread(
    thread_id: str,
    max_comments: int = 50,
    sort_comments_by: str = "top"
) -> dict:
    """Get the full details of a Hacker News thread including all comments.

    Args:
        thread_id: The HN story/thread ID.
        max_comments: Max comments to return (default 50).
        sort_comments_by: Comment sort order — "top" (karma), "recent", or "oldest".

    Returns:
        Dict with story metadata, tags, and comment list.
    """
    return _call_backend("/api/agent/thread", {
        "thread_id": thread_id,
        "max_comments": max_comments,
        "sort_comments_by": sort_comments_by,
    })
```

#### trending.py

```python
@tool
def get_trending(
    time_window: str = "24h",
    category: str | None = None,
    tag: str | None = None,
    limit: int = 10
) -> dict:
    """Get currently trending threads or tags on Hacker News.

    Args:
        time_window: Time window — "1h", "6h", "24h", "7d", "30d".
        category: Optional category to filter by (e.g. "AI/ML").
        tag: Optional tag to filter trending threads by.
        limit: Number of results (default 10).

    Returns:
        Dict with 'threads' list sorted by activity/karma density.
    """
    return _call_backend("/api/agent/trending", {
        "time_window": time_window,
        "category": category,
        "tag": tag,
        "limit": limit,
    })
```

#### user.py

```python
@tool
def get_user_threads(
    username: str,
    limit: int = 20
) -> dict:
    """Get threads and comments by a specific Hacker News user.

    Args:
        username: The HN username.
        limit: Max results (default 20).

    Returns:
        Dict with user's threads and comment activity.
    """
    return _call_backend("/api/agent/user", {
        "username": username,
        "limit": limit,
    })
```

### 5. Frontend

**Role**: Thin chat UI. All intelligence is in the sidecar.

**Replaces**: All of the current intent detection, tool orchestration, and WebLLM code in the frontend. The frontend becomes a pure chat client.

**What stays**:
- `ChatInterface.tsx` — but simplified to just send/receive messages
- `ThreadCard.tsx` — for rendering HN threads in responses
- `Modal.tsx`, `MemoryCard.tsx` — UI chrome
- Tailwind styling, CSS variables

**What gets removed**:
- `src/utils/intentDetector.ts`
- `src/utils/intentDetectorSingleStep.ts`
- `src/utils/tools/` (entire directory)
- `src/utils/webLLMClient.ts` (Chrome extension version)
- `src/tauri-compat/webLLMClient.ts`
- `src/utils/modelState.ts`
- `src/prompts/` (entire directory)
- `src/background/webllm.ts`
- `src/background/embed.js`

**New frontend client**:

```typescript
// src/utils/agentClient.ts

const SIDECAR_URL = "http://localhost:41821";

export type AgentEventType = "tool_call" | "tool_result" | "text" | "agent_transfer";

export interface AgentEvent {
  type: AgentEventType;
  data: Record<string, unknown>;
}

export interface SendMessageOptions {
  message: string;
  sessionId: string;
  onEvent?: (event: AgentEvent) => void;  // all event types
  onText?: (text: string) => void;         // convenience: accumulated text
  signal?: AbortSignal;                     // cancellation
}

export async function sendMessage(opts: SendMessageOptions): Promise<string> {
  const response = await fetch(`${SIDECAR_URL}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: opts.message, session_id: opts.sessionId }),
    signal: opts.signal,
  });

  if (!response.ok) {
    throw new Error(`Sidecar returned ${response.status}`);
  }

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let fullResponse = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    for (const line of chunk.split("\n")) {
      if (line.startsWith("data: ")) {
        const event: AgentEvent = JSON.parse(line.slice(6));
        opts.onEvent?.(event);

        if (event.type === "text") {
          fullResponse += (event.data as { content: string }).content;
          opts.onText?.(fullResponse);
        }
      }
    }
  }

  return fullResponse;
}
```

The frontend uses `onEvent` to show agent activity:
- `tool_call` with `{tool: "search_threads", args: {...}}` → display "Searching..."
- `tool_call` with `{tool: "get_thread", args: {...}}` → display "Reading thread..."
- `text` → stream response text to the chat bubble
- `agent_transfer` (v1.5+) → show which sub-agent is active

### 6. Session Management

**Role**: Track conversation state across the agent tree.

ADK provides built-in session management with pluggable backends:

- **Phase 1-2**: `InMemorySessionService` — simplest, no persistence. Conversations lost on sidecar restart. Good enough for initial development.
- **Phase 2+**: `DatabaseSessionService(db_url="sqlite+aiosqlite:///sessions.db")` — SQLite-backed persistence. Conversations survive restarts.

```python
# Phase 1
from google.adk.sessions import InMemorySessionService
session_service = InMemorySessionService()

# Phase 2+ (swap one line)
from google.adk.sessions import DatabaseSessionService
session_service = DatabaseSessionService(db_url="sqlite+aiosqlite:///sessions.db")
```

**Note**: ADK's `DatabaseSessionService` requires `aiosqlite` as an async SQLite driver. The session schema has had breaking changes (v0→v1 in ADK 1.22) and has no Alembic migration support yet. For a single-user desktop app, deleting the SQLite file on schema changes is acceptable.

## Agent Topology

ADK is chosen specifically for its multi-agent capabilities — composing specialized sub-agents, deterministic workflow pipelines, and structured delegation. The architecture starts flat and grows into a hierarchy as capabilities expand.

### Why start flat

With a local 8B model (Hermes 2 Pro), LLM-based routing between sub-agents is unreliable. The model already has to decide which tool to call — making it also pick which *agent* to transfer to (exposed as additional `transfer_to_agent_X` tool calls) is strictly harder. Four tools is well within what an 8B model handles reliably.

### Why ADK still earns its keep

ADK's **deterministic workflow agents** (`SequentialAgent`, `ParallelAgent`, `LoopAgent`) don't use an LLM at all — they orchestrate sub-agents programmatically. This is the extension seam: we add structured pipelines without requiring a better routing model.

### v1: Flat single agent

```python
# sidecar/agent.py

root_agent = Agent(
    name="hn_agent",
    model=model,
    instruction=SYSTEM_PROMPT,
    tools=[search_threads, get_thread, get_trending, get_user_threads],
    # No sub_agents — all capabilities are direct tools
)
```

All 4 tools on one agent. The LLM picks tools, ADK runs the tool-calling loop. Simple, reliable with 8B.

### v1.5: First sub-agent (deterministic pipeline)

**Trigger**: When the single agent demonstrably fails at multi-step research — e.g., user asks "compare sentiment on Rust vs Go in kernel discussions" and the model fetches one thread then answers prematurely.

```python
# sidecar/agents/deep_research.py

search_step = Agent(
    name="research_searcher",
    model=model,
    description="Searches for relevant HN threads from multiple angles.",
    instruction="Perform 2-3 targeted searches. Store results in state['research:threads'].",
    tools=[search_threads, get_trending, get_user_threads],
)

analyze_step = Agent(
    name="research_analyzer",
    model=model,
    description="Reads threads and synthesizes findings.",
    instruction="Fetch full details for the most relevant threads. Produce a synthesis.",
    tools=[get_thread],
)

deep_research = SequentialAgent(
    name="deep_research",
    description="Complex research requiring multiple searches and thread analysis.",
    sub_agents=[search_step, analyze_step],
)
```

```python
# sidecar/agent.py (updated)

root_agent = Agent(
    name="hn_agent",
    model=model,
    instruction=SYSTEM_PROMPT,
    tools=[search_threads, get_thread, get_trending, get_user_threads],
    sub_agents=[deep_research],  # ← add pipeline
)
```

**Key insight**: `SequentialAgent` runs search→analyze deterministically. The 8B model never has to decide "should I search more or start analyzing?" — that decision is baked into the pipeline. The root agent only needs to decide "is this a simple query (handle directly) or a research question (delegate to deep_research)?" — a much easier routing decision.

### v2: Specialized sub-agents (requires better model)

When local models improve (or if a cloud model is added), the root agent becomes a thin router:

```
root_agent (LlmAgent — router)
├── discovery_agent (LlmAgent)
│   tools: [search_threads, get_trending, get_user_threads]
│
├── analysis_agent (SequentialAgent — deterministic)
│   ├── fetch_step (LlmAgent) — tools: [get_thread]
│   └── synthesize_step (LlmAgent) — no tools, just reasoning
│
└── monitoring_agent (LlmAgent — new capability)
    tools: [create_alert, list_alerts, check_alerts]
```

This requires the router model to reliably distinguish 3 sub-agents from their descriptions. Not feasible with current 8B models, but straightforward with 30B+ or cloud models.

### State namespace convention

ADK sessions use a flat shared state dict. All agents and tools see the same keys. Namespace from day 1 to enable future agent splitting:

```python
# Tool-domain prefixes
"search:last_query"         # str — most recent search query
"search:last_results"       # list — most recent search results
"search:total_count"        # int — total matches for pagination
"thread:last_read_id"       # str — last thread fetched in full
"trending:last_window"      # str — last time window used
"user:last_lookup"          # str — last username looked up

# Pipeline prefixes (v1.5+)
"research:question"         # str — the research question being investigated
"research:threads"          # list — accumulated search results across steps
```

This costs nothing in v1 (single agent reads/writes all keys) and makes future agent splitting painless — each sub-agent reads only its namespace.

## Dependencies

### Python (sidecar/pyproject.toml)

```toml
[project]
name = "techne-agent"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn>=0.34",
    "google-adk~=1.29.0",
    "litellm>=1.50,!=1.82.0,!=1.82.1,!=1.82.2,!=1.82.3,!=1.82.4,!=1.82.5",  # compromised versions excluded

    "httpx>=0.28",
    "pydantic>=2.10",
    "aiosqlite>=0.20",  # required by DatabaseSessionService
]
```

### Rust (src-tauri/Cargo.toml additions)

```toml
[dependencies]
tauri-plugin-shell = "2"  # For sidecar process management
```

### Binaries to bundle

- `llama-server` — from llama.cpp releases (per-platform)
- GGUF model file — `Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf` (~4.5GB)

## Implementation Order

### Phase 1: Agent core

Get the agent loop working end-to-end.

1. **Set up llama-server** — download binary + GGUF, test tool calling manually with `curl`
2. **Rewrite sidecar** — replace current `sidecar/main.py` with ADK agent + LiteLLM + 2 basic tools (`search_threads`, `get_thread`)
3. **Verify agent loop** — test multi-step reasoning from the terminal (no UI yet)
4. **Wire up frontend** — replace `TauriWebLLMClient` with `agentClient.ts`, simplify `ChatInterface.tsx`
5. **Tauri process management** — spawn llama-server + sidecar from Rust, health checks

**Milestone**: User can chat with HN through the desktop app. Agent searches, fetches threads, and synthesizes answers across multiple tool calls.

### Phase 2: Streaming + more tools

6. **Streaming** — SSE streaming from sidecar to frontend with tool-call events for real-time status
7. **Add remaining tools** — `get_trending`, `get_user_threads`
8. **Session persistence** — swap `InMemorySessionService` for `DatabaseSessionService`, conversation resume on restart

**Milestone**: Streaming UX with agent activity visibility, richer queries, persistent conversations.

### Phase 3: Polish

9. **Error handling** — graceful recovery from llama-server crashes, sidecar timeouts
10. **Model hot-swap** — settings UI to pick a different GGUF without restarting the app
11. **Remove dead code** — delete all WebLLM, intent detection, and in-browser tool orchestration code
12. **Packaging** — bundle llama-server binary and GGUF model with the Tauri app

## What gets deleted

All of these files become dead code once the agent is running in the sidecar:

```
REMOVE src/utils/intentDetector.ts
REMOVE src/utils/intentDetectorSingleStep.ts
REMOVE src/utils/tools/                      (entire directory)
REMOVE src/utils/webLLMClient.ts
REMOVE src/utils/modelState.ts
REMOVE src/tauri-compat/webLLMClient.ts
REMOVE src/prompts/                          (entire directory)
REMOVE src/background/webllm.ts
REMOVE src/background/embed.js
```

The Chrome extension build (webpack) can continue using the old code if needed. The Tauri build (vite) will use the new `agentClient.ts`.

## Backend API Design

### Existing Endpoints (keep as-is)

These serve the Chrome extension and the Tauri sidebar. No changes needed.

```
POST /api/story-tags/
POST /api/thread-tags/
POST /api/thread-cards
```

### New Endpoints (agent tools)

The agent needs 4 new backend endpoints. Designed to map 1:1 to ADK tool functions — each tool makes exactly one backend call.

---

#### `POST /api/agent/search`

The primary search endpoint. Replaces the current client-side approach (fetch top 30 story IDs from Firebase → fetch their tags → semantic match in browser). The backend has all the data, so search belongs there.

**Request:**
```json
{
  "query": "local LLMs replacing cloud APIs",
  "tags": ["ai", "infrastructure"],
  "date_from": "2025-10-01",
  "date_to": "2026-04-11",
  "sort_by": "relevance",
  "limit": 20,
  "offset": 0
}
```

All fields except `query` are optional.

- `query` — free text. Backend decides matching strategy (keyword, semantic, hybrid).
- `tags` — filter to threads tagged with ANY of these. Uses existing tag taxonomy (`thread_theme`, `thread_category` values).
- `date_from` / `date_to` — ISO 8601 date strings. Backend filters by thread `updated_at`.
- `sort_by` — `"relevance"` (default), `"date"`, `"karma"`, `"comments"`.
- `limit` — max results, default 20, max 50.
- `offset` — for pagination. Agent can page through results across tool calls.

**Response:**
```json
{
  "threads": [
    {
      "thread_id": 39812345,
      "story_id": 39812340,
      "story_title": "Why I switched from GPT-4 to local Llama",
      "story_url": "https://example.com/article",
      "anchor": "https://news.ycombinator.com/item?id=39812345",
      "theme": "Local AI vs Cloud AI",
      "category": "AI/ML",
      "comment_count": 247,
      "cumulative_karma": 891,
      "summary": "Discussion comparing cost, latency, and privacy tradeoffs...",
      "updated_at": "2026-03-15T14:30:00Z"
    }
  ],
  "total_count": 143,
  "query": "local LLMs replacing cloud APIs"
}
```

Each thread in the response uses the same shape as `ThreadCardData` (what the frontend already renders), plus `story_url`. This means the frontend's `ThreadCard.tsx` can render search results without changes.

**Why this shape:**
- The agent sees enough metadata (title, theme, category, summary, karma, comments) to decide which threads to drill into with `get_thread`.
- `total_count` tells the agent if it should page for more results.
- `summary` lets the agent synthesize an answer without fetching full threads for simple questions.

---

#### `POST /api/agent/thread`

Get full thread details including the comment tree. This is the "read the actual discussion" tool.

**Request:**
```json
{
  "thread_id": 39812345,
  "max_comments": 50,
  "sort_comments_by": "top"
}
```

- `thread_id` — required. The HN item ID.
- `max_comments` — optional, default 50. Caps comment tree depth/breadth to stay within context window.
- `sort_comments_by` — `"top"` (by karma, default), `"recent"`, `"oldest"`.

**Response:**
```json
{
  "thread_id": 39812345,
  "story_id": 39812340,
  "story_title": "Why I switched from GPT-4 to local Llama",
  "story_url": "https://example.com/article",
  "anchor": "https://news.ycombinator.com/item?id=39812345",
  "author": "pg",
  "posted_at": "2026-03-15T10:00:00Z",
  "score": 891,
  "theme": "Local AI vs Cloud AI",
  "category": "AI/ML",
  "tags": [
    {"tag": "LLM Infrastructure", "type": "thread_theme"},
    {"tag": "AI/ML", "type": "thread_category"}
  ],
  "comment_count": 247,
  "comments": [
    {
      "id": 39812400,
      "parent_id": null,
      "author": "tptacek",
      "text": "The privacy argument is strong but the latency story is more nuanced...",
      "score": 89,
      "depth": 0
    },
    {
      "id": 39812450,
      "parent_id": 39812400,
      "author": "dang",
      "text": "Could you elaborate on the latency comparison?",
      "score": 34,
      "depth": 1
    }
  ]
}
```

**Design decisions:**
- Comments are a **flat list** with `parent_id` for reply structure — ~30-40% more token-efficient than nested `children` arrays. The agent understands "this replied to that" from `parent_id` + `depth` without the structural overhead of nested JSON.
- `max_comments` prevents blowing the context window. 50 top comments is usually enough to understand the discussion. The agent can ask for more if needed.
- Comment `score` lets the agent focus on high-signal comments.
- `posted_at` omitted from comments to save tokens — the agent rarely needs exact comment timestamps.

---

#### `POST /api/agent/trending`

Surface what's active right now. The agent uses this for "what's hot" and "what's happening" queries.

**Request:**
```json
{
  "time_window": "24h",
  "category": "AI/ML",
  "tag": "rust",
  "limit": 10
}
```

All fields optional.

- `time_window` — `"1h"`, `"6h"`, `"24h"` (default), `"7d"`, `"30d"`.
- `category` — filter to a specific category.
- `tag` — filter to threads with this tag.
- `limit` — default 10, max 25.

**Response:**
```json
{
  "threads": [
    {
      "thread_id": 39815000,
      "story_title": "Rust in the Linux kernel: Year 3 retrospective",
      "anchor": "https://news.ycombinator.com/item?id=39815000",
      "theme": "Rust Systems Programming",
      "category": "Programming Languages",
      "comment_count": 412,
      "cumulative_karma": 1203,
      "karma_velocity": 89.5,
      "summary": "Comprehensive look at Rust adoption in the kernel...",
      "updated_at": "2026-04-11T08:00:00Z"
    }
  ],
  "time_window": "24h",
  "generated_at": "2026-04-11T12:00:00Z"
}
```

New field vs search: `karma_velocity` — karma per hour within the time window. This is what makes trending different from search sorted by karma. A thread with 200 karma in 2 hours is more "trending" than one with 500 karma over 3 days.

---

#### `POST /api/agent/user`

Look up a specific HN user's thread participation. For queries like "what has tptacek been saying about security lately?"

**Request:**
```json
{
  "username": "tptacek",
  "date_from": "2026-01-01",
  "date_to": "2026-04-11",
  "limit": 20
}
```

- `username` — required.
- `date_from` / `date_to` — optional date range.
- `limit` — default 20, max 50.

**Response:**
```json
{
  "username": "tptacek",
  "thread_count": 156,
  "threads": [
    {
      "thread_id": 39810000,
      "story_title": "The state of TLS certificate transparency",
      "anchor": "https://news.ycombinator.com/item?id=39810000",
      "theme": "Web Security",
      "category": "Security",
      "user_comment_count": 8,
      "user_karma_in_thread": 234,
      "latest_comment_at": "2026-04-10T16:00:00Z"
    }
  ],
  "top_categories": [
    {"category": "Security", "count": 89},
    {"category": "Cryptography", "count": 34},
    {"category": "Programming Languages", "count": 23}
  ],
  "top_themes": [
    {"theme": "Web Security", "count": 45},
    {"theme": "Encryption", "count": 28}
  ]
}
```

**Design decisions:**
- `user_comment_count` and `user_karma_in_thread` — how active THIS user was in each thread, not the thread totals.
- `top_categories` / `top_themes` — pre-aggregated so the agent can characterize a user's interests without fetching all their threads. Saves tool calls.
- `thread_count` — total threads found (vs. the `limit` returned), so the agent knows coverage.

---

### Endpoint Summary

| Endpoint | Tool | Purpose | Key params |
|----------|------|---------|------------|
| `POST /api/agent/search` | `search_threads` | Find threads by query, tags, dates | `query`, `tags[]`, `date_from/to`, `sort_by` |
| `POST /api/agent/thread` | `get_thread` | Read full discussion + comments | `thread_id`, `max_comments` |
| `POST /api/agent/trending` | `get_trending` | What's active now | `time_window`, `category`, `tag` |
| `POST /api/agent/user` | `get_user_threads` | User's thread participation | `username`, `date_from/to` |

All under `/api/agent/` prefix to separate from existing endpoints.

### Design Principles

1. **1:1 tool-to-endpoint mapping.** Each ADK `@tool` function makes one HTTP call. No multi-step client-side orchestration — that's the agent's job now.

2. **Consistent thread shape.** Every endpoint that returns threads uses the same core fields (`thread_id`, `story_title`, `anchor`, `theme`, `category`, `comment_count`, `cumulative_karma`, `summary`, `updated_at`). The frontend can render any of them with `ThreadCard.tsx`.

3. **Agent-friendly response sizes.** Defaults are tuned for a 16K context window with room for reasoning. `search` returns 20 threads with summaries (~3K tokens). `get_thread` returns 50 comments (~4-5K tokens). A search→read flow leaves ~6-7K for system prompt, history, and model reasoning.

4. **Pre-aggregated where possible.** `user` returns `top_categories` / `top_themes` so the agent doesn't need to fetch 50 threads and run analysis just to answer "what does tptacek care about?"

5. **All POST.** Consistent with existing endpoints. Bodies are JSON. No auth (same as current endpoints — the backend is a public Azure Function).

6. **Pagination via offset.** The agent can page through search results across multiple tool calls: first `offset=0, limit=20`, then `offset=20, limit=20`. `total_count` in the response tells it when to stop.

### What the agent does NOT need from the backend

- **Semantic matching.** The current SearchService does client-side embedding + cosine similarity. With the agent, the LLM *is* the semantic matcher — it reads thread summaries/titles and decides relevance. If the backend has vector search, expose it via the `query` param in `/agent/search`. If not, keyword matching + the agent's judgment is sufficient.

- **Intent detection.** Was handled by `intentDetector.ts`. Now the agent loop + tool calling replaces this entirely.

- **Tag personalization.** Was handled by `personalize.ts` in the browser. Could be added later as a tool or as a backend feature, but not needed for v1.

## Open Questions

1. **Model size vs. quality** — Hermes 2 Pro 8B (Q4_K_M, ~4.5GB) is the safe choice. Could go smaller (Hermes 3B) for faster inference or larger (Functionary 70B) for better reasoning. What hardware should we target?
2. **Streaming from ADK** — Does ADK support streaming tool-call events through LiteLLM to llama-server? Need to verify this path works.
3. **App size** — The GGUF model is ~4.5GB. Should it be bundled or downloaded on first launch?
4. **Backend search strategy** — Does the backend already have full-text or vector search over threads? If keyword-only, is that sufficient for v1 given the agent can interpret results?
