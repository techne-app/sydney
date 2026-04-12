# Techne HN Agent: Tauri + ADK + Extism + llama-server

## Vision

A desktop agent that lets you chat with all of Hacker News. Ask questions, discover trends, analyze discussions, and run data analysis — all locally, no cloud LLM required.

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
┌─────────────────┐    ┌──────────────────────────┐
│  llama-server   │    │  Python Sidecar (FastAPI) │
│  (Port 8080)    │    │  (Port 8000)              │
│                 │    │                            │
│  - GGUF model   │◄───│  - ADK Agent              │
│  - --jinja      │    │  - LiteLLM → llama-server │
│  - Tool calling │    │  - Tool registry          │
│  - OpenAI API   │    │  - Extism sandbox host    │
└─────────────────┘    │  - Session management     │
                       └──────────┬───────────────┘
                                  │
                       ┌──────────┼──────────┐
                       ▼          ▼          ▼
                   ┌───────┐ ┌────────┐ ┌────────┐
                   │Search │ │Thread  │ │Extism  │
                   │Tool   │ │Tool    │ │Sandbox │
                   │       │ │        │ │(.wasm) │
                   └───┬───┘ └───┬────┘ └────────┘
                       │         │
                       ▼         ▼
               ┌─────────────────────┐
               │  Techne Backend     │
               │  (Azure Functions)  │
               │  - All HN threads   │
               │  - All thread tags  │
               │  - Search API       │
               └─────────────────────┘
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
  → Spawn llama-server (port 8080, --jinja flag for tool calling)
  → Poll GET http://localhost:8080/health until ready
  → Spawn Python sidecar (port 8000)
  → Poll GET http://localhost:8000/health until ready
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
  --port 8080 \
  --jinja \
  -ngl -1 \
  -c 8192
```

**Model choice**: Start with Hermes 2 Pro 8B (Q4_K_M) for battle-tested tool calling. Can swap to Functionary v3.2 or Gemma 4 later — just change the GGUF file and restart.

**Flags**:
- `--jinja`: Required. Enables the model's chat template for proper tool calling.
- `-ngl -1`: Offload all layers to Metal GPU.
- `-c 8192`: Context window. 8K is enough for multi-step agent reasoning with tool results.

### 3. Python Sidecar (FastAPI + ADK)

**Role**: The agent brain. Receives user messages, reasons about them, calls tools, returns responses.

**Replaces**: All of `src/utils/intentDetector.ts`, `src/utils/tools/`, `src/prompts/`, `src/utils/webLLMClient.ts`, and the current `sidecar/main.py`.

```
sidecar/
├── main.py                  # FastAPI app, /chat and /health endpoints
├── agent.py                 # ADK Agent setup (model, tools, instruction)
├── tools/
│   ├── __init__.py          # Tool registry
│   ├── search.py            # search_threads — keyword/tag/date search
│   ├── thread.py            # get_thread — full thread with comments
│   ├── trending.py          # get_trending — trending threads/tags
│   ├── user.py              # get_user_threads — threads by HN user
│   └── sandbox.py           # run_analysis — Extism WASM sandbox
├── sandbox/
│   ├── executor.py          # Extism plugin loader + execution
│   └── plugins/
│       └── js_runner.wasm   # Pre-compiled JS execution sandbox
├── session/
│   ├── store.py             # Session persistence (SQLite)
│   └── events.py            # Event types for the append-only log
├── prompts/
│   └── system.py            # System prompt for the HN agent
└── pyproject.toml
```

#### main.py — API Surface

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from agent import create_agent
from session.store import SessionStore

app = FastAPI()
sessions = SessionStore("sessions.db")
agent = create_agent()

@app.post("/chat")
async def chat(request: ChatRequest):
    session = sessions.get_or_create(request.session_id)

    # Run the agent — ADK handles the tool-calling loop
    response = agent(
        request.message,
        session_id=session.id
    )

    return ChatResponse(
        reply=str(response),
        session_id=session.id
    )

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    session = sessions.get_or_create(request.session_id)

    async def generate():
        async for event in agent.stream(request.message):
            yield f"data: {event.to_json()}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/health")
def health():
    return {"status": "ok"}
```

#### agent.py — ADK Agent Setup

```python
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from tools import search_threads, get_thread, get_trending, get_user_threads, run_analysis
from prompts.system import SYSTEM_PROMPT

def create_agent() -> Agent:
    model = LiteLlm(
        model="openai/hermes-2-pro",
        api_base="http://localhost:8080/v1",
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
            run_analysis,
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
4. Use run_analysis for data processing (aggregation, trend computation,
   statistics) when you have structured data to crunch.
5. Synthesize your findings into a clear, direct answer.

Always cite specific HN threads when referencing discussions.
Prefer concrete data over vague summaries.
If you're not sure, search more before answering.
"""
```

### 4. Tools

Each tool is a decorated Python function. ADK handles the schema generation and the tool-calling loop.

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
    limit: int = 20
) -> dict:
    """Search Hacker News threads by keyword, tags, and date range.

    Args:
        query: Search query (keywords or natural language).
        tags: Optional list of tags to filter by (e.g. ["ai", "rust"]).
        date_from: Start date in YYYY-MM-DD format.
        date_to: End date in YYYY-MM-DD format.
        sort_by: Sort order — "relevance", "date", or "comments".
        limit: Max number of results (default 20, max 50).

    Returns:
        Dict with 'threads' list and 'total_count'.
    """
    # Call Techne backend API
    response = httpx.get(
        f"{BACKEND_URL}/api/search",
        params={
            "q": query,
            "tags": ",".join(tags) if tags else None,
            "from": date_from,
            "to": date_to,
            "sort": sort_by,
            "limit": min(limit, 50),
        }
    )
    return response.json()
```

#### thread.py

```python
@tool
def get_thread(thread_id: str) -> dict:
    """Get the full details of a Hacker News thread including all comments.

    Args:
        thread_id: The HN story/thread ID.

    Returns:
        Dict with story metadata, tags, and comments tree.
    """
    response = httpx.get(f"{BACKEND_URL}/api/thread/{thread_id}")
    return response.json()
```

#### trending.py

```python
@tool
def get_trending(
    time_window: str = "24h",
    tag: str | None = None,
    limit: int = 10
) -> dict:
    """Get currently trending threads or tags on Hacker News.

    Args:
        time_window: Time window — "1h", "6h", "24h", "7d", "30d".
        tag: Optional tag to filter trending threads by.
        limit: Number of results (default 10).

    Returns:
        Dict with 'threads' list sorted by activity/karma density.
    """
    response = httpx.get(
        f"{BACKEND_URL}/api/trending",
        params={"window": time_window, "tag": tag, "limit": limit}
    )
    return response.json()
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
    response = httpx.get(
        f"{BACKEND_URL}/api/user/{username}/threads",
        params={"limit": limit}
    )
    return response.json()
```

#### sandbox.py

```python
@tool
def run_analysis(code: str, data: str) -> str:
    """Execute JavaScript code in a secure sandbox to analyze data.

    Use this when you need to process, aggregate, or compute statistics
    on structured data (JSON). The code runs in an isolated WebAssembly
    sandbox with no filesystem or network access.

    Args:
        code: JavaScript code to execute. Must read input via
              Host.inputString() and write output via Host.outputString().
        data: JSON string of data to pass to the code.

    Returns:
        The output string from the sandbox execution.
    """
    from sandbox.executor import SandboxExecutor
    executor = SandboxExecutor()
    return executor.run(code, data)
```

### 5. Extism Sandbox

**Role**: Secure execution environment for agent-generated code. The agent writes JavaScript to analyze HN data; the sandbox runs it in isolation.

```
sidecar/sandbox/
├── executor.py              # Python host that loads + runs the WASM plugin
└── plugins/
    └── js_runner.wasm       # Pre-built WASM module that evaluates JS
```

#### executor.py

```python
import extism

PLUGIN_PATH = os.path.join(os.path.dirname(__file__), "plugins", "js_runner.wasm")

class SandboxExecutor:
    def __init__(self, memory_limit_mb: int = 64, timeout_ms: int = 5000):
        self.memory_limit = memory_limit_mb
        self.timeout_ms = timeout_ms

    def run(self, code: str, data: str) -> str:
        """Run JS code in the WASM sandbox with the given data."""
        manifest = {
            "wasm": [{"path": PLUGIN_PATH}],
            "memory": {"max_pages": self.memory_limit * 16},  # 64KB per page
            "timeout_ms": self.timeout_ms,
            "allowed_hosts": [],  # No network access
        }

        plugin = extism.Plugin(manifest, wasi=True)

        try:
            # Pass the data + code as JSON input
            input_payload = json.dumps({"code": code, "data": data})
            result = plugin.call("execute", input_payload.encode())
            return result.decode()
        except Exception as e:
            return json.dumps({"error": str(e)})
        finally:
            plugin.close()
```

#### js_runner.wasm

This is a pre-compiled Extism plugin (built from JS/TS using the Extism PDK). The source:

```javascript
// js_runner_plugin.js — compiled to .wasm via Extism JS PDK

function execute() {
  const input = JSON.parse(Host.inputString());
  const { code, data } = input;

  // Make data available to the user's code
  const parsedData = JSON.parse(data);

  // Create a function from the agent's code and run it
  const fn = new Function("data", code);
  const result = fn(parsedData);

  Host.outputString(JSON.stringify(result));
}

module.exports = { execute };
```

**Build**: `extism-js js_runner_plugin.js -o js_runner.wasm`

**Security constraints**:
- No `fetch`, no `XMLHttpRequest` — no network
- No `fs`, no file access
- No `process`, no `require` — no Node APIs
- Memory capped (default 64MB)
- Execution timeout (default 5s)
- The only I/O is `Host.inputString()` → `Host.outputString()`

### 6. Frontend

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

const SIDECAR_URL = "http://localhost:8000";

export interface AgentMessage {
  role: "user" | "assistant";
  content: string;
}

export async function sendMessage(
  message: string,
  sessionId: string,
  onChunk?: (text: string) => void
): Promise<string> {
  const response = await fetch(`${SIDECAR_URL}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId }),
  });

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let fullResponse = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    // Parse SSE events
    for (const line of chunk.split("\n")) {
      if (line.startsWith("data: ")) {
        const event = JSON.parse(line.slice(6));
        if (event.type === "text") {
          fullResponse += event.content;
          onChunk?.(fullResponse);
        }
      }
    }
  }

  return fullResponse;
}
```

### 7. Session Management

**Role**: Durable event log so conversations survive sidecar restarts.

Following the Anthropic managed agents pattern: an append-only event log stored outside the LLM context window.

```python
# sidecar/session/store.py
import sqlite3
import json
from datetime import datetime

class SessionStore:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._init_tables()

    def _init_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                event_type TEXT,
                data TEXT,
                created_at TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

    def append_event(self, session_id: str, event_type: str, data: dict):
        self.conn.execute(
            "INSERT INTO events (session_id, event_type, data, created_at) VALUES (?, ?, ?, ?)",
            (session_id, event_type, json.dumps(data), datetime.utcnow().isoformat())
        )
        self.conn.commit()

    def get_events(self, session_id: str, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            "SELECT event_type, data, created_at FROM events WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit)
        ).fetchall()
        return [{"type": r[0], "data": json.loads(r[1]), "at": r[2]} for r in reversed(rows)]
```

Event types: `user_message`, `tool_call`, `tool_result`, `assistant_message`, `error`.

## Dependencies

### Python (sidecar/pyproject.toml)

```toml
[project]
name = "techne-agent"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn>=0.34",
    "google-adk>=1.0",
    "litellm>=1.50",
    "extism>=1.7",
    "httpx>=0.28",
    "pydantic>=2.10",
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
- `js_runner.wasm` — pre-compiled Extism plugin (~2MB)

## Implementation Order

### Phase 1: Agent core (no sandbox)

Get the agent loop working end-to-end.

1. **Set up llama-server** — download binary + GGUF, test tool calling manually with `curl`
2. **Rewrite sidecar** — replace current `sidecar/main.py` with ADK agent + LiteLLM + 2 basic tools (`search_threads`, `get_thread`)
3. **Verify agent loop** — test multi-step reasoning from the terminal (no UI yet)
4. **Wire up frontend** — replace `TauriWebLLMClient` with `agentClient.ts`, simplify `ChatInterface.tsx`
5. **Tauri process management** — spawn llama-server + sidecar from Rust, health checks

**Milestone**: User can chat with HN through the desktop app. Agent searches, fetches threads, and synthesizes answers across multiple tool calls.

### Phase 2: More tools + sessions

6. **Add remaining tools** — `get_trending`, `get_user_threads`
7. **Session persistence** — SQLite event log, conversation resume on restart
8. **Streaming** — SSE streaming from sidecar to frontend for real-time responses

**Milestone**: Persistent conversations, richer queries, streaming UX.

### Phase 3: Sandbox

9. **Build js_runner.wasm** — compile the Extism JS plugin
10. **Implement SandboxExecutor** — Extism host in Python
11. **Add run_analysis tool** — wire sandbox into the agent's tool set
12. **Test with data analysis queries** — "What are the most discussed topics this month?", "Show me sentiment trends for Rust"

**Milestone**: Agent can write and execute code to analyze HN data.

### Phase 4: Polish

13. **Error handling** — graceful recovery from llama-server crashes, sidecar timeouts
14. **Model hot-swap** — settings UI to pick a different GGUF without restarting the app
15. **Remove dead code** — delete all WebLLM, intent detection, and in-browser tool orchestration code
16. **Packaging** — bundle llama-server binary and GGUF model with the Tauri app

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
      "author": "tptacek",
      "text": "The privacy argument is strong but the latency story is more nuanced...",
      "score": 89,
      "posted_at": "2026-03-15T10:15:00Z",
      "depth": 0,
      "children": [
        {
          "id": 39812450,
          "author": "dang",
          "text": "Could you elaborate on the latency comparison?",
          "score": 34,
          "posted_at": "2026-03-15T10:22:00Z",
          "depth": 1,
          "children": []
        }
      ]
    }
  ]
}
```

**Design decisions:**
- Comments are a nested tree (not flat) so the agent understands reply structure.
- `depth` is denormalized for easy flattening if the agent passes comments to `run_analysis`.
- `max_comments` prevents blowing the context window. 50 top comments is usually enough to understand the discussion. The agent can ask for more if needed.
- Comment `score` lets the agent focus on high-signal comments.

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
- `top_categories` / `top_themes` — pre-aggregated so the agent can characterize a user's interests without running `run_analysis`. Saves a tool call.
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

3. **Agent-friendly response sizes.** Defaults are tuned to fit in an 8K context window with room for reasoning. `search` returns 20 threads with summaries (~3K tokens). `get_thread` returns 50 comments (~4K tokens). The agent can request more if it has headroom.

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
3. **Extism JS PDK maturity** — The JS PDK compiles to QuickJS inside WASM. Verify it handles the JSON processing patterns we need.
4. **App size** — The GGUF model is ~4.5GB. Should it be bundled or downloaded on first launch?
5. **Backend search strategy** — Does the backend already have full-text or vector search over threads? If keyword-only, is that sufficient for v1 given the agent can interpret results?
