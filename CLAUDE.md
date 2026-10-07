# Techne Navigator

A macOS desktop app for exploring Hacker News with a local AI model. Tauri v2:
a React/TypeScript frontend in a WKWebView, a Rust backend, and both models
running in the same process. No server, no sidecar, no network inference.

The Chrome extension this grew out of lives in **its own repo**. Nothing here
builds or ships it.

## Running it

```bash
npm run tauri:dev      # the whole app — nothing else to start
npm run tauri:build    # .app + .dmg
npm run vite:build     # frontend only, into dist-tauri/
```

Setup and the model downloads are in README.md. The short version: the two GGUFs
live in `models/`, and `src-tauri/tauri.conf.json` bundles them into the app.

## Shape

```
src/                     React 19 + TypeScript, built by Vite
  popup/components/      the UI — ChatInterface is the root
  tauri-compat/          the seams: invoke() clients, WKWebView fixes
  utils/                 contextDb (Dexie), activity, logger, config
src-tauri/src/           Rust
  agent.rs               the Rig agent, its tools, and Gemma
  summary.rs             reading and summarising one HN thread
  corpus.rs              30 days of threads, cached and refreshed
  search.rs              nomic embeddings, cosine, rerank
  memory.rs              conversation history in SQLite
  bin/                   check runners (see Testing)
models/                  the two GGUFs, gitignored
```

**The frontend owns no intelligence.** It calls `invoke()` and renders what
comes back. Routing, tool use and conversation history all live in Rust.

## Models

| | |
|---|---|
| chat | `google_gemma-4-26B-A4B-it-Q3_K_M.gguf` — 12GB, Metal |
| embeddings | `nomic-embed-text-v1.5.Q8_0.gguf` — 146MB, CPU (`n_gpu_layers=0`) |

nomic is not interchangeable: the corpus vectors came from it, and a different
embedder returns noise rather than weak matches. Queries need the
`search_query: ` prefix, and the returned vector must be normalised by hand —
llama.cpp hands back a raw one.

Both are loaded once at startup into Tauri state, and **released on
`RunEvent::Exit`** — see `Loaded<T>` in agent.rs. Without that, llama.cpp frees
the Metal device in a static destructor while a model is still alive and the
process aborts on `GGML_ASSERT`.

## The context window

`N_CTX` is **8192** and cannot currently go higher: `rig-llama-cpp` fails above
it on this model (`Decode Error -3` at 16,384), while `llama-server` runs the
same GGUF at 65,536. One genuine cause was found — `llama-cpp-2` defaults
`swa_full = true` where llama.cpp uses `false` — but patching it only moved the
failure. Not solved.

Everything shares that one window: the agent's turns, the reranker's candidates,
and any thread being read. Useful measurements:

- our thread formatting runs at **~3.5 characters per token**
- the agent's preamble and both tool schemas cost only **316 tokens**
- `max_tokens(2048)` is the real fixed cost — a quarter of the window
- so ~20,000 characters are free in a fresh chat, and the 14,000 budget in
  `summary.rs` fits both the modal and `get_thread`
- what does *not* fit is reading two threads in one conversation: a thread
  already read stays in the history and is re-sent every turn

Prefill (~230 tok/s) dominates, not generation (~50 tok/s). `n_ctx` costs memory,
not time — only tokens actually processed cost time.

## Gemma specifics

- It bills chain-of-thought against `max_tokens` and will spend all of it
  thinking, returning empty content. Disabled with `additional_params({"thinking":
  false})` — in-process the key is `thinking`, **not** the HTTP-era
  `chat_template_kwargs.enable_thinking`, which is silently ignored.
- Rig runs **one** model call by default. Without `.default_max_turns(10)` any
  tool use dies with `MaxTurnsError` before the model answers.
- Never hand it a URL: it retypes it into its prose and the UI renders a second
  copy. Links are resolved from the corpus by `thread_id` instead.
- It will dress a real figure up as a precise invented one. The preamble forbids
  converting, scaling or combining numbers.

## Data

Two stores, for two different lifetimes.

**Dexie / IndexedDB** (`src/utils/contextDb.ts`) — what the user accumulates:
visited threads, searches, settings, conversations. Survives restarts.

**SQLite** (`src-tauri/src/memory.rs`) — the agent's own conversation history,
which Rig needs on its side of the bridge.

Recording activity goes through **`src/utils/activity.ts`**: two write functions
and an `onActivity` subscription. This replaced a `chrome.runtime` message bus
that existed only because a Chrome popup could not write to IndexedDB directly.
A view that lists activity subscribes; a thing that records it calls a function.

## Pinned threads

A thread dragged into the chat rides **on the message**, not in agent state.
State feeds the system prompt, which has no position in time, so a thread just
read with `get_thread` always won "this thread". Measured; no instruction wording
beat it.

The attachment carries the thread's **id**, not a summary. Handing over a summary
decides in advance what can be asked — 130 words cannot answer "did anyone
mention performance?" — so the model is given the id and reads it itself.

## WKWebView

`src/tauri-compat/webview.ts`, installed before React renders. Two things
browsers do that WKWebView does not:

- **Drag and drop.** `dragstart` fires and macOS swallows everything after it,
  and `dataTransfer` is empty on a synthetic drop. Drags are tracked by hand.
  Note that a card is `draggable`, so a mousedown anywhere inside it starts a
  drag that swallows clicks — a button inside needs `draggable={false}` *and*
  `onMouseDown={e => e.stopPropagation()}`.
- **External links.** `tauri-plugin-shell` opens them from a listener on
  `<body>`. Never `stopPropagation` on a container holding links — the click
  never reaches the plugin and the link silently does nothing.

Any HN thread link must carry `data-visit-recorded="true"` and record itself, or
the global interceptor labels the visit with the link's own text and Memory fills
with rows called "Join the thread - 10 comments". `JoinThreadLink` in
ThreadCard.tsx is the one shared copy.

## Testing

There are no unit tests (#28). What exists are check runners that drive the real
code against live data:

```bash
cd src-tauri
cargo run --bin check          # the agent: tools, history, pinned threads
cargo run --bin search_check   # embeddings and reranking
cargo run --bin corpus_check   # the 30-day corpus
cargo run --bin summary_check  # thread summaries
```

**Close the app first.** Each loads its own 12GB Gemma and 24GB cannot hold two.
The symptom is not an out-of-memory error — it is empty replies and `Decode Error
-3`, which reads exactly like a prompt regression. This has cost hours twice.

`npx tsc --noEmit` is clean and can be trusted as a gate. There is no CI.

## Building for release

`npm run tauri:build` produces the `.app` and `.dmg` under
`src-tauri/target/release/bundle/`. Apple Silicon only — a 12GB model is not
worth running on an Intel Mac.

**`bundle.macOS.minimumSystemVersion` must stay at 11.0 or higher.** Without it
Tauri sets `MACOSX_DEPLOYMENT_TARGET` to 10.13 and llama.cpp will not compile:
`'path' is unavailable: introduced in macOS 10.15`. 11.0 is the floor for Apple
Silicon anyway.

If a build has already failed that way, fixing the config is not enough — CMake
caches the old deployment target and goes on using it:

```bash
cd src-tauri && cargo clean --release -p llama-cpp-sys-2
```

A release build holds **three copies of the models at once** — the source in
`models/`, a copy inside the `.app`, and another inside the `.dmg` — roughly
54GB with the Rust artefacts. That is why releases are built locally rather than
on CI, where runners have far less disk than that.

## Conventions

- **Planning lives in GitHub issues**, not markdown in the repo.
- **Delete replaced code.** Git history is the safety net.
- Use the logger in `src/utils/logger.ts`, not `console.*`: `logger.debug()`,
  `.error()`, `.chat()`, `.search()`, `.model()`, `.database()`, `.api()`.
- Rust prints to stdout with a `[tag]` prefix — `[agent]`, `[summary]`,
  `[corpus]`, `[search]`, `[thread]` — which is what shows up in `tauri:dev`.
- Spawn background work with `tauri::async_runtime::spawn`, never
  `tokio::spawn`, which panics with "there is no reactor running" from Tauri
  setup.
- `rig-core` is pinned to **0.40** and `llama-cpp-2` to **0.1.152**.
  `rig-llama-cpp` only targets rig-core ^0.40, and llama-cpp-2 ships breaking
  changes in patch releases. If a `cargo update` breaks the build, that is why —
  re-pin rather than patching around it.
