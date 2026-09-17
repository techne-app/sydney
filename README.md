# Techne — Desktop App Setup

> macOS only (Apple Silicon). Requires ~13GB disk space for the AI models, and
> 24GB RAM to run them comfortably.

## Prerequisites

**Step 1 — Xcode command line tools** *(needed to compile Rust)*
```bash
xcode-select --install
```

**Step 2 — Homebrew** *(skip if already installed)*
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Step 3 — Node.js and UV**
```bash
brew install node
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

**Step 4 — Rust**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.zshrc
```

## Installation

**Step 5 — Clone the repo**
```bash
git clone https://github.com/techne-app/sydney.git
cd sydney
```

**Step 6 — Install Node dependencies**
```bash
npm install
```

**Step 7 — Download the AI models** *(~12GB — go grab a coffee)*

Two models: Gemma 4 answers and reranks, nomic embeds search queries. The
embedding model is not optional — it has to be the one that produced the
vectors stored in the backend, or search returns noise rather than weak matches.

```bash
mkdir -p sidecar/models

# Chat + rerank (~12GB) — from bartowski/google_gemma-4-26B-A4B-it-GGUF
curl -L "https://huggingface.co/bartowski/google_gemma-4-26B-A4B-it-GGUF/resolve/main/google_gemma-4-26B-A4B-it-Q3_K_M.gguf" \
  -o sidecar/models/google_gemma-4-26B-A4B-it-Q3_K_M.gguf

# Search query embeddings (~139MB)
curl -L "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf" \
  -o sidecar/models/nomic-embed-text-v1.5.Q8_0.gguf
```

If you have `techne-pipeline` checked out, copying Gemma from its `models/`
folder is faster than downloading. That copy is slightly older than the one above
— different chat template, same behaviour; both pass the full check suite.

The Gemma filename appears in **`src-tauri/src/lib.rs`** (the `-m` path
production spawns llama-server with) and in `src-tauri/tauri.conf.json`
(`bundle.resources`); the Dev Mode command below has it too. The nomic
filename is `EMBED_MODEL_NAME` in `sidecar/search.py`. Swapping either model
means updating every one of those — the bundle names files explicitly, so a
leftover GGUF in `sidecar/models/` can't quietly add gigabytes to the DMG (which
is why the unused `Qwen3-4B-Q4_K_M.gguf` sitting there costs nothing).

## Dev Mode

Run the model server and Python sidecar from source — no compilation needed,
fast iteration. In production Tauri starts both for you; in dev you start them
by hand.

Open three terminal windows.

First check both ports are free — **quitting the packaged app does not always
stop its llama-server and sidecar**, and leftovers make the commands below fail
to bind:

```bash
lsof -ti :8081 :8000    # kill anything listed, then continue
```

**Terminal 1 — Start the model server**
```bash
./sidecar/llama/llama-server \
  -m sidecar/models/google_gemma-4-26B-A4B-it-Q3_K_M.gguf \
  --jinja \
  -ngl 99 -c 8192 \
  --host 127.0.0.1 --port 8081
```
`--jinja` is not optional. It applies Gemma's own chat template so tool calls
come back as a structured `tool_calls` field. Without it they arrive as raw
`<|tool_call>` text, the sidecar sees no tool call, and search and summarize
silently stop working while chat still looks fine.

Takes a minute or two to load 12GB. Ready when `curl 127.0.0.1:8081/health`
returns OK.

**Terminal 2 — Start the Python sidecar**
```bash
cd sidecar
uv run main.py
```
Starts in seconds — it only loads nomic (139MB). Gemma lives in llama-server.
`curl localhost:8000/health` reports `llama_server: true` once both are up.

**Terminal 3 — Start the app**
```bash
npm run tauri:dev
```

## Production Build (DMG)

Creates a standalone `.dmg` installer that bundles the app, sidecar binary, and AI model.

**Step 8 — Compile the Python sidecar with Nuitka** *(rerun after **any** change under `sidecar/`)*

```bash
cd sidecar
uv run python -m nuitka --onefile \
  --output-filename=sidecar-aarch64-apple-darwin \
  --include-package-data=certifi \
  --include-package-data=llama_cpp \
  --assume-yes-for-downloads \
  main.py
mkdir -p ../src-tauri/binaries
mv sidecar-aarch64-apple-darwin ../src-tauri/binaries/
cd ..
```

**Not yet re-verified since the agent moved to Rust.** The sidecar used to
carry `google-adk` and `litellm`, which needed a dozen extra `--include-*` flags
and made this step take ~70 minutes — one generated file, `google.genai.types`,
was 65 of them. Those dependencies are gone, so the flag list above is back to
the two that predate them and the build should be far quicker. Confirm by
running the compiled binary before trusting a bundle.

Nuitka compiles ahead of time and finds modules by reading `import` statements,
so anything resolved from a runtime *string* is invisible to it and missing
**only in the compiled binary** — never under `uv run`. If the binary dies with
`ModuleNotFoundError` while dev works fine, that is the cause, and the fix is
another `--include-package`.

**Then run the compiled binary before bundling** — a stale or broken binary is
otherwise invisible until the app is installed. It finds models relative to its
own directory, so mirror the production layout:

```bash
mkdir -p src-tauri/Resources && ln -sfn ../../sidecar/models src-tauri/Resources/models
./src-tauri/binaries/sidecar-aarch64-apple-darwin
rm -rf src-tauri/Resources   # Tauri builds its own
```

Healthy startup prints `Embedding model loaded.`, `[agent] sessions at ...`, and
a corpus line. Ctrl+C when you've seen them.

**Step 9 — Build**

```bash
npm run tauri:build
```

**While this runs, a disk image is created and mounted, and a Finder window opens
showing the app beside an Applications shortcut. Leave it alone until the build
says `Finished`.** That window is part of the build, not an invitation to
install. Dragging the app out of it starts a 13GB Finder copy that holds the
volume, so the script cannot eject it and the build fails with
`error running bundle_dmg.sh`. This is by far the most common way for this build
to fail, and it looks like a permissions problem when it isn't — Automation
permission for Finder is not required.

**Output:**
- `src-tauri/target/release/bundle/macos/Techne Navigator.app` — run this directly to test
- `src-tauri/target/release/bundle/dmg/Techne Navigator_0.1.0_aarch64.dmg` — share this to distribute

**To run the app**, either double-click the `.app` in `macos/`, or open the
`.dmg` and drag `Techne Navigator` into `Applications`. Eject the disk image
afterwards, so a stale mount can't interfere with the next build.

First launch takes a minute or two: the app starts llama-server and the sidecar
itself (unlike dev, where you start them by hand), and Gemma has 12GB to load.

### Troubleshooting

**`error running bundle_dmg.sh`** — something touched the mounted disk image
during the build (see Step 9). Not a Finder permission problem, despite
appearances. Clean up and retry without touching anything:

```bash
hdiutil detach /Volumes/dmg.* -force 2>/dev/null
rm -f src-tauri/target/release/bundle/macos/rw.*.dmg src-tauri/target/release/bundle/dmg/rw.*.dmg
npm run tauri:build
```

Deleting those scratch files matters: `bundle/macos/` is what gets imaged, so a
leftover 14GB `rw.*.dmg` ends up *inside* the dmg you ship.

**Compile stuck at `6380/6381`** — it isn't, see Step 8. Check it's alive:
`ps -o %cpu= -p $(pgrep -f "clang -cc1" | head -1)`.

**`couldn't bind ... port 8081` (or 8000)** — the packaged app's llama-server and
sidecar outlive it. `lsof -ti :8081 :8000`, kill what's listed.
