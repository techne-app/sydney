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

**Step 3 — Node.js and cmake**

cmake is required: llama.cpp is compiled into the app for the embedding model.

```bash
brew install node cmake
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

Open two terminal windows. (It used to be three — the Python sidecar is gone;
the corpus, search and the agent all run inside the app now.)

First check the port is free — **quitting the packaged app does not always stop
its llama-server**, and a leftover makes the command below fail to bind:

```bash
lsof -ti :8081    # kill anything listed, then continue
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
`<|tool_call>` text, the agent sees no tool call, and search silently stops
working while chat still looks fine.

Takes a minute or two to load 12GB. Ready when `curl 127.0.0.1:8081/health`
returns OK.

**Terminal 2 — Start the app**
```bash
npm run tauri:dev
```

## Production Build (DMG)

Creates a standalone `.dmg` installer that bundles the app, sidecar binary, and AI model.

**Step 8 — Build**

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
