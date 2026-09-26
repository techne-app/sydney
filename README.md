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

Gemma answers and reranks; nomic embeds search queries. nomic is not
interchangeable — it must be the model that produced the vectors stored in the
backend, or search returns noise rather than weak matches.

```bash
mkdir -p sidecar/models

# Chat + rerank (~12GB) — from bartowski/google_gemma-4-26B-A4B-it-GGUF
curl -L "https://huggingface.co/bartowski/google_gemma-4-26B-A4B-it-GGUF/resolve/main/google_gemma-4-26B-A4B-it-Q3_K_M.gguf" \
  -o sidecar/models/google_gemma-4-26B-A4B-it-Q3_K_M.gguf

# Search query embeddings (~139MB)
curl -L "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf" \
  -o sidecar/models/nomic-embed-text-v1.5.Q8_0.gguf
```

Both filenames appear in `src-tauri/tauri.conf.json` (`bundle.resources`) and
in the Rust — `CHAT_MODEL` in `agent.rs`, `EMBED_MODEL` in `search.rs`.

## Dev Mode

```bash
npm run tauri:dev
```

That is the whole of it. Nothing else to start: the agent, the corpus, search
and both models all run inside the app.

The window appears immediately: llama.cpp memory-maps the GGUFs rather than
reading them, so the 12GB pages in lazily as the model runs.

## Production Build (DMG)

Creates a standalone `.dmg` bundling the app and both models.

**Step 8 — Build**

```bash
npm run tauri:build
```

**A Finder window will open showing the app beside an Applications shortcut.
Leave it alone until the build says `Finished`** — it is part of the build, not
an invitation to install. Dragging the app out starts a 13GB copy that holds the
disk image open, and the build then fails with `error running bundle_dmg.sh`.

**Output:**
- `src-tauri/target/release/bundle/macos/Techne Navigator.app` — run this directly to test
- `src-tauri/target/release/bundle/dmg/Techne Navigator_0.1.0_aarch64.dmg` — share this to distribute

Run the `.app` directly, or install from the `.dmg` and eject it afterwards —
a stale mount interferes with the next build.

### Troubleshooting

**`error running bundle_dmg.sh`** — something touched the mounted disk image
during the build. Not a Finder permission problem, despite appearances. Clean up
and retry without touching anything:

```bash
hdiutil detach /Volumes/dmg.* -force 2>/dev/null
rm -f src-tauri/target/release/bundle/macos/rw.*.dmg src-tauri/target/release/bundle/dmg/rw.*.dmg
npm run tauri:build
```

Deleting those scratch files matters: `bundle/macos/` is what gets imaged, so a
leftover `rw.*.dmg` ends up *inside* the dmg you ship.
