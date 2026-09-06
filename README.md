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

The same Gemma file the pipeline runs (`techne-pipeline/models/`), so copying it
from there is faster if you have it. Note the Hugging Face copy has since been
re-uploaded and differs slightly (13,019,981,440 vs 13,019,979,680 bytes). Gemma's
tool-call syntax lives in the chat template *inside* the GGUF and `sidecar/main.py`
parses it, so after a fresh download re-run the routing checks before trusting it:
a changed template breaks search and summarize silently, without erroring.

Both filenames appear in `sidecar/main.py` (`MODEL_NAME` / `EMBED_MODEL_NAME`)
and in `src-tauri/tauri.conf.json` (`bundle.resources`). Swapping a model means
updating both — the bundle names files explicitly so a leftover GGUF in
`sidecar/models/` can't quietly add gigabytes to the DMG.

## Dev Mode

Run the model server and Python sidecar from source — no compilation needed,
fast iteration. In production Tauri starts both for you; in dev you start them
by hand.

Open three terminal windows:

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

The two `--include-package-data` flags matter, and both fail *only* in the
compiled binary — never in `uv run`, so dev proves nothing here:

- **certifi** ships `cacert.pem`, the trusted-CA list Python needs to verify
  HTTPS. Without it the corpus fetch from `techne.app` fails and search quietly
  returns nothing.
- **llama_cpp** ships the compiled library and Metal shaders. Without them the
  models don't load at all.

Nothing automates this step, so a stale binary is easy to ship: the `.app` runs
whatever `main.py` was compiled last, not what's in the working tree. **Always
run the compiled binary directly** (`./src-tauri/binaries/sidecar-aarch64-apple-darwin`)
and check search works before building the dmg.

**Step 9 — Grant Automation permission** *(one-time setup)*

The build creates a DMG installer with a drag-and-drop window. macOS requires your terminal to have permission to control Finder for this.

System Settings → Privacy & Security → Automation → find your terminal app → enable the **Finder** checkbox.

**Step 10 — Build**
```bash
npm run tauri:build
```

**Output:**
- `src-tauri/target/release/bundle/macos/Techne Navigator.app` — run this directly to test
- `src-tauri/target/release/bundle/dmg/Techne Navigator_0.1.0_aarch64.dmg` — share this to distribute

**To run the app**, either:
1. Double-click `Techne Navigator.app` in the `macos/` folder — quickest way to test
2. Or open the `.dmg`, drag `Techne Navigator` into `Applications`, launch from there — standard macOS install

### Troubleshooting

**Build fails with `error running bundle_dmg.sh`**
→ You didn't do Step 9. Grant Automation permission and retry.

**Orphaned `rw.*.dmg` files in `src-tauri/target/release/bundle/macos/`**
→ Leftover from failed builds. Clean up and retry:
```bash
rm -f src-tauri/target/release/bundle/macos/rw.*.dmg
npm run tauri:build
```
