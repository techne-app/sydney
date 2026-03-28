# Techne — Desktop App Setup

> macOS only. Requires ~3GB disk space for the AI model.

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

**Step 7 — Download the AI model** *(~2.5GB — go grab a coffee)*
```bash
mkdir -p sidecar/models
curl -L "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf" \
  -o sidecar/models/Qwen3-4B-Q4_K_M.gguf
```

## Running

Open two terminal windows:

**Terminal 1 — Start the AI sidecar**
```bash
cd sidecar
uv run main.py
```
Wait until you see `Model loaded.`

**Terminal 2 — Start the app**
```bash
npm run tauri:dev
```
