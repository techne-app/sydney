# Techne — Desktop App Setup

> macOS only (Apple Silicon). Requires ~3GB disk space for the AI model.

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

## Dev Mode

Run the Python sidecar directly from source — no compilation needed, fast iteration.

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

## Production Build (DMG)

Creates a standalone `.dmg` installer that bundles the app, sidecar binary, and AI model.

**Step 8 — Compile the Python sidecar with Nuitka** *(one-time, unless you change `main.py`)*
```bash
cd sidecar
uv run python -m nuitka --onefile --output-filename=sidecar-aarch64-apple-darwin main.py
mkdir -p ../src-tauri/binaries
mv sidecar-aarch64-apple-darwin ../src-tauri/binaries/
cd ..
```

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
