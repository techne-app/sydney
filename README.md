STEP 1 — Xcode tools (needed for compiling Rust)

xcode-select --install

STEP 2 — Install Homebrew (if not already)

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

STEP 3 — Install Node.js

brew install node

STEP 4 — Install Python

brew install python

STEP 5 — Install Rust

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.zshrc

STEP 6 — Clone the repo

git clone https://github.com/techne-app/sydney.git
cd sydney

STEP 7 — Install Node dependencies

npm install

STEP 8 — Install Python dependencies (Metal GPU support for Mac)

CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python fastapi uvicorn

STEP 9 — Download the model (~2.5GB, go grab a coffee)

mkdir -p sidecar/models
curl -L "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf" -o sidecar/models/Qwen3-4B-Q4_K_M.gguf

STEP 10 — Start the sidecar (Terminal 1)

cd sidecar
uvicorn main:app --port 8000
Wait until you see Model loaded.

STEP 11 — Start the app (Terminal 2)


npm run tauri:dev
