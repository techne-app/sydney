from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from llama_cpp import Llama
import os
import re

app = FastAPI()

# Allow requests from Tauri/localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "Qwen3-4B-Q4_K_M.gguf")

print(f"Loading model from {MODEL_PATH}...")
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,   # -1 = offload all layers to Metal GPU
    n_ctx=4096,        # context window
    verbose=False,
)
print("Model loaded.")


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    reply: str


@app.post("/chat")
def chat(request: ChatRequest):
    # Convert messages to the format llama-cpp expects
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Qwen3: disable thinking mode via /no_think in the system prompt
    # This prevents <think>...</think> blocks and keeps responses clean/fast
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = "/no_think\n" + messages[0]["content"]
    else:
        messages = [{"role": "system", "content": "/no_think"}] + messages

    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )

    reply = response["choices"][0]["message"]["content"]
    reply = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL).strip()

    return ChatResponse(reply=reply)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
