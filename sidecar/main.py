from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional
import os

import agent
import corpus
import search
from llm import llama_server_ready

app = FastAPI()

# Allow requests from Tauri/localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# nomic loads here; Gemma is served by llama-server (see llm.py).
search.load_embedder()

# Open the session database and build the agent's runner.
agent.start()

# Start syncing the 30-day thread corpus in the background. Non-blocking: this
# runs while Gemma loads, so the ~44s download costs the user nothing.
corpus.start()


class SearchRequest(BaseModel):
    query: str
    limit: int = 3


@app.post("/search")
def search_endpoint(request: SearchRequest):
    return search.run_search(request.query, request.limit)


# ---------------------------------------------------------------------------
# /sessions/{id}/messages — the agent. One endpoint serves every conversation;
# the session id in the path selects which history to load. The agent runs its
# own tool loop, so it can call a tool, read the result, and call another before
# answering.
# ---------------------------------------------------------------------------

class AgentMessage(BaseModel):
    message: str
    # What the user currently has open, or null when nothing is. Goes into
    # session state, not the prompt — see agent._instruction.
    pinned_thread: Optional[Dict[str, Any]] = None


@app.post("/sessions/{session_id}/messages")
async def send_message(session_id: str, body: AgentMessage):
    if not llama_server_ready():
        return {"reply": "", "error": "The model server is still starting up."}
    try:
        return await agent.send(session_id, body.message, body.pinned_thread)
    except Exception as error:
        print(f"[agent] failed: {error}")
        return {"reply": "", "error": str(error)}


@app.get("/health")
def health():
    # llama_server is reported separately because the sidecar starts fine
    # without it and only fails when a request needs Gemma. Surfacing it here
    # makes "the model server isn't up" diagnosable instead of a 500 later.
    return {
        "status": "ok",
        "llama_server": llama_server_ready(),
        "corpus": corpus.status(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
