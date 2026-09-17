from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

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
# /thread/{id} — what the agent's get_thread tool reads.
#
# The agent is moving into Tauri core (#46), so its tools reach the corpus over
# HTTP the way the frontend already does. Until #48 moves the corpus to Rust
# too, this is the seam between them.
#
# `link` is included for the UI, NOT for the model. The caller must not pass it
# into the tool result: handed a URL, the model writes it into its prose and the
# UI then renders the same link a second time. The UI resolves links from here
# so they are exact rather than retyped.
# ---------------------------------------------------------------------------

@app.get("/thread/{thread_id}")
def thread_endpoint(thread_id: int):
    row = corpus.get(thread_id)
    if row is None:
        return {
            "error": (
                "No thread with that id is in the last 30 days of data. "
                "Search for the topic instead."
            )
        }
    view = corpus.public_view(row, detail=True)
    view["link"] = row.get("anchor")
    return view


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
