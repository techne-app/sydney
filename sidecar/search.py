"""Semantic search over the 30-day thread corpus.

Embed the query with nomic, cosine against ~30k pre-computed vectors, then have
Gemma rerank the shortlist down to a handful. Lives here rather than in main.py
so both the /search endpoint and the agent's search tool run the same code.
"""
import os
from typing import Any, Dict, List

import numpy as np
from llama_cpp import Llama

import corpus
from llm import chat_completion

# nomic stays in-process. Only the chat model needs llama-server (--jinja, for
# tool calling); this is 139MB on CPU embedding one short query per search, so a
# second server would add a process and a failure mode for no gain.
#
# It MUST be the model that produced the stored vectors — nomic-embed-text-v1.5,
# 768-dim (techne-pipeline/scripts/start_embedding_server.sh). A query embedded
# by any other model lands in a different vector space, and cosine against it
# returns noise rather than weak matches.
EMBED_MODEL_NAME = "nomic-embed-text-v1.5.Q8_0.gguf"

# How many candidates the embedding stage hands to the reranker. Mirrors the
# pipeline's feed generation, which cosines to 100 then has the LLM pick ~6.
RERANK_CANDIDATES = 100

_embedder: Llama | None = None


def _resolve_model(name: str) -> str:
    """Locate a GGUF in the bundled .app or in the dev checkout."""
    import sys
    base = os.path.dirname(os.path.abspath(sys.argv[0]))
    bundled = os.path.join(base, "..", "Resources", "models", name)
    local = os.path.join(base, "models", name)
    return bundled if os.path.exists(bundled) else local


def load_embedder() -> None:
    """Load nomic once at startup."""
    global _embedder
    path = _resolve_model(EMBED_MODEL_NAME)
    print(f"Loading embedding model from {path}...")
    _embedder = Llama(
        model_path=path,
        embedding=True,
        n_gpu_layers=0,   # CPU: keeps the Metal context free for llama-server's Gemma
        verbose=False,
    )
    print("Embedding model loaded.")


def embed_query(query: str) -> np.ndarray:
    """Embed and normalize a search query.

    The "search_query: " prefix is required, not decorative: every theme was
    stored with "search_document: ", and nomic places queries and documents
    differently. The wrong prefix degrades relevance silently — no error.
    """
    if _embedder is None:
        raise RuntimeError("embedder not loaded — call load_embedder() first")
    embedding = _embedder.create_embedding(f"search_query: {query}")
    vector = np.asarray(embedding["data"][0]["embedding"], dtype=np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def _build_rerank_prompt(query: str, candidates, limit: int) -> str:
    listing = "\n".join(
        f"{i + 1}. {meta.get('story_title', '')} — {meta.get('theme', '')}"
        for i, (meta, _) in enumerate(candidates)
    )
    return (
        f"A user searched Hacker News for: \"{query}\"\n\n"
        f"Below are {len(candidates)} candidate discussions, already filtered by "
        "semantic similarity. Pick the "
        f"{limit} most genuinely relevant to what the user is asking for.\n\n"
        f"{listing}\n\n"
        f"Reply with ONLY the {limit} numbers, best first, separated by commas. "
        "No explanation, no other text."
    )


def _parse_rerank_picks(text: str, count: int, limit: int) -> List[int]:
    """Pull the chosen indices out of the model's reply.

    Deliberately just scrapes integers rather than asking for JSON or a tool
    call — a bare list of numbers is the one format that is hard to get wrong.
    Out-of-range or duplicate numbers are dropped; if nothing usable comes back
    the caller falls back to plain cosine order.
    """
    import re
    picks: List[int] = []
    for token in re.findall(r"\d+", text):
        index = int(token) - 1
        if 0 <= index < count and index not in picks:
            picks.append(index)
        if len(picks) == limit:
            break
    return picks


def run_search(query: str, limit: int = 3) -> Dict[str, Any]:
    """Full search: embed, cosine, dedupe by story, rerank. Never raises."""
    if not corpus.is_ready():
        return {"results": [], "error": "Thread data is still loading, try again shortly."}

    query = (query or "").strip()
    if not query:
        return {"results": [], "error": "Empty query."}

    vector = embed_query(query)

    # Pull well past RERANK_CANDIDATES, then keep only the best-scoring thread
    # per story. A row is one comment thread, not one story, so a popular post
    # contributes dozens of near-identical rows — without this, a query like
    # "self hosting email" returns three threads from the same discussion.
    # (Per-story on purpose; the corpus itself stays keyed by thread_id, which
    # is what makes delta merges correct.)
    ranked = corpus.search(vector, k=RERANK_CANDIDATES * 4)
    seen_stories = set()
    candidates = []
    for meta, score in ranked:
        story_id = meta.get("story_id")
        if story_id in seen_stories:
            continue
        seen_stories.add(story_id)
        candidates.append((meta, score))
        if len(candidates) == RERANK_CANDIDATES:
            break

    if not candidates:
        return {"results": [], "error": "No matching discussions found."}

    # Rerank with Gemma. If it returns nothing usable, fall through to cosine
    # order rather than failing the search.
    picks: List[int] = []
    try:
        completion = chat_completion(
            messages=[{"role": "user", "content": _build_rerank_prompt(query, candidates, limit)}],
            temperature=0.1,
            max_tokens=64,
        )
        picks = _parse_rerank_picks(completion.get("content") or "", len(candidates), limit)
    except Exception as error:
        print(f"[search] rerank failed, using cosine order: {error}")

    if not picks:
        picks = list(range(min(limit, len(candidates))))

    return {"results": [{**candidates[i][0], "score": candidates[i][1]} for i in picks]}
