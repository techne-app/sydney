"""The 30-day thread corpus the sidecar searches over.

Owns fetching from techne.app, caching to disk, aging out old threads, and the
cosine similarity itself. Kept out of main.py so the endpoint stays readable.

Why the data lives here and not in the webview: the threads arrive already
embedded (768-dim nomic vectors, computed by the pipeline), so a search only has
to embed the query — one string — no matter how big the pool is. That is what
lets us go from 30 candidates to ~30,000 while doing *less* work per query than
the old in-browser MiniLM path.
"""
import base64
import json
import os
import threading
import time as _time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

CORPUS_URL = os.environ.get(
    "TECHNE_CORPUS_URL", "https://www.techne.app/api/threads-30d/"
)

WINDOW_DAYS = 30
EMBEDDING_DIMS = 768
REFRESH_SECONDS = 15 * 60

# A cold Vercel function took 107s once before settling to 2-3s, so the timeout
# has to be generous or the first fetch after an idle period fails spuriously.
FETCH_TIMEOUT = 300.0

# Written by the app, so it cannot live in Contents/Resources — writing there
# breaks the code signature. Rust passes the sidecar no env or args, so resolve
# this here; it then works identically in dev and in the bundled .app.
CACHE_DIR = os.path.expanduser(
    "~/Library/Application Support/com.technesystems.techne/corpus"
)
VECTORS_FILE = os.path.join(CACHE_DIR, "vectors.npy")
META_FILE = os.path.join(CACHE_DIR, "meta.json")

# Fields every cached row must carry. Add to this when the endpoint starts
# returning a new one: a delta fetch only brings NEW rows, so rows already on
# disk would never gain the field and would stay silently incomplete forever.
# A mismatch discards the cache and refetches the window.
#
# Only list a field the endpoint actually returns. Naming one it doesn't send
# rejects the cache on every start, refetches, saves a cache still lacking the
# field, and rejects it again — a full re-download on every launch, forever.
REQUIRED_FIELDS = ("thread_id", "story_title", "theme", "anchor", "time", "summary")

_lock = threading.Lock()
_vectors: Optional[np.ndarray] = None   # (N, 768) float32, L2-normalized
_meta: List[Dict[str, Any]] = []        # parallel to _vectors, same order
_ready = False
_status = "starting"


def is_ready() -> bool:
    return _ready


def status() -> Dict[str, Any]:
    return {"ready": _ready, "status": _status, "rows": len(_meta)}


def _parse_time(value: str) -> datetime:
    """Parse the endpoint's ISO-UTC timestamps.

    `time` may or may not carry microseconds ('...:14Z' vs '...:14.123456Z').
    Dropping them floors the value, which is what we want for the `since`
    cursor: `since` is whole seconds, and flooring can only ever re-send a row
    we already have (deduped by thread_id), whereas rounding up would silently
    skip one.
    """
    text = value.rstrip("Z").split(".")[0]
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)


def _decode_embedding(encoded: str) -> np.ndarray:
    """base64 of little-endian float32 -> (768,) array. See X-Embedding-Encoding."""
    vec = np.frombuffer(base64.b64decode(encoded), dtype="<f4")
    if vec.shape != (EMBEDDING_DIMS,):
        raise ValueError(f"expected {EMBEDDING_DIMS} dims, got {vec.shape}")
    return vec


def fetch(since: Optional[int] = None, days: int = WINDOW_DAYS) -> List[Dict[str, Any]]:
    """Pull rows from the corpus endpoint (NDJSON, one thread per line).

    A stream that ends WITHOUT the final {"done":true,...} sentinel is
    incomplete and raises. The endpoint emits that line precisely because the
    200 and headers are already sent by the time a mid-stream failure can
    happen — treating a truncated download as a full corpus would silently
    shrink the search pool with nothing in the logs to show for it.
    """
    body: Dict[str, Any] = {"days": days}
    if since is not None:
        body["since"] = since

    rows: List[Dict[str, Any]] = []
    saw_sentinel = False

    with httpx.stream("POST", CORPUS_URL, json=body, timeout=FETCH_TIMEOUT) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.strip():
                continue
            record = json.loads(line)

            if record.get("done"):
                saw_sentinel = True
                break
            if "error" in record:
                raise RuntimeError(f"corpus endpoint failed mid-stream: {record['error']}")

            rows.append(record)

    if not saw_sentinel:
        raise RuntimeError(
            f"corpus stream truncated after {len(rows)} rows (no done sentinel)"
        )

    return rows


def _rebuild(entries: Dict[int, Tuple[Dict[str, Any], np.ndarray]]) -> None:
    """Swap in new state from a {thread_id: (meta, vector)} map."""
    global _vectors, _meta, _ready

    if not entries:
        _vectors, _meta = None, []
        return

    ordered = sorted(entries.items())
    _meta = [meta for _, (meta, _) in ordered]
    matrix = np.stack([vec for _, (_, vec) in ordered]).astype(np.float32)

    # The endpoint's vectors already arrive normalized (every norm measured
    # exactly 1.0), but normalizing here is cheap and makes `search` a plain
    # dot product regardless of what a future producer sends.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.divide(matrix, norms, out=matrix, where=norms > 0)

    _vectors = matrix
    _ready = True


def merge(rows: List[Dict[str, Any]]) -> int:
    """Add rows to the corpus and drop anything outside the 30-day window.

    Deduped on `thread_id`, NOT `story_id` — 30,625 threads span only 5,618
    stories, so story_id would collide on the large majority of rows.

    Aging is ours to do: a delta fetch only ever sends *new* threads, it never
    tells us which old ones have fallen out of the window. Without this the
    local copy would grow forever while still calling itself 30 days.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)

    with _lock:
        entries: Dict[int, Tuple[Dict[str, Any], np.ndarray]] = {}
        if _vectors is not None:
            for i, meta in enumerate(_meta):
                entries[meta["thread_id"]] = (meta, _vectors[i])

        for row in rows:
            try:
                vec = _decode_embedding(row["embedding"])
            except (ValueError, KeyError):
                continue  # a malformed row is not worth failing the whole merge
            meta = {k: row[k] for k in
                    ("thread_id", "story_id", "story_title", "theme", "category",
                     "anchor", "time", "summary")
                    if k in row}
            entries[meta["thread_id"]] = (meta, vec)

        aged = {tid: entry for tid, entry in entries.items()
                if _parse_time(entry[0]["time"]) > cutoff}

        _rebuild(aged)
        return len(aged)


def newest_time() -> Optional[int]:
    """Unix seconds of the newest thread held, for use as the `since` cursor."""
    if not _meta:
        return None
    newest = max(_parse_time(m["time"]) for m in _meta)
    return int(newest.timestamp())


def get(thread_id: int) -> Optional[Dict[str, Any]]:
    """Look up one thread by id.

    A plain dict lookup rather than a network call: the whole 30-day window is
    already in memory, so fetching a thread the user asked about costs nothing
    and works offline. This is why `summary` is worth carrying in the corpus
    payload — without it this returns metadata the model already had from
    search, and answering "what's that one about?" would need a round-trip.
    """
    with _lock:
        for meta in _meta:
            if meta.get("thread_id") == thread_id:
                return dict(meta)
    return None


def search(query_vec: np.ndarray, k: int = 100) -> List[Tuple[Dict[str, Any], float]]:
    """Top-k by cosine similarity. One matrix multiply — ~10ms over 30k rows."""
    with _lock:
        if _vectors is None or not len(_meta):
            return []

        scores = _vectors @ query_vec
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(_meta[i], float(scores[i])) for i in top]


def _load_cache() -> bool:
    """Restore from disk. Any problem returns False and triggers a full fetch."""
    global _vectors, _meta, _ready
    try:
        if not (os.path.exists(VECTORS_FILE) and os.path.exists(META_FILE)):
            return False

        matrix = np.load(VECTORS_FILE)
        with open(META_FILE) as handle:
            meta = json.load(handle)

        if len(meta) != matrix.shape[0] or matrix.shape[1] != EMBEDDING_DIMS:
            return False

        missing = [f for f in REQUIRED_FIELDS if f not in (meta[0] if meta else {})]
        if missing:
            print(f"[corpus] cache is missing {missing}; refetching full window")
            return False

        # Reject a cache that doesn't span the window. Startup only ever does a
        # *delta* on top of the cache, and a delta asks for newer threads — it
        # can never backfill older ones. So a cache written from a partial fetch
        # would stay permanently short with nothing to signal it. If the oldest
        # row is well inside the window, refetch the lot instead.
        oldest = min(_parse_time(m["time"]) for m in meta)
        expected = datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS - 1)
        if oldest > expected:
            print(f"[corpus] cache spans only back to {oldest.date()}; refetching full window")
            return False

        with _lock:
            _vectors, _meta = matrix.astype(np.float32), meta
            _ready = True
        return True
    except Exception as error:  # corrupt/partial file — refetching is cheap enough
        print(f"[corpus] cache unusable ({error}); refetching")
        return False


def _save_cache() -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with _lock:
            if _vectors is None:
                return
            matrix, meta = _vectors, list(_meta)
        # Write-then-rename so an interrupted save can't leave a half-file that
        # looks loadable on the next launch.
        np.save(VECTORS_FILE + ".tmp.npy", matrix)
        os.replace(VECTORS_FILE + ".tmp.npy", VECTORS_FILE)
        with open(META_FILE + ".tmp", "w") as handle:
            json.dump(meta, handle)
        os.replace(META_FILE + ".tmp", META_FILE)
    except Exception as error:
        print(f"[corpus] could not write cache: {error}")


def refresh() -> None:
    """One sync cycle: delta if we have data, full pull otherwise."""
    global _status
    since = newest_time()
    try:
        if since is not None:
            rows = fetch(since=since)
            total = merge(rows)
            print(f"[corpus] delta: +{len(rows)} rows, {total} total")
        else:
            _status = "downloading"
            rows = fetch()
            total = merge(rows)
            print(f"[corpus] full fetch: {total} rows")
        _status = "ready"
        _save_cache()
    except Exception as error:
        # Keep serving whatever we already have rather than dropping to zero.
        _status = f"refresh failed: {error}"
        print(f"[corpus] refresh failed: {error}")


def _loop() -> None:
    if _load_cache():
        print(f"[corpus] loaded {len(_meta)} rows from cache")
    while True:
        refresh()
        _time.sleep(REFRESH_SECONDS)


def start() -> None:
    """Begin syncing in the background.

    Deliberately non-blocking: Gemma takes 1-2 minutes to load and a full corpus
    pull takes ~44s, so running them concurrently makes the download free. Until
    it finishes, /search reports that it is still loading.
    """
    threading.Thread(target=_loop, daemon=True, name="corpus-sync").start()
