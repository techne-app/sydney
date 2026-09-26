//! The 30-day thread corpus: fetch, cache, age, and vector search.
//!
//! Ported from `sidecar/corpus.py`. Search used to run in the webview and could
//! only see 30 stories, because it embedded every candidate per query. The
//! endpoint hands us pre-computed 768-dim vectors instead, so the expensive step
//! disappears: we embed one string per search — the query — regardless of pool
//! size. That is what makes 30,000 candidates cheaper per query than 30 was.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;

use base64::Engine;
use chrono::{DateTime, Duration, TimeZone, Utc};
use serde::{Deserialize, Serialize};

const CORPUS_URL: &str = "https://techne.app/api/threads-30d/";
const WINDOW_DAYS: i64 = 30;
const EMBEDDING_DIMS: usize = 768;
const REFRESH_SECONDS: u64 = 15 * 60;
/// A cold start on the endpoint has taken 107s; the usual case is 2–3s for a
/// delta and ~45s for the full window.
const FETCH_TIMEOUT_SECONDS: u64 = 300;

/// One thread, as we keep it. `embedding` is held separately in a flat matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadMeta {
    pub thread_id: u64,
    pub story_id: Option<u64>,
    pub story_title: Option<String>,
    pub theme: Option<String>,
    pub category: Option<String>,
    pub anchor: Option<String>,
    pub time: String,
    pub summary: Option<String>,
}

/// A row exactly as the endpoint sends it, before the embedding is split off.
#[derive(Debug, Deserialize)]
struct WireRow {
    #[serde(flatten)]
    meta: ThreadMeta,
    embedding: String,
}

#[derive(Default)]
struct Inner {
    /// Row-major, `meta.len() * EMBEDDING_DIMS`. One flat Vec rather than a Vec
    /// of Vecs so scoring is a single pass over contiguous memory.
    vectors: Vec<f32>,
    meta: Vec<ThreadMeta>,
    ready: bool,
}

pub struct Corpus {
    inner: RwLock<Inner>,
    cache_dir: PathBuf,
}

fn default_cache_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home)
        .join("Library/Application Support/com.technesystems.techne")
        .join("corpus")
}

/// Parse the endpoint's ISO-UTC timestamps.
///
/// `time` may or may not carry microseconds. Dropping them floors the value,
/// which is what the `since` cursor wants: flooring can only re-send a row we
/// already have (deduped by thread_id), whereas rounding up would silently skip
/// one.
fn parse_time(value: &str) -> Option<DateTime<Utc>> {
    let text = value.trim_end_matches('Z').split('.').next()?;
    chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S")
        .ok()
        .map(|naive| Utc.from_utc_datetime(&naive))
}

/// base64 of little-endian float32 -> 768 floats. See the endpoint's
/// X-Embedding-Encoding header.
fn decode_embedding(encoded: &str) -> Option<Vec<f32>> {
    let bytes = base64::engine::general_purpose::STANDARD.decode(encoded).ok()?;
    if bytes.len() != EMBEDDING_DIMS * 4 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
    )
}

impl Corpus {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(Inner::default()),
            cache_dir: default_cache_dir(),
        }
    }

    pub fn is_ready(&self) -> bool {
        self.inner.read().map(|i| i.ready).unwrap_or(false)
    }

    pub fn len(&self) -> usize {
        self.inner.read().map(|i| i.meta.len()).unwrap_or(0)
    }

    pub fn get(&self, thread_id: u64) -> Option<ThreadMeta> {
        let inner = self.inner.read().ok()?;
        inner.meta.iter().find(|m| m.thread_id == thread_id).cloned()
    }

    /// One row's stored vector. The corpus check queries with a row's own
    /// embedding, which must return that row first.
    pub fn vector_of(&self, thread_id: u64) -> Option<Vec<f32>> {
        let inner = self.inner.read().ok()?;
        let index = inner.meta.iter().position(|m| m.thread_id == thread_id)?;
        let start = index * EMBEDDING_DIMS;
        Some(inner.vectors[start..start + EMBEDDING_DIMS].to_vec())
    }

    /// Every thread id held, in storage order.
    pub fn thread_ids(&self) -> Vec<u64> {
        self.inner
            .read()
            .map(|i| i.meta.iter().map(|m| m.thread_id).collect())
            .unwrap_or_default()
    }

    /// Load the on-disk cache without starting the refresh loop.
    pub fn load_cached(&self) -> bool {
        self.load_cache()
    }

    /// The k rows nearest this vector — an index lookup, not the search
    /// feature. Named `nearest` to keep that distinction: `search::run_search`
    /// is what answers a user's question, and it calls this as one step.
    ///
    /// The stored vectors are already normalized, so a dot product IS the
    /// cosine — one pass over ~30k rows, a few milliseconds.
    pub fn nearest(&self, query: &[f32], k: usize) -> Vec<(ThreadMeta, f32)> {
        let Ok(inner) = self.inner.read() else {
            return vec![];
        };
        if inner.meta.is_empty() || query.len() != EMBEDDING_DIMS {
            return vec![];
        }

        let mut scored: Vec<(usize, f32)> = inner
            .vectors
            .chunks_exact(EMBEDDING_DIMS)
            .enumerate()
            .map(|(i, row)| {
                let score = row.iter().zip(query).map(|(a, b)| a * b).sum::<f32>();
                (i, score)
            })
            .collect();

        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        scored
            .into_iter()
            .take(k)
            .map(|(i, score)| (inner.meta[i].clone(), score))
            .collect()
    }

    /// The k nearest rows, at most one per story.
    ///
    /// A row is one comment thread, not one story: a busy post contributes
    /// dozens of near-identical rows, so without this a query returns three
    /// threads from the same discussion and the same title three times.
    ///
    /// Done in the same pass as scoring rather than by over-fetching and
    /// filtering afterwards. Python asked for 4k rows and hoped k distinct
    /// stories survived — if the top 400 happened to come from 60 stories, you
    /// silently got 60 candidates instead of 100.
    pub fn nearest_per_story(&self, query: &[f32], k: usize) -> Vec<(ThreadMeta, f32)> {
        let Ok(inner) = self.inner.read() else {
            return vec![];
        };
        if inner.meta.is_empty() || query.len() != EMBEDDING_DIMS {
            return vec![];
        }

        let mut scored: Vec<(usize, f32)> = inner
            .vectors
            .chunks_exact(EMBEDDING_DIMS)
            .enumerate()
            .map(|(i, row)| (i, row.iter().zip(query).map(|(a, b)| a * b).sum::<f32>()))
            .collect();
        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::with_capacity(k);
        for (i, score) in scored {
            let meta = &inner.meta[i];
            // A row with no story_id cannot be deduped, so let it through.
            if meta.story_id.is_some_and(|id| !seen.insert(id)) {
                continue;
            }
            out.push((meta.clone(), score));
            if out.len() == k {
                break;
            }
        }
        out
    }

    /// Unix seconds of the newest thread held, for use as the `since` cursor.
    pub fn newest_time(&self) -> Option<i64> {
        let inner = self.inner.read().ok()?;
        inner
            .meta
            .iter()
            .filter_map(|m| parse_time(&m.time))
            .max()
            .map(|t| t.timestamp())
    }

    /// Add rows and drop anything outside the 30-day window.
    ///
    /// Deduped on `thread_id`, NOT `story_id` — ~30k threads span only ~5.6k
    /// stories, so story_id would collide on the large majority of rows.
    ///
    /// Aging is ours to do: a delta fetch only ever sends *new* threads, it
    /// never says which old ones have fallen out of the window. Without this the
    /// local copy would grow forever while still calling itself 30 days.
    fn merge(&self, rows: Vec<WireRow>) -> usize {
        let cutoff = Utc::now() - Duration::days(WINDOW_DAYS);
        let mut inner = match self.inner.write() {
            Ok(inner) => inner,
            Err(_) => return 0,
        };

        let mut entries: HashMap<u64, (ThreadMeta, Vec<f32>)> = inner
            .meta
            .iter()
            .enumerate()
            .map(|(i, meta)| {
                let start = i * EMBEDDING_DIMS;
                (
                    meta.thread_id,
                    (meta.clone(), inner.vectors[start..start + EMBEDDING_DIMS].to_vec()),
                )
            })
            .collect();

        for row in rows {
            // A malformed row is skipped, not fatal — one bad embedding is not
            // worth failing a 30,000-row merge.
            if let Some(vector) = decode_embedding(&row.embedding) {
                entries.insert(row.meta.thread_id, (row.meta, vector));
            }
        }

        entries.retain(|_, (meta, _)| parse_time(&meta.time).is_some_and(|t| t > cutoff));

        let mut meta = Vec::with_capacity(entries.len());
        let mut vectors = Vec::with_capacity(entries.len() * EMBEDDING_DIMS);
        for (_, (m, v)) in entries {
            meta.push(m);
            vectors.extend_from_slice(&v);
        }

        inner.meta = meta;
        inner.vectors = vectors;
        inner.ready = true;
        inner.meta.len()
    }

    /// Pull rows from the corpus endpoint (NDJSON, one thread per line).
    ///
    /// A stream that ends WITHOUT the final {"done":true} sentinel is incomplete
    /// and fails. The endpoint emits that line precisely because the 200 and
    /// headers are already sent by the time a mid-stream failure can happen —
    /// treating a truncated download as a full corpus would silently shrink the
    /// search pool with nothing in the logs to show for it.
    async fn fetch(&self, since: Option<i64>) -> Result<Vec<WireRow>, String> {
        let mut body = serde_json::json!({ "days": WINDOW_DAYS });
        if let Some(since) = since {
            body["since"] = since.into();
        }

        let http = reqwest::Client::new();
        let response = http
            .post(CORPUS_URL)
            .json(&body)
            .timeout(std::time::Duration::from_secs(FETCH_TIMEOUT_SECONDS))
            .send()
            .await
            .map_err(|e| e.to_string())?
            .error_for_status()
            .map_err(|e| e.to_string())?;

        let text = response.text().await.map_err(|e| e.to_string())?;

        let mut rows = Vec::new();
        let mut saw_sentinel = false;
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let record: serde_json::Value = match serde_json::from_str(line) {
                Ok(value) => value,
                Err(_) => continue,
            };
            if record.get("done").is_some_and(|v| v.as_bool() == Some(true)) {
                saw_sentinel = true;
                break;
            }
            if let Some(error) = record.get("error") {
                return Err(format!("corpus endpoint failed mid-stream: {error}"));
            }
            if let Ok(row) = serde_json::from_value::<WireRow>(record) {
                rows.push(row);
            }
        }

        if !saw_sentinel {
            return Err(format!(
                "corpus stream truncated after {} rows (no done sentinel)",
                rows.len()
            ));
        }
        Ok(rows)
    }

    // -- cache ---------------------------------------------------------------
    //
    // Raw little-endian f32 for the vectors — the same encoding the wire uses —
    // plus JSON for the metadata. Python wrote .npy here; that file is not read
    // back, so the first run after this change refetches the full window once.

    fn vectors_file(&self) -> PathBuf {
        self.cache_dir.join("vectors.f32")
    }

    fn meta_file(&self) -> PathBuf {
        self.cache_dir.join("meta.json")
    }

    /// Restore from disk. Any problem returns false and triggers a full fetch.
    fn load_cache(&self) -> bool {
        let Ok(bytes) = std::fs::read(self.vectors_file()) else {
            return false;
        };
        let Ok(json) = std::fs::read_to_string(self.meta_file()) else {
            return false;
        };
        let Ok(meta) = serde_json::from_str::<Vec<ThreadMeta>>(&json) else {
            println!("[corpus] cache metadata unreadable; refetching");
            return false;
        };

        if bytes.len() != meta.len() * EMBEDDING_DIMS * 4 || meta.is_empty() {
            return false;
        }

        // Reject a cache that does not span the window. Startup only ever does a
        // *delta* on top of the cache, and a delta asks for newer threads — it
        // can never backfill older ones. So a cache written from a partial fetch
        // would stay permanently short with nothing to signal it.
        let Some(oldest) = meta.iter().filter_map(|m| parse_time(&m.time)).min() else {
            return false;
        };
        if oldest > Utc::now() - Duration::days(WINDOW_DAYS - 1) {
            println!(
                "[corpus] cache spans only back to {}; refetching full window",
                oldest.date_naive()
            );
            return false;
        }

        let vectors: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        match self.inner.write() {
            Ok(mut inner) => {
                inner.vectors = vectors;
                inner.meta = meta;
                inner.ready = true;
                true
            }
            Err(_) => false,
        }
    }

    /// Write-then-rename, so an interrupted save cannot leave a half-file that
    /// looks loadable on the next launch.
    fn save_cache(&self) {
        let Ok(inner) = self.inner.read() else { return };
        if inner.meta.is_empty() {
            return;
        }
        let _ = std::fs::create_dir_all(&self.cache_dir);

        let mut bytes = Vec::with_capacity(inner.vectors.len() * 4);
        for value in &inner.vectors {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        let write = |path: PathBuf, data: &[u8]| {
            let tmp = path.with_extension("tmp");
            if std::fs::write(&tmp, data).is_ok() {
                let _ = std::fs::rename(&tmp, &path);
            }
        };
        write(self.vectors_file(), &bytes);
        if let Ok(json) = serde_json::to_vec(&inner.meta) {
            write(self.meta_file(), &json);
        }
    }

    /// One refresh cycle: delta if we have rows, full window otherwise.
    pub async fn refresh(&self) {
        let since = self.newest_time();
        match self.fetch(since).await {
            Ok(rows) => {
                let added = rows.len();
                let total = self.merge(rows);
                println!(
                    "[corpus] {}: +{added} rows, {total} total",
                    if since.is_some() { "delta" } else { "full fetch" }
                );
                self.save_cache();
            }
            Err(error) => println!("[corpus] refresh failed: {error}"),
        }
    }
}

/// Load the cache, then refresh in the background.
///
/// Fetching must not block the app coming up: the corpus download takes ~45s
/// cold, and the UI has plenty to show before search is needed.
///
/// Spawned through Tauri's runtime handle rather than `tokio::spawn`, which
/// panics with "there is no reactor running" when called from Tauri's setup —
/// that code is not itself inside a runtime. The check binaries never hit this
/// because they run under #[tokio::main].
pub fn start(corpus: std::sync::Arc<Corpus>) {
    tauri::async_runtime::spawn(async move {
        if corpus.load_cache() {
            println!("[corpus] loaded {} rows from cache", corpus.len());
        }
        loop {
            corpus.refresh().await;
            tokio::time::sleep(std::time::Duration::from_secs(REFRESH_SECONDS)).await;
        }
    });
}

/// A thread as the model should see it.
///
/// Our column names are database vocabulary. "theme", "category", "summary",
/// "anchor", "story_title" mean something to us and nothing to the person
/// reading the reply — and the model echoes the shape it is handed, so given a
/// `theme` key it writes "Theme: …" in prose meant for someone who has never
/// heard of it. `category` is dropped rather than renamed: values like
/// "Industry Analysis" are our taxonomy, and answer nothing the user asks.
///
/// No link, deliberately, not even on the detail view. Handed a URL the model
/// writes it into its prose and the UI then renders the same link a second time.
pub fn public_view(meta: &ThreadMeta, detail: bool) -> serde_json::Value {
    let mut view = serde_json::json!({
        "thread_id": meta.thread_id,
        "title": meta.story_title,
        "about": meta.theme,
    });
    if detail {
        view["discussion"] = meta.summary.clone().into();
    }
    view
}
