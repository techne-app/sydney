//! Semantic search over the 30-day corpus.
//!
//! Ported from `sidecar/search.py`. Embed the query with nomic, take the
//! nearest rows one-per-story, then have Gemma rerank that shortlist down to a
//! handful.
//!
//! Only one string is embedded per search — the query. The corpus rows arrive
//! with their vectors already computed, which is what lets 31,000 candidates
//! cost less per query than the old in-browser search's 30 did.

use std::sync::Arc;

use rig_core::client::CompletionClient;
use rig_core::completion::Prompt;
use rig_core::embeddings::EmbeddingModel;
use rig_llama_cpp::{EmbeddingClient, EmbeddingModelHandle};

use crate::corpus::Corpus;

/// It MUST be the model that produced the stored vectors — nomic-embed-text-v1.5,
/// 768-dim. A query embedded by any other model lands in a different vector
/// space, and the cosine against it returns noise rather than weak matches.
const EMBED_MODEL: &str = "nomic-embed-text-v1.5.Q8_0.gguf";

/// How many candidates the embedding stage hands the reranker. Mirrors the
/// pipeline's feed generation, which cosines to 100 then has the LLM pick a few.
const RERANK_CANDIDATES: usize = 100;


/// Locate a GGUF in the bundled .app, or in the dev checkout.
///
/// Production is one fixed hop from the executable. Dev is not: `cargo run`,
/// `cargo run --bin check` and `tauri dev` all sit at different depths under
/// target/, so counting `..` gets it wrong for at least one of them. Walking up
/// until `models/` appears works from all of them.
pub fn resolve_model(name: &str) -> String {
    let exe = std::env::current_exe().unwrap_or_default();
    let base = exe.parent().unwrap_or(std::path::Path::new("."));

    // Production: Contents/MacOS/app -> Contents/Resources/models/
    let bundled = base.join("../Resources/models").join(name);
    if bundled.exists() {
        return bundled.to_string_lossy().into_owned();
    }

    // Dev: walk up to the repo's own `models/`. Build directories are skipped
    // deliberately — Tauri copies `bundle.resources` to `target/debug/models/`,
    // which sits directly beside the dev binary and would otherwise win every
    // time. It holds the same weights today, but it is a build artifact: it
    // survives a change to tauri.conf.json, and still has a Qwen3 from before
    // the Gemma swap. Loading the source is the one that is always current.
    for dir in base.ancestors() {
        if dir.components().any(|c| c.as_os_str() == "target") {
            continue;
        }
        let candidate = dir.join("models").join(name);
        if candidate.exists() {
            return candidate.to_string_lossy().into_owned();
        }
    }
    // Nothing found: hand back the plain name so the error names the model
    // rather than a misleading path.
    name.to_string()
}

pub struct Embedder {
    /// Dropping this shuts the worker thread down, so it has to outlive the
    /// handle even though the handle carries its own channel.
    _client: EmbeddingClient,
    model: EmbeddingModelHandle,
}

impl Embedder {
    /// nomic runs on the CPU deliberately: it is 139MB embedding one short
    /// string per search, and leaving the GPU to the chat model matters more.
    pub fn load() -> Result<Self, String> {
        let path = resolve_model(EMBED_MODEL);
        println!("[search] loading embedding model from {path}");
        let client = EmbeddingClient::from_gguf(path, 0, 2048).map_err(|e| e.to_string())?;
        let model = client.embedding_model(EMBED_MODEL);
        println!("[search] embedding model loaded ({} dims)", model.ndims());
        Ok(Self {
            _client: client,
            model,
        })
    }

    /// The "search_query: " prefix is required, not decorative: every theme was
    /// stored with "search_document: ", and nomic places queries and documents
    /// differently. The wrong prefix degrades relevance silently — no error.
    ///
    /// The result is normalized because the corpus scores with a plain dot
    /// product, which only equals the cosine when both sides are unit length.
    /// llama.cpp hands back a raw vector (measured: length ~21), so without
    /// this every score is scaled by an arbitrary constant. Ranking still looks
    /// right — the order is unchanged — so the only symptom is that the scores
    /// themselves are meaningless, which is exactly the kind of thing that goes
    /// unnoticed until something compares one against a threshold.
    pub async fn embed_query(&self, query: &str) -> Result<Vec<f32>, String> {
        let embedding = self
            .model
            .embed_text(&format!("search_query: {query}"))
            .await
            .map_err(|e| e.to_string())?;

        let vector: Vec<f32> = embedding.vec.into_iter().map(|v| v as f32).collect();
        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        Ok(if norm > 0.0 {
            vector.into_iter().map(|v| v / norm).collect()
        } else {
            vector
        })
    }
}

fn build_rerank_prompt(query: &str, candidates: &[(crate::corpus::ThreadMeta, f32)], limit: usize) -> String {
    let listing = candidates
        .iter()
        .enumerate()
        .map(|(i, (meta, _))| {
            format!(
                "{}. {} — {}",
                i + 1,
                meta.story_title.as_deref().unwrap_or(""),
                meta.theme.as_deref().unwrap_or("")
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "A user searched Hacker News for: \"{query}\"\n\n\
         Below are {} candidate discussions, already filtered by semantic \
         similarity. Pick the {limit} most genuinely relevant to what the user \
         is asking for.\n\n{listing}\n\n\
         Reply with ONLY the {limit} numbers, best first, separated by commas. \
         No explanation, no other text.",
        candidates.len()
    )
}

/// Pull the chosen indices out of the model's reply.
///
/// Deliberately scrapes integers rather than asking for JSON or a tool call — a
/// bare list of numbers is the one format that is hard to get wrong. Out-of-range
/// or duplicate numbers are dropped; if nothing usable comes back the caller
/// falls back to plain cosine order.
fn parse_rerank_picks(text: &str, count: usize, limit: usize) -> Vec<usize> {
    let mut picks = Vec::new();
    let mut digits = String::new();
    for ch in text.chars().chain(std::iter::once(' ')) {
        if ch.is_ascii_digit() {
            digits.push(ch);
            continue;
        }
        if !digits.is_empty() {
            if let Ok(n) = digits.parse::<usize>() {
                if n >= 1 && n <= count && !picks.contains(&(n - 1)) {
                    picks.push(n - 1);
                }
            }
            digits.clear();
        }
        if picks.len() == limit {
            break;
        }
    }
    picks
}

/// Full search: embed, nearest-per-story, rerank. Never fails outright — a
/// rerank that errors falls through to cosine order rather than failing the
/// search.
pub async fn run_search(
    corpus: &Arc<Corpus>,
    embedder: &Embedder,
    model: &Arc<rig_llama_cpp::Client>,
    query: &str,
    limit: usize,
) -> serde_json::Value {
    if !corpus.is_ready() {
        return serde_json::json!({
            "results": [],
            "error": "Thread data is still loading, try again shortly."
        });
    }

    let query = query.trim();
    if query.is_empty() {
        return serde_json::json!({ "results": [], "error": "Empty query." });
    }

    let vector = match embedder.embed_query(query).await {
        Ok(vector) => vector,
        Err(error) => {
            return serde_json::json!({ "results": [], "error": error });
        }
    };

    let candidates = corpus.nearest_per_story(&vector, RERANK_CANDIDATES);
    if candidates.is_empty() {
        return serde_json::json!({ "results": [], "error": "No matching discussions found." });
    }

    let picks = rerank(model, query, &candidates, limit).await;
    let picks = if picks.is_empty() {
        (0..limit.min(candidates.len())).collect()
    } else {
        picks
    };

    let results: Vec<_> = picks
        .into_iter()
        .map(|i| crate::corpus::public_view(&candidates[i].0, false))
        .collect();
    serde_json::json!({ "results": results })
}

/// One completion against the very same loaded model the agent uses — the
/// handle is shared, so there is one Gemma in memory, not two. That sharing is
/// why the corpus had to move to Rust first: while the reranker was still
/// Python, going in-process would have meant a second 12GB copy.
async fn rerank(
    model: &Arc<rig_llama_cpp::Client>,
    query: &str,
    candidates: &[(crate::corpus::ThreadMeta, f32)],
    limit: usize,
) -> Vec<usize> {
    let agent = rig_core::agent::AgentBuilder::new(model.completion_model("gemma"))
        .preamble("You rank search results. Answer with numbers only.")
        // In-process the key is `thinking`, NOT the HTTP-era
        // `chat_template_kwargs.enable_thinking` — rig-llama-cpp reads it from
        // additional_params and feeds it to the Jinja template directly. The old
        // key is silently ignored, and Gemma then spends its whole token budget
        // thinking and returns empty content with finish_reason=Length.
        .additional_params(serde_json::json!({ "thinking": false }))
        .temperature(0.1)
        .max_tokens(64)
        .build();

    match agent.prompt(build_rerank_prompt(query, candidates, limit)).await {
        Ok(reply) => parse_rerank_picks(&reply, candidates.len(), limit),
        Err(error) => {
            println!("[search] rerank failed, using cosine order: {error}");
            vec![]
        }
    }
}
