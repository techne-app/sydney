//! Proves the Rust search port. `cargo run --bin search_check`
//!
//! Needs llama-server on :8081 for the rerank step. The corpus is fetched
//! directly, so the Python sidecar is not involved.
//!
//! The assertion that matters is relevance: a query about Rust has to return
//! threads about Rust. If the "search_query: " prefix were missing, or nomic
//! were the wrong model, nothing would error — the results would just quietly
//! become noise. That is the failure this guards against.

use std::sync::Arc;

use app_lib::corpus::Corpus;
use app_lib::search::{run_search, Embedder};

fn titles(value: &serde_json::Value) -> Vec<String> {
    value["results"]
        .as_array()
        .map(|rows| {
            rows.iter()
                .map(|r| {
                    format!(
                        "{} — {}",
                        r["title"].as_str().unwrap_or("?"),
                        r["about"].as_str().unwrap_or("?")
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

#[tokio::main]
async fn main() {
    let mut failures = 0usize;
    let mut check = |name: &str, passed: bool, detail: String| {
        println!("  {}  {name}  {detail}", if passed { "PASS" } else { "FAIL" });
        if !passed {
            failures += 1;
        }
    };

    println!("\n=== load the corpus ===");
    let corpus = Arc::new(Corpus::new());
    if !corpus.load_cached() {
        println!("  (no cache, fetching — ~45s)");
        corpus.refresh().await;
    }
    check("corpus ready", corpus.is_ready(), format!("{} rows", corpus.len()));

    println!("\n=== load nomic in-process ===");
    let started = std::time::Instant::now();
    println!("(loading Gemma for the rerank step — a minute or two)");
    let model = std::sync::Arc::new(app_lib::agent::load_model().expect("load chat model"));

    let embedder = match Embedder::load() {
        Ok(embedder) => embedder,
        Err(error) => {
            println!("  FAIL  embedding model loads  {error}");
            std::process::exit(1);
        }
    };
    check("embedding model loads", true, format!("{}s", started.elapsed().as_secs()));

    let vector = embedder.embed_query("rust async runtimes").await;
    match &vector {
        Ok(v) => {
            check("query embeds to 768 dims", v.len() == 768, format!("{}", v.len()));
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            // nomic returns normalized vectors; the corpus assumes it, because
            // the dot product is only the cosine when both sides are unit length.
            check("vector is normalized", (norm - 1.0).abs() < 0.05, format!("norm {norm:.4}"));
        }
        Err(error) => check("query embeds", false, error.clone()),
    }

    println!("\n=== relevance: does a Rust query return Rust threads? ===");
    let started = std::time::Instant::now();
    let results = run_search(&corpus, &embedder, &model, "rust async runtimes", 3).await;
    let rows = titles(&results);
    for row in &rows {
        println!("     · {row}");
    }
    check("returns 3 results", rows.len() == 3, format!("{} in {}s", rows.len(), started.elapsed().as_secs()));
    check(
        "at least one mentions rust or async",
        rows.iter().any(|r| {
            let low = r.to_lowercase();
            low.contains("rust") || low.contains("async") || low.contains("concurren")
        }),
        String::new(),
    );

    println!("\n=== a different query returns different threads ===");
    let other = run_search(&corpus, &embedder, &model, "sourdough bread baking", 3).await;
    let other_rows = titles(&other);
    for row in &other_rows {
        println!("     · {row}");
    }
    check(
        "results differ from the rust query",
        other_rows != rows,
        String::new(),
    );

    println!("\n=== shape the model sees ===");
    if let Some(first) = results["results"].as_array().and_then(|r| r.first()) {
        let keys: Vec<_> = first.as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default();
        check(
            "hit is thread_id/title/about only",
            keys.len() == 3 && keys.contains(&"thread_id".to_string()),
            format!("{keys:?}"),
        );
        // A search hit carrying a link is what made the UI append a second copy
        // of the list the model had just written.
        check("no link on a search hit", first.get("link").is_none(), String::new());
        check("no summary on a search hit", first.get("discussion").is_none(), String::new());
    }

    println!("\n=== empty query is handled ===");
    let empty = run_search(&corpus, &embedder, &model, "   ", 3).await;
    check("returns an error, not a panic", empty["error"].is_string(), String::new());

    println!("\n{}", "=".repeat(46));
    if failures == 0 {
        println!("ALL PASSED");
    } else {
        println!("{failures} FAILED");
        std::process::exit(1);
    }
}
