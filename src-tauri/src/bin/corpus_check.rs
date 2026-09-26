//! Proves the Rust corpus port against the live endpoint. `cargo run --bin corpus_check`
//!
//! No model needed. Search is checked by querying with a row's OWN vector: that
//! row must come back first with a score near 1.0, which exercises the whole
//! path — base64 decode, the flat matrix layout, the dot product — with no
//! embedding model in the way.

use std::sync::Arc;

use app_lib::corpus::Corpus;

#[tokio::main]
async fn main() {
    let corpus = Arc::new(Corpus::new());
    let mut failures = 0usize;
    let mut check = |name: &str, passed: bool, detail: String| {
        println!("  {}  {name}  {detail}", if passed { "PASS" } else { "FAIL" });
        if !passed {
            failures += 1;
        }
    };

    println!("\n=== fetch the full window (~45s cold) ===");
    let started = std::time::Instant::now();
    corpus.refresh().await;
    check(
        "fetched a full corpus",
        corpus.len() > 20_000,
        format!("{} rows in {}s", corpus.len(), started.elapsed().as_secs()),
    );
    check("ready", corpus.is_ready(), String::new());
    check("newest_time gives a since cursor", corpus.newest_time().is_some(), String::new());

    println!("\n=== a row carries everything downstream needs ===");
    let id = corpus.thread_ids().first().copied().unwrap_or(0);
    match corpus.get(id) {
        Some(meta) => {
            check("get(thread_id) finds it", true, format!("#{}", meta.thread_id));
            check("title", meta.story_title.is_some(), String::new());
            check("anchor (the UI's link)", meta.anchor.is_some(), String::new());
            check("summary (what get_thread returns)", meta.summary.is_some(), String::new());
            check("time (for aging)", !meta.time.is_empty(), meta.time.clone());
        }
        None => check("get(thread_id) finds it", false, "no rows".into()),
    }

    println!("\n=== search: a row's own vector must find itself ===");
    match corpus.vector_of(id) {
        Some(vector) => {
            let hits = corpus.nearest(&vector, 3);
            let first = hits.first();
            check(
                "returns itself first",
                first.is_some_and(|(m, _)| m.thread_id == id),
                format!("got #{:?}", first.map(|(m, _)| m.thread_id)),
            );
            check(
                "with a score near 1.0",
                first.is_some_and(|(_, s)| *s > 0.99),
                format!("score {:.4}", first.map(|(_, s)| *s).unwrap_or(0.0)),
            );
            check("returns k hits", hits.len() == 3, format!("{}", hits.len()));
            check(
                "ranked descending",
                hits.windows(2).all(|w| w[0].1 >= w[1].1),
                String::new(),
            );
        }
        None => check("row has a vector", false, String::new()),
    }

    println!("\n=== nearest_per_story: no story twice ===");
    match corpus.vector_of(id) {
        Some(vector) => {
            let hits = corpus.nearest_per_story(&vector, 100);
            let stories: std::collections::HashSet<_> =
                hits.iter().filter_map(|(m, _)| m.story_id).collect();
            check("returns the full k", hits.len() == 100, format!("{}", hits.len()));
            check(
                "every story appears once",
                stories.len() == hits.iter().filter(|(m, _)| m.story_id.is_some()).count(),
                format!("{} stories in {} hits", stories.len(), hits.len()),
            );
            check(
                "still ranked descending",
                hits.windows(2).all(|w| w[0].1 >= w[1].1),
                String::new(),
            );
        }
        None => check("row has a vector", false, String::new()),
    }

    println!("\n=== cache round trip ===");
    let fresh = Corpus::new();
    let loaded = fresh.load_cached();
    check("a new instance loads the cache", loaded, format!("{} rows", fresh.len()));
    check(
        "same row count as the fetch",
        fresh.len() == corpus.len(),
        format!("{} vs {}", fresh.len(), corpus.len()),
    );
    check(
        "vectors survived the round trip",
        fresh.vector_of(id) == corpus.vector_of(id),
        String::new(),
    );

    println!("\n{}", "=".repeat(46));
    if failures == 0 {
        println!("ALL PASSED");
    } else {
        println!("{failures} FAILED");
        std::process::exit(1);
    }
}
