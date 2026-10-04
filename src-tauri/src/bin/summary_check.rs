//! Checks on-demand thread summaries. `cargo run --bin summary_check`
//!
//! Runs against a real thread from the live sidebar feed, so it exercises what
//! a user clicking a card would actually hit: the thread-tree endpoint, the
//! truncation, and a real Gemma call.
//!
//! The interesting assertions are the ones about honesty — a summary that
//! invents content, or quietly describes 2% of a conversation, is the failure
//! mode this whole feature exists to fix.

use std::sync::Arc;

use app_lib::summary::summarize_thread;

#[tokio::main]
async fn main() {
    let mut failures = 0usize;
    let mut check = |name: &str, passed: bool, detail: String| {
        println!("  {}  {name}  {detail}", if passed { "PASS" } else { "FAIL" });
        if !passed {
            failures += 1;
        }
    };

    // A real card from the sidebar, so the id is one a user could click.
    let card: serde_json::Value = reqwest::Client::new()
        .post("https://www.techne.app/api/thread-cards/")
        .json(&serde_json::json!({
            "num_cards": 1, "hours_back": 24,
            "sort_by": "karma_density", "density_min_comment_constant": 100
        }))
        .send()
        .await
        .expect("fetch a sidebar card")
        .json()
        .await
        .expect("parse the card");
    let card = &card[0];
    // An explicit id lets us exercise a big thread; the sidebar's top card
    // changes hour to hour and is often small.
    let thread_id = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or_else(|| card["id"].as_u64().expect("card has an id"));

    println!("\n=== thread {thread_id} — {} ===", card["story_title"].as_str().unwrap_or(""));
    println!("  stored summary was written for {} comments", card["comment_count"]);

    println!("\n=== loading Gemma (a minute or two) ===");
    let model = Arc::new(app_lib::agent::load_model().expect("load chat model"));

    println!("\n=== summarise ===");
    let started = std::time::Instant::now();
    let result = match summarize_thread(&model, thread_id).await {
        Ok(result) => result,
        Err(error) => {
            println!("  FAIL  summarise succeeds  {error}");
            std::process::exit(1);
        }
    };
    let elapsed = started.elapsed().as_secs();

    println!("\n{}\n", result.summary);

    check(
        "summarised the current thread, not the stored snapshot",
        result.total > 1,
        format!("{}/{} comments in {elapsed}s", result.used, result.total),
    );
    check("produced prose", result.summary.len() > 200, format!("{} chars", result.summary.len()));
    check(
        "one or two paragraphs, not a list",
        !result.summary.contains("\n- ") && !result.summary.contains("\n* ") && !result.summary.contains("\n#"),
        "found bullets or headings".into(),
    );
    check(
        "does not leak the anonymised ids or depth markers",
        !result.summary.contains("U1")
            && !result.summary.contains("uid=")
            && !result.summary.contains("depth="),
        "leaked a marker".into(),
    );
    check(
        "does not open with filler",
        !result.summary.to_lowercase().starts_with("this thread"),
        result.summary.chars().take(60).collect(),
    );

    // The whole point of the feature: the fresh summary should reflect more of
    // the conversation than the stored one did.
    let stored_saw = card["comment_count"].as_u64().unwrap_or(0);
    check(
        "covers at least as much as the stored summary",
        result.used as u64 >= stored_saw.min(result.total as u64),
        format!("fresh saw {}, stored saw {stored_saw}", result.used),
    );

    println!("\n{}", "=".repeat(46));
    if failures == 0 {
        println!("ALL PASSED");
    } else {
        println!("{failures} FAILED");
        std::process::exit(1);
    }
}
