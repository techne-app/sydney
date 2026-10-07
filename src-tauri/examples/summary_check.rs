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

use app_lib::summary::ThreadStore;

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
    let store = ThreadStore::new();
    let result = match store.summary(&model, thread_id).await {
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
    // The shape the modal lays out: an opening, quotations from the discussion,
    // a closing. Quotations are the part worth reading and the only part that
    // can be verified, so a summary without one has lost the point.
    let quotes = result
        .summary
        .lines()
        .filter(|l| l.trim_start().starts_with(['"', '\u{201c}']))
        .count();
    check(
        "quotes the discussion",
        quotes >= 2,
        format!("{quotes} quotation(s)"),
    );
    check(
        "says something either side of the quotes",
        result.summary.lines().filter(|l| {
            !l.trim().is_empty() && !l.trim_start().starts_with(['"', '\u{201c}'])
        }).count() >= 2,
        "opening and closing".into(),
    );
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

    // Reads as much of the thread as it can hold.
    //
    // This used to compare `used` against the card's `comment_count`, which was
    // wrong twice over: that number is the thread's size, not what the stored
    // summary saw, and we clip on purpose, so "read every comment" can never
    // pass on a busy thread. It only ever passed because an explicit thread id
    // left it comparing one thread's reading against another thread's size.
    //
    // What actually matters is that nothing is wasted: either the whole thread
    // fitted, or the budget came out close to full. A long comment early on used
    // to end the loop and discard every shorter one behind it, leaving 18% of
    // the budget unspent — this is the check that would have caught it.
    let whole_thread = result.used == result.total;
    check(
        "reads as much of the thread as it can hold",
        whole_thread || result.chars * 100 / 14_000 >= 90,
        if whole_thread {
            format!("whole thread, {} comments", result.total)
        } else {
            format!("{}/{} comments, {} of 14,000 chars", result.used, result.total, result.chars)
        },
    );

    println!("\n{}", "=".repeat(46));

    // Free the model before the process tears down. llama.cpp frees the Metal
    // device in a static destructor, and aborts there if a model is still
    // alive — the same crash `Loaded` exists to prevent in the app. Without
    // this the run ends in a GGML_ASSERT backtrace after the results, which
    // looks exactly like a failure and is not one.
    drop(store);
    drop(model);

    if failures == 0 {
        println!("ALL PASSED");
    } else {
        println!("{failures} FAILED");
        std::process::exit(1);
    }
}
