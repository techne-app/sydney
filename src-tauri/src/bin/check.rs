//! Behaviour checks for the agent. `cargo run --bin check`
//!
//! These drive the real `agent::run_turn`, not a copy of it, so they cannot
//! drift from what the app does. Every check here exists because the behaviour
//! it asserts broke at least once:
//!
//!   · jargon labels — the model echoed our database column names at the user
//!   · URLs in prose — the model retyped a link the UI was also rendering
//!   · pinned vs decoy — "explain this thread" picked the thread just read
//!   · false apology on unpin — the model disowned a correct earlier answer
//!
//! Needs llama-server on :8081 and the Python sidecar on :8000. Slow: each
//! check is one or more real turns through a 12GB model.

use std::sync::Arc;

use app_lib::agent::{run_turn, AgentReply, PinnedThread};
use app_lib::corpus::Corpus;
use app_lib::memory::SqliteConversationMemory;
use app_lib::search::Embedder;

/// A throwaway store, so a run cannot pollute real conversations.
fn scratch_memory() -> Arc<SqliteConversationMemory> {
    let path = std::env::temp_dir().join(format!("techne-check-{}.db", std::process::id()));
    let _ = std::fs::remove_file(&path);
    Arc::new(SqliteConversationMemory::open_at(path).expect("open scratch memory"))
}

fn session(name: &str) -> String {
    format!("check-{name}-{}", std::process::id())
}

/// A thread from the live sidebar, as a user would have pinned it.
///
/// Taken live rather than hardcoded because the attachment no longer carries a
/// summary — it names a thread and the model is expected to go and read it, so
/// the id has to be one the thread service will actually serve. A fixed id would
/// also rot: the corpus holds 30 days, and last month's thread stops being one.
async fn pinned() -> PinnedThread {
    let cards: serde_json::Value = reqwest::Client::new()
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
    let card = &cards[0];
    PinnedThread {
        id: card["id"].as_u64().expect("card has an id"),
        story_title: card["story_title"].as_str().map(str::to_string),
        theme: card["theme"].as_str().map(str::to_string),
    }
}

/// The threads `get_thread` was asked to read this turn.
fn thread_ids(reply: &AgentReply) -> Vec<u64> {
    reply
        .tool_calls
        .iter()
        .filter(|c| c.name == "get_thread")
        .filter_map(|c| c.arguments.get("thread_id").and_then(serde_json::Value::as_u64))
        .collect()
}

struct Report {
    failures: Vec<String>,
}

impl Report {
    fn check(&mut self, name: &str, passed: bool, detail: &str) {
        println!("  {}  {name}", if passed { "PASS" } else { "FAIL" });
        if !passed {
            if !detail.is_empty() {
                println!("        {}", detail.chars().take(120).collect::<String>());
            }
            self.failures.push(name.to_string());
        }
    }
}

fn jargon(text: &str) -> usize {
    ["Theme:", "Category:", "Summary:"]
        .iter()
        .map(|label| text.matches(label).count())
        .sum()
}

fn urls(text: &str) -> usize {
    text.matches("http://").count() + text.matches("https://").count()
}

fn tools(reply: &AgentReply) -> Vec<&str> {
    reply.tool_calls.iter().map(|c| c.name.as_str()).collect()
}

#[tokio::main]
async fn main() {
    let memory = scratch_memory();

    let corpus = Arc::new(Corpus::new());
    if !corpus.load_cached() {
        println!("(no corpus cache, fetching — ~45s)");
        corpus.refresh().await;
    }
    let embedder = Arc::new(Embedder::load().expect("load embedding model"));
    println!("(loading Gemma in-process — a minute or two)");
    let model = Arc::new(app_lib::agent::load_model().expect("load chat model"));
    let store = Arc::new(app_lib::summary::ThreadStore::new());

    let mut report = Report { failures: vec![] };

    macro_rules! turn {
        ($sid:expr, $msg:expr) => {
            run_turn(memory.clone(), corpus.clone(), embedder.clone(), model.clone(), store.clone(), $sid.to_string(), $msg.to_string(), None).await
        };
        ($sid:expr, $msg:expr, $pin:expr) => {
            run_turn(memory.clone(), corpus.clone(), embedder.clone(), model.clone(), store.clone(), $sid.to_string(), $msg.to_string(), Some($pin)).await
        };
    }

    println!("\n=== 1. chit-chat needs no tools");
    let s = session("chat");
    let r = turn!(s, "hey there");
    report.check("no tool call for a greeting", tools(&r).is_empty(), &format!("{:?}", tools(&r)));
    report.check("replies with something", r.reply.len() > 10, &r.reply);

    println!("\n=== 2. search");
    let s = session("search");
    let r = turn!(s, "find me discussions about rust programming");
    report.check("calls search_threads", tools(&r) == ["search_threads"], &format!("{:?}", tools(&r)));
    report.check("no jargon labels", jargon(&r.reply) == 0, &r.reply);
    report.check("model writes no URLs", urls(&r.reply) == 0, &r.reply);
    report.check("no links appended for a search", r.results.is_empty(), "search hits carry no link");
    report.check(
        "reports the keyword for search history",
        r.tool_calls.first().is_some_and(|c| c.arguments.get("keyword_filter").is_some()),
        "",
    );

    println!("\n=== 3. follow-up by position");
    let r = turn!(s, "tell me more about the second one");
    report.check("calls get_thread", tools(&r) == ["get_thread"], &format!("{:?}", tools(&r)));
    report.check("exactly one link", r.results.len() == 1, &format!("{}", r.results.len()));
    report.check(
        "link is a real HN item",
        r.results.first().is_some_and(|x| x.link.contains("news.ycombinator.com/item?id=")),
        "",
    );
    report.check("no jargon", jargon(&r.reply) == 0, &r.reply);
    report.check("model writes no URLs", urls(&r.reply) == 0, &r.reply);

    println!("\n=== 4. history: it remembers the earlier turn");
    let r = turn!(s, "how many discussions did you find for me earlier?");
    let low = r.reply.to_lowercase();
    report.check("recalls the earlier search", low.contains("three") || low.contains('3'), &r.reply);

    println!("\n=== 5. pin mid-conversation, vague wording  <- the regression");
    let pin = pinned().await;
    println!("    pinned: {} — {:?}", pin.id, pin.story_title.as_deref().unwrap_or(""));
    let s = session("pin");
    turn!(s, "find me discussions about career");
    let decoy = turn!(s, "yes, the second one"); // a thread just read in detail
    let decoy_id = thread_ids(&decoy).first().copied();
    let r = turn!(s, "okay, can you explain about this thread now?", pin.clone());
    let read = thread_ids(&r);

    // Checked by id rather than by what the reply says. The old version matched
    // keywords from a hardcoded thread, which could pass on the title alone —
    // and did: a made-up id meant the read failed and the model answered from
    // the attachment, which the keyword check could not tell apart from success.
    report.check(
        "reads the PINNED thread",
        read.contains(&pin.id),
        &format!("read {read:?}, pinned {}", pin.id),
    );
    report.check(
        "not the decoy",
        decoy_id.is_none_or(|d| !read.contains(&d)),
        &format!("read {read:?}, decoy {decoy_id:?}"),
    );
    report.check("says something about it", r.reply.len() > 40, &r.reply);

    println!("\n=== 6. unpin");
    let r = turn!(s, "what is this thread about?");
    let low = r.reply.to_lowercase();
    let apology = ["i made", "i invented", "hallucinat", "i fabricated"].iter().any(|w| low.contains(w));
    report.check("does not falsely apologise", !apology, &r.reply);

    println!("\n=== 7. query with nothing to match");
    let s = session("nomatch");
    let r = turn!(s, "find discussions about medieval Bulgarian goat farming subsidies");
    report.check("still answers, no crash", r.reply.len() > 10, &r.reply);
    report.check("no jargon", jargon(&r.reply) == 0, &r.reply);

    println!("\n{}", "=".repeat(46));
    if report.failures.is_empty() {
        println!("ALL PASSED");
    } else {
        println!("{} FAILED: {}", report.failures.len(), report.failures.join(", "));
        std::process::exit(1);
    }
}
