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
use app_lib::memory::SqliteConversationMemory;

/// A throwaway store, so a run cannot pollute real conversations.
fn scratch_memory() -> Arc<SqliteConversationMemory> {
    let path = std::env::temp_dir().join(format!("techne-check-{}.db", std::process::id()));
    let _ = std::fs::remove_file(&path);
    Arc::new(SqliteConversationMemory::open_at(path).expect("open scratch memory"))
}

fn session(name: &str) -> String {
    format!("check-{name}-{}", std::process::id())
}

fn pinned() -> PinnedThread {
    PinnedThread {
        story_title: Some("100s of flights cancelled at UK airports due to ATC issue".into()),
        theme: Some("Corporate communication failures".into()),
        summary: Some(
            "Users value detailed frequent status updates over polite vague PR statements, \
             and argue withholding information is a strategic choice to manage expectations \
             and limit liability."
                .into(),
        ),
    }
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
    let mut report = Report { failures: vec![] };

    macro_rules! turn {
        ($sid:expr, $msg:expr) => {
            run_turn(memory.clone(), $sid.to_string(), $msg.to_string(), None).await
        };
        ($sid:expr, $msg:expr, $pin:expr) => {
            run_turn(memory.clone(), $sid.to_string(), $msg.to_string(), Some($pin)).await
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
    let s = session("pin");
    turn!(s, "find me discussions about career");
    turn!(s, "yes, the second one"); // decoy: a thread just read in detail
    let r = turn!(s, "okay, can you explain about this thread now?", pinned());
    let low = r.reply.to_lowercase();
    let on_pin = ["flight", "atc", "airport", "transparen", "pr statement", "withhold"]
        .iter()
        .any(|w| low.contains(w));
    let on_decoy = ["techie to management", "career mobility"].iter().any(|w| low.contains(w));
    report.check("answers about the PINNED thread", on_pin, &r.reply);
    report.check("not the decoy", !on_decoy, &r.reply);
    report.check("no tool call needed", tools(&r).is_empty(), &format!("{:?}", tools(&r)));

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
