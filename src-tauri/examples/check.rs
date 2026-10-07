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
//! Loads Gemma in-process, so close the app first — 24GB cannot hold two
//! copies of a 12GB model, and the symptom is empty replies, not an error. Slow:
//! each
//! check is one or more real turns through a 12GB model.

use std::sync::Arc;

use app_lib::agent::{run_turn, AgentReply, PinnedThread};
use app_lib::corpus::Corpus;
use app_lib::memory::SqliteConversationMemory;
use rig_core::memory::ConversationMemory;
use rig_core::message::{Message, ToolResultContent, UserContent};
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
/// The `count` busiest threads the sidebar is offering.
///
/// Busiest, not simply the top cards: the feed ranks by karma density, and its
/// second card came back with three comments and 922 characters — nowhere near
/// the 14,000-character budget, so the window check it was meant to exercise
/// passed without ever being tested. A thread has to be big to be interesting
/// here.
async fn pinned_cards(count: usize) -> Vec<PinnedThread> {
    let mut cards: serde_json::Value = reqwest::Client::new()
        .post("https://www.techne.app/api/thread-cards/")
        .json(&serde_json::json!({
            "num_cards": 20, "hours_back": 24,
            "sort_by": "karma_density", "density_min_comment_constant": 100
        }))
        .send()
        .await
        .expect("fetch sidebar cards")
        .json()
        .await
        .expect("parse the cards");
    let list = cards.as_array_mut().expect("cards is a list");
    list.sort_by_key(|c| std::cmp::Reverse(c["comment_count"].as_u64().unwrap_or(0)));
    list.iter()
        .take(count)
        .map(|card| {
            println!(
                "    card {} — {} comments",
                card["id"], card["comment_count"]
            );
            PinnedThread {
                id: card["id"].as_u64().expect("card has an id"),
                story_title: card["story_title"].as_str().map(str::to_string),
                theme: card["theme"].as_str().map(str::to_string),
            }
        })
        .collect()
}

async fn pinned() -> PinnedThread {
    pinned_cards(1).await.into_iter().next().expect("at least one card")
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

    // The case that was missing, and so the bug that got through: nothing here
    // had ever read two threads in one conversation. A thread read with
    // get_thread is ~14,000 characters and the whole history is re-sent every
    // turn, so the second read used to blow the window outright —
    // `Prompt 11193 tokens exceeds n_ctx 8192` — on a completely ordinary
    // sequence of questions.
    println!("\n=== 8. two threads in one conversation  <- the window");
    let s = session("two-threads");
    turn!(s, "find me discussions about rust programming");
    let first = turn!(s, "tell me about the first one");
    let second = turn!(s, "now tell me about the second one");
    let follow_up = turn!(s, "so what did people disagree about in that one?");

    let read: Vec<u64> = [&first, &second].iter().flat_map(|r| thread_ids(r)).collect();
    report.check(
        "reads two different threads",
        read.len() == 2 && read[0] != read[1],
        &format!("{read:?}"),
    );
    for (name, reply) in [("second read", &second), ("the follow-up", &follow_up)] {
        report.check(
            &format!("{name} does not overflow the window"),
            reply.error.is_none(),
            reply.error.as_deref().unwrap_or("no error"),
        );
    }
    report.check("still answering", follow_up.reply.len() > 20, &follow_up.reply);

    // The assertions above are a smoke test and will pass on small threads
    // whatever the history does — which is exactly why the bug got through. This
    // one measures the mechanism instead: whatever size the threads happen to
    // be, only the most recently read one may still carry its comments.
    let history = memory.load(&s).await.expect("load the conversation back");
    let mut carrying = 0;
    let mut hidden = 0;
    for message in &history {
        let Message::User { content } = message else { continue };
        for item in content.iter() {
            let UserContent::ToolResult(result) = item else { continue };
            for part in result.content.iter() {
                let ToolResultContent::Text(text) = part else { continue };
                let Ok(body) = serde_json::from_str::<serde_json::Value>(&text.text) else {
                    continue;
                };
                match body.get("comments").and_then(|c| c.as_str()) {
                    Some(c) if c.starts_with("(not shown") => hidden += 1,
                    Some(_) => carrying += 1,
                    None => {}
                }
            }
        }
    }
    report.check(
        "only the newest thread keeps its comments",
        carrying == 1 && hidden >= 1,
        &format!("{carrying} carrying, {hidden} hidden"),
    );

    // The sequence that actually broke, which case 8 does not reach: two cards
    // from the sidebar, pinned one after the other. Search results are small —
    // three comments, 500 characters — and never approach the window. Sidebar
    // cards are ranked by karma density and run to hundreds of comments, so each
    // read is a full 14,000-character budget and the second one is what used to
    // fail with `Prompt 10711 tokens exceeds n_ctx 8192`.
    println!("\n=== 9. two big pinned threads  <- #58");
    let cards = pinned_cards(2).await;
    if cards.len() < 2 {
        report.check("two sidebar cards available", false, "the feed returned fewer than two");
    } else {
        let s = session("two-pins");
        println!("    pinning {} then {}", cards[0].id, cards[1].id);
        // Questions that cannot be answered from the attachment's title and
        // theme, so the model has to read. Asked "what is this thread about?" it
        // answers from the title and never calls the tool — and then only one
        // thread is ever in the window, which is how this case passed twice
        // while the bug it is named after was still live.
        let one = turn!(s, "what specific figures did people give in this thread?", cards[0].clone());
        let two = turn!(s, "and what figures are in this one?", cards[1].clone());

        // Checked first and deliberately harshly: if the model did not read both
        // threads, nothing below is evidence of anything, and a quiet pass would
        // be worse than a failure.
        report.check(
            "actually reads both threads (otherwise this case proves nothing)",
            thread_ids(&one).contains(&cards[0].id) && thread_ids(&two).contains(&cards[1].id),
            &format!("first read {:?}, second read {:?}", thread_ids(&one), thread_ids(&two)),
        );

        for (name, reply) in [("first pin", &one), ("second pin", &two)] {
            report.check(
                &format!("{name} does not overflow the window"),
                reply.error.is_none(),
                reply.error.as_deref().unwrap_or("no error"),
            );
        }
    }

    println!("\n{}", "=".repeat(46));
    if report.failures.is_empty() {
        println!("ALL PASSED");
    } else {
        println!("{} FAILED: {}", report.failures.len(), report.failures.join(", "));
        std::process::exit(1);
    }
}
