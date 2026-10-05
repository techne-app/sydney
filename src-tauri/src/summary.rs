//! Reading a Hacker News thread, and summarising one.
//!
//! The corpus already carries a summary for every thread, but it was written
//! once when the pipeline first analysed the thread and never revised. Measured
//! on a live card: the stored summary had seen 2 comments; the thread had 151.
//! It described 2% of the conversation. So anything the user asks for has to
//! come from the comments as they stand now.
//!
//! The raw comments come from techne.app rather than HN's own API: the backend
//! holds every comment already, within about six minutes of live HN, so this is
//! one request instead of the ~600 that walking HN's tree would take.
//!
//! Reading a thread and summarising it are kept separate, because they cost
//! wildly different amounts — about a second against about twenty. The agent
//! needs the reading so it can answer what a summary cannot ("did anyone mention
//! performance?"), and should not pay for a summary to get it.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use rig_core::agent::AgentBuilder;
use rig_core::client::CompletionClient;
use rig_core::completion::Prompt;
use serde::Deserialize;

const THREAD_TREE_URL: &str = "https://techne.app/api/thread-tree";

/// Closes every comment block. Clipping splits on this, so a thread is always
/// cut between comments and never through the middle of one.
const END_MARKER: &str = "---END-COMMENT---\n";

/// How much thread text reaches the model, in characters.
///
/// Characters rather than tokens because counting tokens needs the tokenizer,
/// and the budget has to be known before the model is involved. Measured on
/// real HN threads, our formatting runs at **~3.5 characters per token** —
/// consistently, across budgets from 8k to 30k.
///
/// 14,000 chars is about 4,000 tokens, and it serves both callers. That it fits
/// the agent too was worth measuring (`--bin context_budget`) rather than
/// assuming: the agent's instruction and both tool schemas come to only 316
/// tokens, and `max_tokens(2048)` is the real fixed cost, which leaves ~20,000
/// characters free in a fresh chat.
///
/// The number is chosen from where threads actually divide rather than from the
/// window alone. Comments arrive in depth order, and across a sample of live
/// cards 14,000 characters covered the first three reply levels in full for most
/// of them, and the entire thread for the smaller ones. Only genuinely busy
/// threads get cut, and they get cut inside depth 2 — still the main exchange,
/// not the deep sub-arguments.
///
/// It is also what the wait costs: prefill runs at ~230 tok/s and dominates
/// (generation is ~50 tok/s but only ~400 tokens), so this is roughly 20s on a
/// big thread and ~11s on a small one.
///
/// What it does *not* leave room for is the agent reading two threads in one
/// conversation: a thread it has read stays in the history and is re-sent every
/// turn, so a second read overflows the window where fifty exchanges would not.
///
/// `n_ctx` is 8,192 and cannot currently go higher: rig-llama-cpp fails above
/// it on this model (Decode Error -3 at 16,384), while llama-server runs the
/// same GGUF at 65,536. If that is fixed, raise this with it — at 32,768 a
/// whole thread would fit.
const MAX_THREAD_CHARS: usize = 14_000;

/// One comment, as the endpoint returns it. Rows arrive in depth order.
#[derive(Debug, Deserialize)]
pub struct Comment {
    pub depth: u32,
    pub by: Option<String>,
    pub text: Option<String>,
}

/// A thread as the model sees it: every comment, formatted, nothing dropped.
///
/// Held whole rather than pre-clipped because the budget belongs to the caller,
/// not to the thread. Clipping is taking whole blocks off the front, which costs
/// microseconds, so holding all of it costs one fetch and ~200KB and leaves the
/// budget free to change.
#[derive(Clone)]
pub struct Thread {
    text: String,
    /// Comments the thread has, including the ones a clip will drop.
    pub total: usize,
}

impl Thread {
    /// The first whole comments that fit in `budget`, and how many that was.
    pub fn clip(&self, budget: usize) -> (String, usize) {
        let mut out = String::new();
        let mut used = 0;

        for block in self.text.split_inclusive(END_MARKER) {
            if block.trim().is_empty() {
                continue;
            }
            // Skip a comment that will not fit rather than stopping here: one
            // long comment early in the thread would otherwise discard every
            // shorter one behind it. Measured on a live thread, stopping wasted
            // 18% of the budget. Later comments are deeper, so this trades
            // strict depth order for using the window we have.
            if out.len() + block.len() > budget {
                continue;
            }
            out.push_str(block);
            used += 1;
        }
        (out, used)
    }
}

/// Flatten the tree for the model.
///
/// Usernames become stable anonymous ids — U1, U2 — so the model can follow who
/// is replying to whom without real handles steering it toward a person's
/// reputation rather than their argument.
///
/// Comments are kept in the order given, which is depth order, so a clip drops
/// the deep nested tangents and keeps the shallow spine of the conversation.
/// That ordering is the endpoint's contract, not an accident.
fn format_thread(comments: &[Comment]) -> String {
    let mut users: Vec<&str> = Vec::new();
    let mut out = String::new();

    for comment in comments {
        let Some(text) = comment.text.as_deref().filter(|t| !t.is_empty()) else {
            continue;
        };
        let name = comment.by.as_deref().unwrap_or("");
        let uid = match users.iter().position(|u| *u == name) {
            Some(i) => i + 1,
            None => {
                users.push(name);
                users.len()
            }
        };

        out.push_str(&format!(
            "<<depth={}>> [[uid=U{uid}]]\n{text}\n{END_MARKER}",
            comment.depth
        ));
    }
    out
}

/// Fetch one thread and format it. About a second; the model is not involved.
async fn read_thread(thread_id: u64) -> Result<Thread, String> {
    let started = std::time::Instant::now();
    let comments: Vec<Comment> = reqwest::Client::new()
        .get(format!("{THREAD_TREE_URL}/{thread_id}/"))
        .timeout(std::time::Duration::from_secs(60))
        .send()
        .await
        .map_err(|e| format!("could not reach the thread service: {e}"))?
        .error_for_status()
        .map_err(|e| format!("thread service returned {e}"))?
        .json()
        .await
        .map_err(|e| format!("could not read the thread: {e}"))?;

    let total = comments.len();
    let text = format_thread(&comments);
    if text.is_empty() {
        return Err("That discussion has no readable comments yet.".into());
    }
    println!(
        "[thread] {thread_id}: {total} comments, {} chars (fetch {:.1}s)",
        text.len(),
        started.elapsed().as_secs_f32()
    );
    Ok(Thread { text, total })
}

const INSTRUCTION: &str = concat!(
    "You write the blurb that tells someone whether a Hacker News discussion is ",
    "worth their time. Three parts, in this order, separated by blank lines.\n\n",
    "1. ONE sentence on what people are doing with the subject — arguing a ",
    "point, comparing their own experience, explaining how something works, ",
    "remembering someone — and what is at stake in it. The reader has already ",
    "seen the title and the topic directly above this, so do not restate either: ",
    "say the thing they do not know yet.\n\n",
    "2. Two or three quotations, each on its own line, wrapped in double quotes. ",
    "Copy them WORD FOR WORD from the comments — do not tidy, shorten, join or ",
    "reword them, and never invent one. Pick the lines that are concrete, ",
    "surprising or sharply put. Where the thread divides, take them from ",
    "different sides of it. A quotation with a real number or a vivid comparison ",
    "in it beats a general one.\n\n",
    "3. Two sentences saying what the thread amounts to, and naming anything ",
    "substantial the quotations did not cover. Where people genuinely disagree, ",
    "say where the disagreement lies. Where they mostly agree, or are trading ",
    "experience rather than arguing, say that instead — never invent a ",
    "disagreement to give the thread a shape.\n\n",
    "No headings, no numbering, no bullets, no preamble — just the three parts. ",
    "Write about the ideas, never the depth or author markers. Use only what is ",
    "in the comments."
);

/// Flatten text so a quotation can be compared with its source.
///
/// Comments arrive as HTML — `<p>`, `&#x27;`, `&gt;` — and the model quotes what
/// it reads, not what it was sent, so it writes `it's` where the source holds
/// `it&#x27;s`. Curly quotes, inner quote marks and line wrapping differ too.
/// None of that is the model inventing anything, so none of it should fail a
/// check whose job is to catch invention.
fn flatten(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_tag = false;
    for ch in text.chars() {
        match ch {
            '<' => in_tag = true,
            '>' if in_tag => in_tag = false,
            _ if in_tag => {}
            // Curly quotes and apostrophes become their straight equivalents,
            // and an inner double quote becomes a single one: a quotation is
            // wrapped in double quotes, so the model correctly switches the ones
            // inside it.
            '\u{2018}' | '\u{2019}' | '\u{201c}' | '\u{201d}' | '"' => out.push('\''),
            c if c.is_whitespace() => {
                if !out.ends_with(' ') {
                    out.push(' ');
                }
            }
            c => out.extend(c.to_lowercase()),
        }
    }
    out.replace("&#x27;", "'")
        .replace("&quot;", "'")
        .replace("&#x2f;", "/")
        .replace("&gt;", ">")
        .replace("&lt;", "<")
        .replace("&amp;", "&")
        .trim()
        .to_string()
}

/// Drop any quotation that is not in the thread, word for word.
///
/// This is the reason the summary quotes rather than paraphrases. A paraphrase
/// can only be asked not to invent; a quotation can be checked. Measured on this
/// very thread, the agent turned a comment saying 13 million gallons "is less
/// than half of what the City of Lincoln reports using on Sept. 29" into
/// "Lincoln uses 30M to 60M gallons per day" — a fabricated range sitting
/// indistinguishably among eleven real figures.
///
/// The match has to be **contiguous**. An honest trim starts mid-sentence and
/// capitalises, which flattening forgives; stitching two distant lines into one
/// quotation is exactly what it must not forgive, and contiguity is the
/// difference between the two.
fn drop_unverifiable_quotes(summary: &str, source: &str) -> String {
    let haystack = flatten(source);
    let mut kept = Vec::new();

    for line in summary.lines() {
        let trimmed = line.trim();
        let is_quote = trimmed.len() > 2
            && trimmed.starts_with(['"', '\u{201c}'])
            && trimmed.ends_with(['"', '\u{201d}']);

        if is_quote {
            let inner: String = trimmed.chars().skip(1).take(trimmed.chars().count() - 2).collect();
            // Terminal punctuation is ignored. A quotation that stops early gets
            // a full stop where the source had a comma or a question mark, which
            // is how anyone trims a quotation and is not what this is looking
            // for. Every word still has to match, contiguously.
            let needle = flatten(&inner);
            let needle = needle.trim_end_matches(['.', ',', ';', ':', '!', '?', '\'', ' ']);
            if needle.is_empty() || !haystack.contains(needle) {
                println!("[summary] dropped an unverifiable quote: {}", &inner.chars().take(200).collect::<String>());
                continue;
            }
        }
        kept.push(trimmed.to_string());
    }

    kept.join("\n")
}

pub struct ThreadSummary {
    pub summary: String,
    /// Comments actually read, and how many the thread has. These differ on a
    /// busy thread, and the gap is the honest bound on everything downstream —
    /// the UI says so rather than implying the summary covers the whole thing.
    pub used: usize,
    pub total: usize,
    /// Characters actually handed to the model. Only the checks look at this —
    /// it is how they tell a full budget from a wasted one.
    pub chars: usize,
}

/// What one thread costs, paid at most once.
#[derive(Default)]
struct Entry {
    thread: Option<Thread>,
    summary: Option<String>,
}

/// Every thread anyone has looked at this session.
///
/// This exists because the modal and the chat are otherwise strangers. The
/// modal's cache used to be a `Map` in the webview, which the Rust agent cannot
/// see, so asking the chat about a thread just read in the modal would re-do —
/// or contradict — work already done. One store in Rust is what makes the two
/// surfaces agree.
///
/// In memory only, so it empties on quit. Deliberate: a thread grows, and a
/// summary kept across restarts would recreate the staleness this exists to fix.
/// Nothing moves in the minutes a session lasts.
pub struct ThreadStore {
    /// The outer lock guards the map and is never held across an await. The
    /// inner one guards a single thread's work and is — which is what makes two
    /// callers arriving at once wait on one fetch rather than racing to do it
    /// twice. Worth the care: the summary they would race on costs 20 seconds.
    entries: Mutex<HashMap<u64, Arc<tokio::sync::Mutex<Entry>>>>,
}

impl ThreadStore {
    pub fn new() -> Self {
        Self { entries: Mutex::new(HashMap::new()) }
    }

    fn entry(&self, thread_id: u64) -> Arc<tokio::sync::Mutex<Entry>> {
        self.entries.lock().unwrap().entry(thread_id).or_default().clone()
    }

    /// The thread's comments, clipped to what the model can hold, with the count
    /// of what was left out. Fetched the first time, free afterwards.
    pub async fn comments(&self, thread_id: u64) -> Result<(String, usize, usize), String> {
        let entry = self.entry(thread_id);
        let mut entry = entry.lock().await;

        if entry.thread.is_none() {
            entry.thread = Some(read_thread(thread_id).await?);
        }
        let thread = entry.thread.as_ref().expect("just filled in");
        let (text, used) = thread.clip(MAX_THREAD_CHARS);
        Ok((text, used, thread.total))
    }

    /// A summary written from the thread's current comments. Twenty seconds the
    /// first time, free afterwards — including when the other surface asks.
    pub async fn summary(
        &self,
        model: &Arc<rig_llama_cpp::Client>,
        thread_id: u64,
    ) -> Result<ThreadSummary, String> {
        let entry = self.entry(thread_id);
        let mut entry = entry.lock().await;

        if entry.thread.is_none() {
            entry.thread = Some(read_thread(thread_id).await?);
        }
        let thread = entry.thread.as_ref().expect("just filled in");
        let total = thread.total;
        let (text, used) = thread.clip(MAX_THREAD_CHARS);

        if let Some(summary) = &entry.summary {
            println!("[summary] thread {thread_id}: already written this session");
            return Ok(ThreadSummary { summary: summary.clone(), used, total, chars: text.len() });
        }

        let text_len = text.len();
        println!("[summary] thread {thread_id}: {used}/{total} comments, {text_len} chars");
        let started = std::time::Instant::now();

        let agent = AgentBuilder::new(model.completion_model("gemma"))
            .preamble(INSTRUCTION)
            // Low, unlike the pipeline's 1.0: this is triggered by a click, and
            // clicking the same card twice should not produce two different
            // summaries of the same conversation.
            .temperature(0.1)
            // Room for one or two paragraphs. Measured: summaries land around
            // 400 tokens and generation runs at ~50 tok/s, so this caps waste
            // without truncating. It is not where the time goes — prefill is
            // ~3x the cost of generation here.
            .max_tokens(500)
            .additional_params(serde_json::json!({ "thinking": false }))
            .build();

        let source = text.clone();
        let summary = agent
            .prompt(text)
            .await
            .map_err(|e| format!("the model could not summarise that thread: {e}"))?
            .trim()
            .to_string();

        let summary = drop_unverifiable_quotes(&summary, &source);
        println!("[summary] model took {:.1}s", started.elapsed().as_secs_f32());
        entry.summary = Some(summary.clone());

        Ok(ThreadSummary { summary, used, total, chars: text_len })
    }
}

/// What the frontend calls when someone clicks a thread card. Not an agent
/// turn — there is nothing for a model to decide, so it is one completion.
#[tauri::command]
pub async fn summarize_thread_command(
    model: tauri::State<'_, crate::agent::Model>,
    store: tauri::State<'_, Arc<ThreadStore>>,
    thread_id: u64,
) -> Result<serde_json::Value, String> {
    let Some(model) = model.0.get() else {
        return Err("The model is still loading.".into());
    };
    let result = store.summary(&model, thread_id).await?;
    Ok(serde_json::json!({
        "summary": result.summary,
        "used": result.used,
        "total": result.total,
    }))
}

#[cfg(test)]
mod tests {
    use super::drop_unverifiable_quotes;

    /// Two real comments, as the endpoint sends them: HTML entities, curly
    /// apostrophes, and an inner quotation in double quotes.
    const SOURCE: &str = "<<depth=0>> [[uid=U1]]\n\
        <p>I do this work too, but for a different reason: in my area they make as \
        much or more than mid level software engineers, so when it&#x27;s about \
        adding a breaker box they want 8k.\n---END-COMMENT---\n\
        <<depth=1>> [[uid=U2]]\n\
        <p>Why aren\u{2019}t the data centres just honest about their water usage if \
        it\u{2019}s seemingly so low? Perhaps it is not a meaningful amount.\n---END-COMMENT---\n";

    fn kept(summary: &str) -> usize {
        drop_unverifiable_quotes(summary, SOURCE)
            .lines()
            .filter(|l| l.starts_with('"'))
            .count()
    }

    #[test]
    fn keeps_a_quote_that_is_really_there() {
        assert_eq!(kept("\"adding a breaker box they want 8k.\""), 1);
    }

    #[test]
    fn keeps_a_trim_that_changes_the_last_mark() {
        // The source has a comma here and a question mark there; ending a
        // shortened quotation with a full stop is ordinary trimming, and both of
        // these were rejected by the first version of this check.
        assert_eq!(kept("\"in my area they make as much or more than mid level software engineers.\""), 1);
        assert_eq!(
            kept("\"Why aren't the data centres just honest about their water usage if it's seemingly so low.\""),
            1
        );
    }

    #[test]
    fn keeps_a_quote_through_html_and_curly_marks() {
        // `it&#x27;s` in the source, `it's` in the quotation.
        assert_eq!(kept("\"so when it's about adding a breaker box\""), 1);
    }

    #[test]
    fn drops_an_invented_quote() {
        assert_eq!(kept("\"Lincoln uses 30M to 60M gallons per day.\""), 0);
    }

    #[test]
    fn drops_two_distant_fragments_stitched_together() {
        // Both halves are in the source; they are not next to each other. This
        // is the failure contiguity exists to catch, and the one a reader could
        // never spot.
        assert_eq!(
            kept("\"in my area they make as much or more than mid level software engineers and the data centres are not honest about their water usage.\""),
            0
        );
    }

    #[test]
    fn leaves_the_prose_alone() {
        let summary = "Users are arguing about pay.\n\"not a real quote at all\"\nThe thread amounts to little.";
        let out = drop_unverifiable_quotes(summary, SOURCE);
        assert!(out.contains("Users are arguing about pay."));
        assert!(out.contains("The thread amounts to little."));
        assert!(!out.contains("not a real quote"));
    }
}
