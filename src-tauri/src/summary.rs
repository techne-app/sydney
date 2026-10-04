//! On-demand thread summaries, generated locally.
//!
//! The corpus already carries a summary for every thread, but it was written
//! once when the pipeline first analysed the thread and never revised. Measured
//! on a live card: the stored summary had seen 2 comments; the thread had 151.
//! It described 2% of the conversation. So a summary the user asks for has to be
//! generated from the comments as they stand now.
//!
//! The raw comments come from techne.app rather than HN's own API: the backend
//! holds every comment already, within about six minutes of live HN, so this is
//! one request instead of the ~600 that walking HN's tree would take.

use std::sync::Arc;

use rig_core::agent::AgentBuilder;
use rig_core::client::CompletionClient;
use rig_core::completion::Prompt;
use serde::Deserialize;

const THREAD_TREE_URL: &str = "https://techne.app/api/thread-tree";

/// How much thread text reaches the model, in characters.
///
/// Characters rather than tokens because counting tokens needs the tokenizer,
/// and the budget has to be known before the model is involved. Measured on
/// real HN threads, our formatting runs at **~3.5 characters per token** —
/// consistently, across budgets from 8k to 30k.
///
/// 14,000 chars is about 4,000 tokens — comfortably inside the 8,192 window
/// alongside the instruction and the answer.
///
/// The number is chosen from where threads actually divide rather than from the
/// window alone. Comments arrive in depth order, and across a sample of live
/// cards 14,000 characters covered the first three reply levels in full for most
/// of them, and the entire thread for the smaller ones. Only genuinely busy
/// threads get cut, and they get cut inside depth 2 — still the main exchange,
/// not the deep sub-arguments.
///
/// It is also what the wait costs: prefill runs at ~230 tok/s and dominates
/// (generation is ~50 tok/s but only ~400 tokens), so this is roughly 25s on a
/// big thread against 32s at 20,000, and ~12s on a small one either way.
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

/// Flatten the tree for the model.
///
/// Usernames become stable anonymous ids — U1, U2 — so the model can follow who
/// is replying to whom without real handles steering it toward a person's
/// reputation rather than their argument.
///
/// Comments are kept in the order given, which is depth order, so truncation
/// drops the deep nested tangents and keeps the shallow spine of the
/// conversation. That ordering is the endpoint's contract, not an accident.
fn format_thread(comments: &[Comment]) -> (String, usize) {
    let mut users: Vec<&str> = Vec::new();
    let mut out = String::new();
    let mut used = 0;

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

        let block = format!("<<depth={}>> [[uid=U{uid}]]\n{text}\n---END-COMMENT---\n", comment.depth);
        // Skip a comment that will not fit rather than stopping here: one long
        // comment early in the thread would otherwise discard every shorter one
        // behind it. Measured on a live thread, stopping wasted 18% of the
        // budget. Later comments are deeper, so this trades strict depth order
        // for using the window we have.
        if out.len() + block.len() > MAX_THREAD_CHARS {
            continue;
        }
        out.push_str(&block);
        used += 1;
    }
    (out, used)
}

const INSTRUCTION: &str = concat!(
    "You summarise Hacker News discussions for someone deciding whether to read ",
    "the whole thing. Write at most 130 words, one or two paragraphs of plain ",
    "prose — no headings, no bullets, no preamble.\n\n",
    "Say what people actually argued: the main positions, where they disagreed, ",
    "and any concrete experience or surprising claim. Where opinion is divided, ",
    "say so rather than inventing a consensus.\n\n",
    "Comments carry depth and anonymised author markers. Write about the ideas, ",
    "never the markers. Use only what is in the comments."
);

pub struct ThreadSummary {
    pub summary: String,
    /// How many comments were actually fed to the model, and how many the thread
    /// has. A long thread is truncated, and the UI should be able to say so
    /// rather than implying the summary covers everything.
    pub used: usize,
    pub total: usize,
}

pub async fn summarize_thread(
    model: &Arc<rig_llama_cpp::Client>,
    thread_id: u64,
) -> Result<ThreadSummary, String> {
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

    let fetched = started.elapsed();
    let total = comments.len();
    let (text, used) = format_thread(&comments);
    if text.is_empty() {
        return Err("That discussion has no readable comments yet.".into());
    }
    println!(
        "[summary] thread {thread_id}: {used}/{total} comments, {} chars (fetch {:.1}s)",
        text.len(),
        fetched.as_secs_f32()
    );
    let before_model = std::time::Instant::now();

    let agent = AgentBuilder::new(model.completion_model("gemma"))
        .preamble(INSTRUCTION)
        // Low, unlike the pipeline's 1.0: this is triggered by a click, and
        // clicking the same card twice should not produce two different
        // summaries of the same conversation.
        .temperature(0.1)
        // Room for one or two paragraphs. Measured: summaries land around 400
        // tokens and generation runs at ~50 tok/s, so this caps waste without
        // truncating. It is not where the time goes — prefill is ~3x the cost
        // of generation here.
        .max_tokens(500)
        .additional_params(serde_json::json!({ "thinking": false }))
        .build();

    let summary = agent
        .prompt(text)
        .await
        .map_err(|e| format!("the model could not summarise that thread: {e}"))?;

    println!("[summary] model took {:.1}s", before_model.elapsed().as_secs_f32());

    Ok(ThreadSummary {
        summary: summary.trim().to_string(),
        used,
        total,
    })
}

/// What the frontend calls when someone clicks a thread card. Not an agent
/// turn — there is nothing for a model to decide, so it is one completion.
#[tauri::command]
pub async fn summarize_thread_command(
    model: tauri::State<'_, crate::agent::Model>,
    thread_id: u64,
) -> Result<serde_json::Value, String> {
    let Some(model) = model.0.get() else {
        return Err("The model is still loading.".into());
    };
    let result = summarize_thread(&model, thread_id).await?;
    Ok(serde_json::json!({
        "summary": result.summary,
        "used": result.used,
        "total": result.total,
    }))
}
