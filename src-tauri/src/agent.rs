//! The agent, in Tauri core.
//!
//! Replaces `sidecar/agent.py`. The model still runs in llama-server on :8081 —
//! only the agent framework moved, ADK to Rig. The corpus and search still live
//! in the Python sidecar, so the tools below reach them over HTTP the way the
//! frontend already does; #48 removes that hop.
//!
//! The problem the agent solves is unchanged: the model calls a tool, sees the
//! result, and decides again. Routing used to give it one tool call per turn and
//! then stop, so when a question needed something the model didn't have it
//! invented it instead. Every hallucination this project hit came from that gap.

use std::sync::{Arc, Mutex};

use rig_agent::agent::AgentBuilder;
use rig_agent::prelude::*;
use rig_agent::tool::{Tool, ToolContext};
use rig_core::providers::llamafile;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Where the Python sidecar still serves the corpus. Removed by #48.
const SIDECAR_URL: &str = "http://localhost:8000";
/// The inference server. Replaced by in-process llama-cpp-2 in #47.
const LLAMA_SERVER_URL: &str = "http://localhost:8081";

/// Carried over verbatim from agent.py. The "or links" clause is load-bearing:
/// without it the model builds `item?id={thread_id}` out of the id it was given
/// and writes it into its prose, and the UI then renders the same link a second
/// time. Observed while spiking this port.
const BASE_INSTRUCTION: &str = concat!(
    "You help a user explore Hacker News. Use search_threads to find ",
    "discussions on a topic, and get_thread to read one in detail when the ",
    "user asks about a specific result. Answer from what the tools return — do ",
    "not invent discussions, titles, or links. If you do not have enough ",
    "information to answer, say so plainly rather than guessing. Do not show ",
    "thread ids to the user; they are for your own tool calls.",
    "\n\nA message may begin with a note about a discussion the user has open ",
    "in front of them. When they say \"this thread\", \"this discussion\", ",
    "\"this\", or \"this one\", they mean that one — not an item from a list ",
    "of search results. Answer from the note; there is no need to search."
);

/// The thread the user has dragged into the chat, as the frontend sends it.
#[derive(Debug, Clone, Deserialize)]
pub struct PinnedThread {
    pub story_title: Option<String>,
    pub theme: Option<String>,
    pub summary: Option<String>,
}

/// The pinned thread travels WITH the message it was sent alongside, rather
/// than living in agent state.
///
/// State was the wrong home for it, and failed in a way worth recording. State
/// feeds the system prompt, which has no position in time — as far as the model
/// can tell it has always said this — while a thread just read with get_thread
/// sits in the newest message. Asked to explain "this thread", the model picked
/// the recent one. No instruction wording outranked it.
fn with_attachment(message: &str, pinned: Option<&PinnedThread>) -> String {
    match pinned {
        None => message.to_string(),
        Some(t) => format!(
            "[The user has this discussion open in front of them:\n\"{}\" — {}.\nWhat people said: {}]\n\n{}",
            // It's t.story_title or ""
            t.story_title.as_deref().unwrap_or(""),
            t.theme.as_deref().unwrap_or(""),
            t.summary.as_deref().unwrap_or(""),
            message
        ),
    }
}

/// One thread as the UI renders it. `link` never reaches the model — see the
/// note on GetThread.
#[derive(Debug, Clone, Serialize)]
pub struct UiResult {
    pub thread_id: u64,
    pub title: String,
    pub link: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolCallRecord {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// What the tools report back for the UI, filled in as the agent loop runs.
///
/// The frontend renders links from `results` rather than from the model's prose:
/// the model retypes a 50-character URL every time it writes a list, and one
/// wrong character is a dead link nobody notices until it's clicked.
#[derive(Default)]
struct Collected {
    tool_calls: Vec<ToolCallRecord>,
    results: Vec<UiResult>,
}

type Shared = Arc<Mutex<Collected>>;

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ToolFailed(String);

// ---------------------------------------------------------------------------
// search_threads
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct SearchArgs {
    keyword_filter: String,
}

struct SearchThreads {
    http: reqwest::Client,
    collected: Shared,
}

impl Tool for SearchThreads {
    const NAME: &'static str = "search_threads";
    type Args = SearchArgs;
    type Output = serde_json::Value;
    type Error = ToolFailed;

    fn description(&self) -> String {
        "Search Hacker News discussion threads by topic.".into()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "keyword_filter": {
                    "type": "string",
                    "description": "The topic or keywords to search discussions for."
                }
            },
            "required": ["keyword_filter"]
        })
    }

    async fn call(
        &self,
        _ctx: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.collected.lock().unwrap().tool_calls.push(ToolCallRecord {
            name: Self::NAME.into(),
            arguments: json!({ "keyword_filter": args.keyword_filter }),
        });

        let body = self
            .http
            .post(format!("{SIDECAR_URL}/search"))
            .json(&json!({ "query": args.keyword_filter, "limit": 3 }))
            .send()
            .await
            .map_err(|e| ToolFailed(e.to_string()))?
            .json::<serde_json::Value>()
            .await
            .map_err(|e| ToolFailed(e.to_string()))?;

        // Search lists options; get_thread is how you explore one. A hit carries
        // neither the summary nor a link — that is what stops the UI appending a
        // second copy of the list the model just wrote.
        Ok(body)
    }
}

// ---------------------------------------------------------------------------
// get_thread
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ThreadArgs {
    thread_id: u64,
}

struct GetThread {
    http: reqwest::Client,
    collected: Shared,
}

impl Tool for GetThread {
    const NAME: &'static str = "get_thread";
    type Args = ThreadArgs;
    type Output = serde_json::Value;
    type Error = ToolFailed;

    fn description(&self) -> String {
        "Get the full summary of one Hacker News discussion thread. Use this when \
         the user asks about a specific discussion you have already found — for \
         example \"what was the second one about?\". Search results carry only a \
         title and a one-line description; this returns what was actually discussed."
            .into()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "thread_id": {
                    "type": "integer",
                    "description": "The id of the thread, taken from an earlier search result."
                }
            },
            "required": ["thread_id"]
        })
    }

    async fn call(
        &self,
        _ctx: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.collected.lock().unwrap().tool_calls.push(ToolCallRecord {
            name: Self::NAME.into(),
            arguments: json!({ "thread_id": args.thread_id }),
        });

        let mut body = self
            .http
            .get(format!("{SIDECAR_URL}/thread/{}", args.thread_id))
            .send()
            .await
            .map_err(|e| ToolFailed(e.to_string()))?
            .json::<serde_json::Value>()
            .await
            .map_err(|e| ToolFailed(e.to_string()))?;

        // The link is split off here and never handed to the model. Given a URL
        // it writes it into its prose, and the UI renders the same link again
        // from `results`. Taking it out is what makes that impossible rather
        // than merely discouraged.
        if let Some(link) = body.get("link").and_then(|v| v.as_str()).map(str::to_string) {
            let title = body
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            self.collected.lock().unwrap().results.push(UiResult {
                thread_id: args.thread_id,
                title,
                link,
            });
        }
        if let Some(obj) = body.as_object_mut() {
            obj.remove("link");
        }

        Ok(body)
    }
}

// ---------------------------------------------------------------------------
// The command the frontend calls
// ---------------------------------------------------------------------------

/// The durable store Rig loads from and appends to. Built once at startup.
pub struct AgentMemory(pub Arc<crate::memory::SqliteConversationMemory>);

#[derive(Debug, Default, Serialize)]
pub struct AgentReply {
    pub reply: String,
    pub tool_calls: Vec<ToolCallRecord>,
    pub results: Vec<UiResult>,
    /// Failures come back as a normal reply with this set, not as a rejected
    /// promise: the frontend already renders `response.error` inline, and a
    /// throw would bypass that and surface as an unhandled rejection instead.
    pub error: Option<String>,
}

impl AgentReply {
    fn failed(message: impl Into<String>) -> Self {
        Self {
            error: Some(message.into()),
            ..Default::default()
        }
    }
}

/// Gemma takes a minute or two to load 12GB, and until it has, every request
/// fails with a bare connection error. Saying so plainly is worth a round trip:
/// a not-yet-ready server has twice sent us chasing bugs that did not exist.
async fn llama_server_ready(http: &reqwest::Client) -> bool {
    matches!(
        http.get(format!("{LLAMA_SERVER_URL}/health"))
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await,
        Ok(response) if response.status().is_success()
    )
}

#[tauri::command]
pub async fn send_message(
    memory: tauri::State<'_, AgentMemory>,
    session_id: String,
    message: String,
    pinned_thread: Option<PinnedThread>,
) -> Result<AgentReply, String> {
    Ok(run_turn(memory.0.clone(), session_id, message, pinned_thread).await)
}

/// One turn, with no Tauri in sight, so the check runner can drive the real
/// agent rather than a copy of it that might drift.
pub async fn run_turn(
    memory: Arc<crate::memory::SqliteConversationMemory>,
    session_id: String,
    message: String,
    pinned_thread: Option<PinnedThread>,
) -> AgentReply {
    let collected: Shared = Arc::new(Mutex::new(Collected::default()));
    let http = reqwest::Client::new();

    if !llama_server_ready(&http).await {
        return AgentReply::failed("The model server is still starting up.");
    }

    println!(
        "[agent] attached={}",
        match pinned_thread.as_ref().and_then(|t| t.story_title.as_deref()) {
            Some(title) => format!("yes: {title}"),
            None => "none".to_string(),
        }
    );

    let client = match llamafile::Client::from_url(LLAMA_SERVER_URL) {
        Ok(client) => client,
        Err(error) => return AgentReply::failed(error.to_string()),
    };
    let model = client.completion_model("gemma");

    let agent = AgentBuilder::new(model)
        .preamble(BASE_INSTRUCTION)
        .tool(SearchThreads {
            http: http.clone(),
            collected: collected.clone(),
        })
        .tool(GetThread {
            http,
            collected: collected.clone(),
        })
        // Rig runs ONE model call by default, so without this any tool use dies
        // with MaxTurnsError before the model ever answers. ADK looped by default.
        .default_max_turns(10)
        // Gemma bills chain-of-thought against the same max_tokens as the answer
        // and will spend all of it thinking, returning empty content with
        // finish_reason=Length.
        .additional_params(json!({
            "chat_template_kwargs": { "enable_thinking": false }
        }))
        .temperature(0.1)
        .max_tokens(2048)
        // Rig loads this conversation's history before the turn and appends the
        // new messages after it. Doing it here rather than by hand is what makes
        // the store swappable: the agent code never touches SQLite.
        .memory(memory)
        .conversation(session_id)
        .build();

    let prompt = with_attachment(&message, pinned_thread.as_ref());
    let reply = match agent.prompt(prompt).await {
        Ok(reply) => reply,
        Err(error) => {
            println!("[agent] failed: {error}");
            return AgentReply::failed(error.to_string());
        }
    };

    let out = collected.lock().unwrap();
    // turns ≈ tool calls + 1. Logged so the default_max_turns cap can be set
    // from real usage rather than guessed at.
    println!("[agent] tool calls: {}", out.tool_calls.len());
    AgentReply {
        reply,
        tool_calls: out.tool_calls.clone(),
        results: out.results.clone(),
        error: None,
    }
}
