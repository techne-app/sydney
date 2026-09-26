//! The agent, in Tauri core.
//!
//! Replaces `sidecar/agent.py`. Everything now runs in this one process: the
//! agent, the corpus, search, and — since #47 — Gemma itself. There is no
//! sidecar and no model server, which is what iOS requires: it forbids the
//! subprocesses the original design was built on.
//!
//! The problem the agent solves is unchanged: the model calls a tool, sees the
//! result, and decides again. Routing used to give it one tool call per turn and
//! then stop, so when a question needed something the model didn't have it
//! invented it instead. Every hallucination this project hit came from that gap.

use std::sync::{Arc, Mutex};

use rig_core::agent::AgentBuilder;
use rig_core::client::CompletionClient;
use rig_core::completion::Prompt;
use rig_core::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Gemma. Loaded once at startup — see `Model` — because this reads 12GB from
/// disk, which would be fatal per request.
const CHAT_MODEL: &str = "google_gemma-4-26B-A4B-it-Q3_K_M.gguf";

/// Gemma's own template is applied by rig-llama-cpp, which also parses the
/// `<|tool>` markers back into structured tool calls. That is the work
/// `llama-server --jinja` used to do for us over HTTP.
const N_CTX: u32 = 8192;

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
    corpus: Arc<crate::corpus::Corpus>,
    embedder: Arc<crate::search::Embedder>,
    model: Arc<rig_llama_cpp::Client>,
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

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.collected.lock().unwrap().tool_calls.push(ToolCallRecord {
            name: Self::NAME.into(),
            arguments: json!({ "keyword_filter": args.keyword_filter }),
        });

        let body = crate::search::run_search(&self.corpus, &self.embedder, &self.model, &args.keyword_filter, 3).await;

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
    corpus: Arc<crate::corpus::Corpus>,
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

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.collected.lock().unwrap().tool_calls.push(ToolCallRecord {
            name: Self::NAME.into(),
            arguments: json!({ "thread_id": args.thread_id }),
        });

        let Some(meta) = self.corpus.get(args.thread_id) else {
            return Ok(serde_json::json!({
                "error": "No thread with that id is in the last 30 days of data. \
                          Search for the topic instead."
            }));
        };

        // The link is split off here and never handed to the model. Given a URL
        // it writes it into its prose, and the UI renders the same link again
        // from `results`. Taking it out is what makes that impossible rather
        // than merely discouraged.
        if let Some(link) = meta.anchor.clone() {
            self.collected.lock().unwrap().results.push(UiResult {
                thread_id: meta.thread_id,
                title: meta.story_title.clone().unwrap_or_default(),
                link,
            });
        }

        let body = crate::corpus::public_view(&meta, true);
        Ok(body)
    }
}

// ---------------------------------------------------------------------------
// The command the frontend calls
// ---------------------------------------------------------------------------

/// Built once at startup and shared by every turn. The embedder in particular
/// must not be rebuilt per request — it loads a model from disk.
pub struct AgentMemory(pub Arc<crate::memory::SqliteConversationMemory>);
pub struct AppCorpus(pub Arc<crate::corpus::Corpus>);
pub struct AppEmbedder(pub Loaded<crate::search::Embedder>);

/// A model that must be released *before* the process exits.
///
/// llama.cpp frees the Metal device from a C++ static destructor at exit. A
/// model still alive at that point leaves its residency sets non-empty, ggml
/// aborts, and macOS shows "quit unexpectedly" on every close. Destructor
/// ordering across libraries is not something a program can rely on, so the
/// owner releases first: `RunEvent::Exit` calls `release`, the client's own Drop
/// joins its worker thread and frees the context, and the static destructor
/// then finds nothing to complain about.
pub struct Loaded<T>(std::sync::RwLock<Option<Arc<T>>>);

impl<T> Loaded<T> {
    pub fn new(value: T) -> Self {
        Self(std::sync::RwLock::new(Some(Arc::new(value))))
    }

    pub fn get(&self) -> Option<Arc<T>> {
        self.0.read().ok().and_then(|slot| slot.clone())
    }

    /// Drop the value now. A turn still in flight holds its own Arc, so the
    /// release simply waits for that clone to go — never mid-generation.
    pub fn release(&self) {
        if let Ok(mut slot) = self.0.write() {
            slot.take();
        }
    }
}

/// Gemma, loaded once. The reranker in search.rs borrows the same handle, so
/// there is one 12GB model in memory rather than two.
pub struct Model(pub Loaded<rig_llama_cpp::Client>);

/// Load Gemma. Blocking and slow — a minute or two — so the caller must show
/// the user something while it runs.
pub fn load_model() -> Result<rig_llama_cpp::Client, String> {
    let path = crate::search::resolve_model(CHAT_MODEL);
    println!("[agent] loading {path}");
    let client = rig_llama_cpp::Client::builder(path)
        .n_ctx(N_CTX)
        .build()
        .map_err(|e| e.to_string())?;
    println!("[agent] chat model loaded");
    Ok(client)
}

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

#[tauri::command]
pub async fn send_message(
    memory: tauri::State<'_, AgentMemory>,
    corpus: tauri::State<'_, AppCorpus>,
    embedder: tauri::State<'_, AppEmbedder>,
    model: tauri::State<'_, Model>,
    session_id: String,
    message: String,
    pinned_thread: Option<PinnedThread>,
) -> Result<AgentReply, String> {
    let (Some(embedder), Some(model)) = (embedder.0.get(), model.0.get()) else {
        return Ok(AgentReply::failed("The app is shutting down."));
    };
    Ok(run_turn(
        memory.0.clone(),
        corpus.0.clone(),
        embedder,
        model,
        session_id,
        message,
        pinned_thread,
    )
    .await)
}

/// One turn, with no Tauri in sight, so the check runner can drive the real
/// agent rather than a copy of it that might drift.
pub async fn run_turn(
    memory: Arc<crate::memory::SqliteConversationMemory>,
    corpus: Arc<crate::corpus::Corpus>,
    embedder: Arc<crate::search::Embedder>,
    model: Arc<rig_llama_cpp::Client>,
    session_id: String,
    message: String,
    pinned_thread: Option<PinnedThread>,
) -> AgentReply {
    let collected: Shared = Arc::new(Mutex::new(Collected::default()));

    println!(
        "[agent] attached={}",
        match pinned_thread.as_ref().and_then(|t| t.story_title.as_deref()) {
            Some(title) => format!("yes: {title}"),
            None => "none".to_string(),
        }
    );

    let completion_model = model.completion_model(CHAT_MODEL);

    let agent = AgentBuilder::new(completion_model)
        .preamble(BASE_INSTRUCTION)
        .tool(SearchThreads {
            corpus: corpus.clone(),
            embedder,
            model: model.clone(),
            collected: collected.clone(),
        })
        .tool(GetThread {
            corpus,
            collected: collected.clone(),
        })
        // Rig runs ONE model call by default, so without this any tool use dies
        // with MaxTurnsError before the model ever answers. ADK looped by default.
        .default_max_turns(10)
        // Gemma bills chain-of-thought against the same max_tokens as the
        // answer and will spend all of it thinking, returning empty content.
        //
        // In-process the key is `thinking`, NOT the HTTP-era
        // `chat_template_kwargs.enable_thinking`: rig-llama-cpp reads this from
        // additional_params and feeds it to the Jinja template itself. The old
        // key is silently ignored.
        .additional_params(json!({ "thinking": false }))
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
