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
///
/// One window, shared by everything: the agent's turns, the reranker's 100
/// candidates, and a whole HN thread to summarise. The last of those sets the
/// size — a busy thread runs to 196,000 characters, which 8,192 tokens cannot
/// hold. Gemma is trained to 262,144.
///
/// 8× the window costs about 1.1 GB, not 8× the memory: Gemma uses sliding-
/// window attention on 25 of its 30 layers, and that cache is a fixed 4,608
/// cells however large the context. Only the 5 full-attention layers scale.
/// Measured: 13.7 GB resident at 8,192, 14.9 GB at 65,536, on a 24 GB machine.
/// That arithmetic will not survive a phone, so iOS will have to size this
/// again from scratch.
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
    "\n\nA message may begin with a note naming a discussion the user has open ",
    "in front of them. When they say \"this thread\", \"this discussion\", ",
    "\"this\", or \"this one\", they mean that one — not an item from a list ",
    "of search results. Read it with get_thread rather than searching.",
    "\n\nget_thread returns the first comments of a thread, not all of them, and ",
    "says how many. If what you were asked about is not in them, say it was not ",
    "in the part you read. Never settle a question by what is missing.",
    "\n\nGive figures exactly as the comments give them. Do not convert, scale, ",
    "combine or estimate a number, and do not turn a single figure into a range. ",
    "If a comment says a city used some amount on one day, that is what it says — ",
    "not what the city uses per day."
);

/// The thread the user has dragged into the chat, as the frontend sends it.
#[derive(Debug, Clone, Deserialize)]
pub struct PinnedThread {
    /// Needed so the model can actually go and read it. Without an id the
    /// attachment was a dead end: it described a thread the model had no way to
    /// open, so the only thing it could do was repeat the description back.
    pub id: u64,
    pub story_title: Option<String>,
    pub theme: Option<String>,
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
        // The id, not a summary. This used to carry the corpus's stored summary,
        // which the pipeline writes once and never revises — on a measured card
        // it described 2 comments of a 151-comment discussion. Worse, handing
        // over a summary decides in advance what can be asked: 130 words cannot
        // answer "did anyone mention performance?". Naming the thread instead
        // lets the model go and read it when the question needs it.
        Some(t) => format!(
            "[The user has this discussion open in front of them: thread {} — \"{}\", about {}. \
             Call get_thread with that id to read what people actually said.]\n\n{}",
            t.id,
            t.story_title.as_deref().unwrap_or(""),
            t.theme.as_deref().unwrap_or(""),
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
    store: Arc<crate::summary::ThreadStore>,
    collected: Shared,
}

impl Tool for GetThread {
    const NAME: &'static str = "get_thread";
    type Args = ThreadArgs;
    type Output = serde_json::Value;
    type Error = ToolFailed;

    fn description(&self) -> String {
        "Read the comments of one Hacker News discussion. Use it whenever the user \
         asks anything about a specific thread — what it is about, what people \
         argued, whether anyone mentioned something in particular — including the \
         discussion they currently have open. Search results carry only a title \
         and a one-line description; this returns what people actually wrote."
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

        // The corpus is consulted for the title and the link, not for permission.
        // It used to refuse outright for an id it did not hold, which was right
        // when the corpus WAS the answer — the tool returned a row from it. Now
        // the comments come from the thread service, which has every thread, so
        // refusing on a corpus miss would turn "not in our last 30 days of
        // metadata" into "cannot be read", which is not true.
        let meta = self.corpus.get(args.thread_id);

        // The link is split off here and never handed to the model. Given a URL
        // it writes it into its prose, and the UI renders the same link again
        // from `results`. Taking it out is what makes that impossible rather
        // than merely discouraged.
        if let Some(link) = meta.as_ref().and_then(|m| m.anchor.clone()) {
            self.collected.lock().unwrap().results.push(UiResult {
                thread_id: args.thread_id,
                title: meta.as_ref().and_then(|m| m.story_title.clone()).unwrap_or_default(),
                link,
            });
        }

        // The thread itself, not the corpus's stored summary. `public_view` still
        // supplies the title and theme, which are cheap and orient the model,
        // but the body is now what people wrote.
        let mut body = match &meta {
            Some(meta) => crate::corpus::public_view(meta, false),
            None => json!({ "thread_id": args.thread_id }),
        };

        match self.store.comments(args.thread_id).await {
            Ok((text, used, total)) => {
                body["comments"] = text.into();
                // Said plainly because the model cannot see the edge of what it
                // was given. Without this it answers "nobody mentioned X" when
                // the truth is "nobody in the first 27 comments mentioned X",
                // which is the shape every hallucination in this project took.
                body["coverage"] = format!(
                    "These are the first {used} of {total} comments, in reply order.                      If the answer is not here, say it was not in the part you could read."
                )
                .into();
            }
            // The corpus row alone is still worth answering from.
            Err(error) => body["comments_unavailable"] = error.into(),
        }

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
            // Taking our Arc frees the value only if nobody else holds a clone.
            // A turn still in flight keeps its own, which is what stops a model
            // being freed mid-generation.
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
    store: tauri::State<'_, Arc<crate::summary::ThreadStore>>,
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
        store.inner().clone(),
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
    store: Arc<crate::summary::ThreadStore>,
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
            store,
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
