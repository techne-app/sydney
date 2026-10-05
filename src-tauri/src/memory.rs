//! Durable conversation memory for the agent.
//!
//! Replaces what ADK's `DatabaseSessionService` gave us for free. Sessions
//! persist rather than living in memory for one reason: reopening the app and
//! asking a follow-up on an old conversation has to still work, and an
//! in-process store loses that on every restart — while the UI goes on showing
//! the conversation, so the app looks like it has amnesia rather than like it
//! restarted.
//!
//! This is NOT the Dexie store. The frontend keeps its own copy of the rendered
//! prose in IndexedDB; this holds the structured messages the model reasons
//! over. Same conversation seen from two sides, linked by the conversation id.

use std::path::PathBuf;
use std::sync::Mutex;

use rig_core::memory::{ConversationMemory, MemoryError};
use rig_core::message::{Message, ToolResultContent, UserContent};
use rig_core::OneOrMany;
use rig_core::wasm_compat::WasmBoxedFuture;
use rusqlite::Connection;

/// Beside the corpus cache and ADK's old sessions.db. Contents/Resources is
/// read-only in a signed .app, so anything written at runtime lives here.
fn db_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home)
        .join("Library/Application Support/com.technesystems.techne")
        .join("agent-memory.db")
}

pub struct SqliteConversationMemory {
    /// rusqlite's Connection is Send but not Sync, and the trait needs both.
    /// One lock is fine here: these are tiny local reads on a desktop app, not
    /// a server under load.
    conn: Mutex<Connection>,
    /// The thread the user currently has pinned, set before each turn.
    ///
    /// `load` needs it because only one thread's comments fit in the window, and
    /// which one to keep cannot be worked out from the history alone. The first
    /// attempt at this kept whichever was read most recently — which is precisely
    /// the wrong one at the moment the user pins something new, because the model
    /// is about to read the new thread and add it on top of the old.
    focus: Mutex<Option<u64>>,
}

impl SqliteConversationMemory {
    pub fn open() -> Result<Self, rusqlite::Error> {
        Self::open_at(db_path())
    }

    /// Open a store at an explicit path. Tests use a throwaway file so a check
    /// run cannot pollute — or be polluted by — real conversations.
    pub fn open_at(path: PathBuf) -> Result<Self, rusqlite::Error> {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let conn = Connection::open(&path)?;
        // One row per message rather than one blob per conversation, so
        // appending a turn does not rewrite the whole history.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS agent_messages (
                conversation_id TEXT NOT NULL,
                seq             INTEGER NOT NULL,
                message         TEXT NOT NULL,
                PRIMARY KEY (conversation_id, seq)
            )",
            [],
        )?;
        println!("[agent] memory at {}", path.display());
        Ok(Self {
            conn: Mutex::new(conn),
            focus: Mutex::new(None),
        })
    }

    /// Tell `load` which thread's comments to keep. Set before each turn from
    /// whatever the user has pinned; `None` falls back to the most recent.
    pub fn focus_on(&self, thread_id: Option<u64>) {
        if let Ok(mut focus) = self.focus.lock() {
            *focus = thread_id;
        }
    }

    fn load_blocking(&self, conversation_id: &str) -> Result<Vec<Message>, MemoryError> {
        let conn = self.conn.lock().map_err(|e| MemoryError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare("SELECT message FROM agent_messages WHERE conversation_id = ?1 ORDER BY seq")
            .map_err(MemoryError::backend)?;
        let rows = stmt
            .query_map([conversation_id], |row| row.get::<_, String>(0))
            .map_err(MemoryError::backend)?;

        let mut out = Vec::new();
        for row in rows {
            let json = row.map_err(MemoryError::backend)?;
            // A message we can no longer parse is skipped rather than fatal:
            // a stored history should not make the app unopenable.
            match serde_json::from_str::<Message>(&json) {
                Ok(message) => out.push(message),
                Err(error) => eprintln!("[agent] skipping unreadable message: {error}"),
            }
        }
        let focus = self.focus.lock().ok().and_then(|f| *f);
        let out = hide_superseded_threads(out, focus);

        // What the model is actually about to be given. Guessing at this twice
        // produced two wrong fixes, so it is now printed: `kept` is the thread
        // whose comments survived, and `chars` is everything going into the
        // window beside the preamble and the reply budget.
        let chars: usize = out
            .iter()
            .map(|m| serde_json::to_string(m).map(|j| j.len()).unwrap_or(0))
            .sum();
        println!(
            "[history] {} messages, {chars} chars (~{} tokens), focus={focus:?}",
            out.len(),
            chars * 2 / 7
        );
        Ok(out)
    }

    fn append_blocking(
        &self,
        conversation_id: &str,
        messages: Vec<Message>,
    ) -> Result<(), MemoryError> {
        let mut conn = self.conn.lock().map_err(|e| MemoryError::Internal(e.to_string()))?;
        let tx = conn.transaction().map_err(MemoryError::backend)?;

        let next: i64 = tx
            .query_row(
                "SELECT COALESCE(MAX(seq), -1) + 1 FROM agent_messages WHERE conversation_id = ?1",
                [conversation_id],
                |row| row.get(0),
            )
            .map_err(MemoryError::backend)?;

        for (offset, message) in messages.iter().enumerate() {
            let json = serde_json::to_string(message).map_err(MemoryError::backend)?;
            tx.execute(
                "INSERT INTO agent_messages (conversation_id, seq, message) VALUES (?1, ?2, ?3)",
                rusqlite::params![conversation_id, next + offset as i64, json],
            )
            .map_err(MemoryError::backend)?;
        }
        tx.commit().map_err(MemoryError::backend)
    }

    fn clear_blocking(&self, conversation_id: &str) -> Result<(), MemoryError> {
        let conn = self.conn.lock().map_err(|e| MemoryError::Internal(e.to_string()))?;
        conn.execute(
            "DELETE FROM agent_messages WHERE conversation_id = ?1",
            [conversation_id],
        )
        .map(|_| ())
        .map_err(MemoryError::backend)
    }
}

/// Hide the comments of every thread but the one in focus.
///
/// A thread read with `get_thread` runs to ~14,000 characters, and the entire
/// history is re-sent on every turn, so a second thread in one conversation
/// overflows the 8,192-token window outright:
/// `Prompt 11193 tokens exceeds n_ctx 8192`. Reading one thread and then another
/// is an ordinary thing to do, so the history is what has to give.
///
/// Only the `comments` field goes. The thread's id, title and theme stay, so the
/// model still knows it read that thread and can call `get_thread` again if the
/// conversation comes back to it — which costs nothing, since `ThreadStore` has
/// it. It simply stops carrying the full text of a discussion nobody is asking
/// about any more.
///
/// `focus` is the pinned thread. Keeping the most *recent* one instead looks
/// equivalent and is not: at the moment the user pins a new thread, the newest
/// in the history is the OLD one, and the model is about to read the new one on
/// top of it. That is the overflow this is here to prevent, so it has to be told
/// which thread matters rather than inferring it. With nothing pinned there is
/// nothing to infer from, and the most recent is the best guess available.
///
/// Nothing is deleted: the stored rows keep their full text, and this only
/// shapes what is handed to the model for one turn.
fn hide_superseded_threads(messages: Vec<Message>, focus: Option<u64>) -> Vec<Message> {
    let mut kept_one = false;
    let mut out: Vec<Message> = Vec::with_capacity(messages.len());

    for message in messages.into_iter().rev() {
        let Message::User { content } = message else {
            out.push(message);
            continue;
        };

        let items: Vec<UserContent> = content
            .iter()
            .cloned()
            .map(|item| {
                let UserContent::ToolResult(mut result) = item else {
                    return item;
                };
                let parts: Vec<ToolResultContent> = result
                    .content
                    .iter()
                    .cloned()
                    .map(|part| {
                        let ToolResultContent::Text(mut text) = part else {
                            return part;
                        };
                        // Only a get_thread result has `comments`; a search
                        // result or an error passes through untouched.
                        let Ok(mut body) = serde_json::from_str::<serde_json::Value>(&text.text)
                        else {
                            return ToolResultContent::Text(text);
                        };
                        if body.get("comments").is_none() {
                            return ToolResultContent::Text(text);
                        }
                        let is_focus = focus.is_some_and(|id| {
                            body.get("thread_id").and_then(serde_json::Value::as_u64) == Some(id)
                        });
                        // Walking backwards, so with no focus the first one
                        // reached is the most recently read.
                        if is_focus || (focus.is_none() && !kept_one) {
                            kept_one = true;
                            return ToolResultContent::Text(text);
                        }
                        body["comments"] = serde_json::Value::String(
                            "(not shown: this is no longer the thread being discussed. \
                             Call get_thread with its id to read it again.)"
                                .into(),
                        );
                        text.text = body.to_string();
                        ToolResultContent::Text(text)
                    })
                    .collect();
                if let Ok(parts) = OneOrMany::many(parts) {
                    result.content = parts;
                }
                UserContent::ToolResult(result)
            })
            .collect();

        // many() fails only on an empty list, and we map one-for-one, so this
        // cannot lose a message.
        match OneOrMany::many(items) {
            Ok(content) => out.push(Message::User { content }),
            Err(_) => out.push(Message::User { content }),
        }
    }

    out.reverse();
    out
}

/// Rig calls `load` before each turn and `append` after a successful one, with
/// the user prompt, the assistant reply, and any tool-call/tool-result pairs.
/// Getting that lifecycle from the framework is the reason to implement this
/// trait rather than hand-manage a history map.
impl ConversationMemory for SqliteConversationMemory {
    fn load<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<Vec<Message>, MemoryError>> {
        Box::pin(async move { self.load_blocking(conversation_id) })
    }

    fn append<'a>(
        &'a self,
        conversation_id: &'a str,
        messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async move { self.append_blocking(conversation_id, messages) })
    }

    fn clear<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async move { self.clear_blocking(conversation_id) })
    }
}
