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
use rig_core::message::Message;
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
        })
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
