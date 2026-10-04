pub mod agent;
pub mod corpus;
pub mod memory;
pub mod search;
pub mod summary;


// Unused since external links moved to tauri-plugin-shell's own handler (it
// already intercepts target="_blank" clicks; ours ran alongside it and the
// plugin's denied call was the "shell.open not allowed" console error).
// Kept registered as a fallback while that path is proven — note it opens any
// string with no scope validation, which the plugin's path does apply.
#[tauri::command]
fn open_external_url(url: String) {
  let _ = std::process::Command::new("open").arg(&url).spawn();
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .plugin(tauri_plugin_shell::init())
    .manage(agent::AgentMemory(std::sync::Arc::new(
      memory::SqliteConversationMemory::open().expect("failed to open agent memory"),
    )))
    .manage(agent::AppCorpus({
      // Loads the cache, then refreshes in the background — the UI has plenty
      // to show before search is needed.
      let corpus = std::sync::Arc::new(corpus::Corpus::new());
      corpus::start(corpus.clone());
      corpus
    }))
    .manage(agent::AppEmbedder(agent::Loaded::new(
      search::Embedder::load().expect("failed to load the embedding model"),
    )))
    // Gemma: 12GB, a minute or two. Loaded once here rather than per request,
    // which is the whole reason it lives in Tauri state.
    .manage(agent::Model(agent::Loaded::new(
      agent::load_model().expect("failed to load the chat model"),
    )))
    .invoke_handler(tauri::generate_handler![
      open_external_url,
      agent::send_message,
      summary::summarize_thread_command
    ])
    .setup(|app| {
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }

      // Nothing is spawned any more. The agent, the corpus, search and both
      // models all run in this process — no sidecar, no model server. That is
      // what iOS requires: it forbids child processes outright, and every
      // piece of the old design depended on one.

      Ok(())
    })
    .build(tauri::generate_context!())
    .expect("error while building tauri application")
    .run(|handle, event| {
      // Release both models before the process tears down — see Loaded.
      if matches!(event, tauri::RunEvent::Exit) {
        use tauri::Manager;
        handle.state::<agent::Model>().0.release();
        handle.state::<agent::AppEmbedder>().0.release();
      }
    });
}
