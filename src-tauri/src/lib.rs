pub mod agent;
pub mod corpus;
pub mod memory;
pub mod search;

use tauri::Manager;

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
    .manage(agent::AppEmbedder(std::sync::Arc::new(
      search::Embedder::load().expect("failed to load the embedding model"),
    )))
    .invoke_handler(tauri::generate_handler![open_external_url, agent::send_message])
    .setup(|app| {
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }

      // Auto-start the model server, production only.
      // In dev, run both by hand for fast iteration — see README.
      if !cfg!(debug_assertions) {
        // llama-server is NOT an externalBin: it resolves eight dylibs via
        // @loader_path, i.e. from its own directory, and externalBin copies a
        // single file. It ships as a resource folder instead and is launched
        // by path so its libraries sit beside it.
        let llama_dir = app
          .path()
          .resource_dir()
          .expect("no resource dir")
          .join("llama");
        let model = app
          .path()
          .resource_dir()
          .expect("no resource dir")
          .join("models")
          .join("google_gemma-4-26B-A4B-it-Q3_K_M.gguf");

        std::process::Command::new(llama_dir.join("llama-server"))
          .args([
            "-m", model.to_str().expect("bad model path"),
            // --jinja is load-bearing: it applies Gemma's own chat template so
            // tool calls come back as structured `tool_calls`. Without it they
            // arrive as raw <|tool_call> text and routing silently stops working.
            "--jinja",
            "-ngl", "99",
            "-c", "8192",
            "--host", "127.0.0.1",
            "--port", "8081",
          ])
          .spawn()
          .expect("failed to start llama-server");

        // No Python sidecar any more: the corpus, search and the agent all run
        // in this process (#48). Only the model server is still spawned, and #47
        // removes that too.
      }

      Ok(())
    })
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
