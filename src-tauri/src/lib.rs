use tauri::Manager;
use tauri_plugin_shell::ShellExt;

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
  //import the plugin to allow us to spawn the sidecar
    .plugin(tauri_plugin_shell::init())
    .invoke_handler(tauri::generate_handler![open_external_url])
    .setup(|app| {
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }

      // Auto-start the model server and Python sidecar, production only.
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

        // The sidecar starts immediately rather than waiting for the model to
        // load. It only needs llama-server when a request reaches Gemma, and
        // that is minutes away — the webview has to render and the user has to
        // type. /health reports `llama_server` so the state is visible.
        let sidecar = app.shell().sidecar("sidecar").expect("sidecar not found");
        sidecar.spawn().expect("failed to start sidecar");
      }

      Ok(())
    })
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
