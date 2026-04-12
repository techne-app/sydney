use tauri_plugin_shell::ShellExt;

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

      // Auto-start the Python sidecar only in production
      // In dev mode, run `uv run main.py` manually for fast iteration
      if !cfg!(debug_assertions) {
        let sidecar = app.shell().sidecar("sidecar").expect("sidecar not found");
        sidecar.spawn().expect("failed to start sidecar");
      }

      Ok(())
    })
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
