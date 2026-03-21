import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  root: ".",
  build: {
    outDir: "dist-tauri",
    emptyOutDir: true,
  },
  resolve: {
    alias: {
      // Use the Tauri-specific WebLLM client instead of the Chrome extension one
      // Covers deep imports (e.g. from popup/components/) and shallow imports (e.g. from utils/)
      "../../utils/webLLMClient": path.resolve(
        __dirname,
        "src/tauri-compat/webLLMClient.ts"
      ),
      "./webLLMClient": path.resolve(
        __dirname,
        "src/tauri-compat/webLLMClient.ts"
      ),
    },
  },
  server: {
    port: 5173,
    strictPort: true,
  },
  optimizeDeps: {
    exclude: ["@mlc-ai/web-llm", "@huggingface/transformers", "onnxruntime-web"],
    entries: ["index.html"],
  },
});
