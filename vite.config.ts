import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  root: ".",
  build: {
    outDir: "dist-tauri",
    emptyOutDir: true,
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
