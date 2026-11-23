import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` serves the UI on :5173 and forwards /api to the Python backend on :8000.
// `npm run build` writes ui/dist, which ui_server.py serves directly on :8000.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
