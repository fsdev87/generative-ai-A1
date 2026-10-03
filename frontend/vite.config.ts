/// <reference types="vitest/config" />
// Vite build, dev server and Vitest configuration.
// In development `npm run dev` serves the app on http://localhost:5173 and forwards /api to the
// FastAPI backend (BACKEND_URL, default http://localhost:8000), so the browser only ever talks to
// one origin. In production nginx does the same (see nginx.conf).
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const backendUrl = process.env.BACKEND_URL ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: { "/api": { target: backendUrl, changeOrigin: true } },
  },
  preview: {
    port: 5173,
    proxy: { "/api": { target: backendUrl, changeOrigin: true } },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    css: false,
    restoreMocks: true,
    unstubGlobals: true,
  },
});
