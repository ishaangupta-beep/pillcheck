import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the API runs on :8000 (uvicorn); in production the backend
// serves this app itself, so every API call is same-origin.
const api = process.env.VITE_DEV_API ?? "http://localhost:8000";
const proxied = ["/health", "/demo", "/patients", "/medications", "/references", "/checks", "/actions", "/agent"];

export default defineConfig({
  plugins: [react()],
  server: { proxy: Object.fromEntries(proxied.map((p) => [p, { target: api, changeOrigin: true }])) },
  build: { outDir: "dist", assetsDir: "assets", sourcemap: false },
});
