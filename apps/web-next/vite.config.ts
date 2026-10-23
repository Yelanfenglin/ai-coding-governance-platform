import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  server: {
    host: "127.0.0.1",
    proxy: {
      "/api/agent": {
        target: process.env.AGENT_API_URL ?? "http://127.0.0.1:8081",
        // Keep the browser origin/Host pair intact for the local Agent's same-origin check.
        changeOrigin: false,
      },
    },
  },
});
