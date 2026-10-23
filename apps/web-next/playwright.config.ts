import { defineConfig, devices } from "@playwright/test";
import { resolve } from "node:path";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  retries: 0,
  use: {
    baseURL: "http://127.0.0.1:5175",
    trace: "retain-on-failure",
  },
  webServer: [{
    command: `"${resolve(import.meta.dirname, "../agent-runtime/.venv/Scripts/python.exe")}" tests/web_fixture.py --port 8082`,
    cwd: resolve(import.meta.dirname, "../agent-runtime"),
    env: {
      PYTHONNOUSERSITE: "1", PYTHONIOENCODING: "utf-8",
      // Local model and RAG traffic must work even with an unusable system proxy.
      HTTP_PROXY: "http://127.0.0.1:1", HTTPS_PROXY: "http://127.0.0.1:1", NO_PROXY: "",
    },
    url: "http://127.0.0.1:8082/api/agent/projects",
    reuseExistingServer: false,
  }, {
    command: "npm run dev -- --port 5175 --strictPort",
    env: { AGENT_API_URL: "http://127.0.0.1:8082" },
    url: "http://127.0.0.1:5175",
    reuseExistingServer: false,
  }],
  projects: [{ name: "chrome", use: { ...devices["Desktop Chrome"], channel: "chrome" } }],
});
