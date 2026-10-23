# Repository Guidelines

## Product structure

The project is a local embedded programming assistant:

```text
Vue workspace -> localhost HTTP/SSE -> Python Agent Runtime -> model / RAG / local tools
```

- `apps/web-next/`: Vue 3, TypeScript, Vite, Pinia and Naive UI.
- `apps/agent-runtime/`: runtime, local tools, memory, compaction, CLI and Web adapter.
- `apps/rag-api/`: specification retrieval, PostgreSQL/Milvus and evaluation.

## Integration

- Reuse `app.cli.run_once` for Web assembly and `app.loop.run_agent` for execution.
  Do not add another loop, duplicate tool registration or parse CLI stdout as a protocol.
- Forward actual events. `TaskFinished` follows host checks and delivery recording;
  `TurnFinished` is not task completion.
- The host selects workspace, credentials and verification commands. Keep execution,
  sessions and memory local; the service binds to loopback and checks browser origins.
- Tools execute with the local user's permissions. Do not describe them as an OS sandbox.
- Default model: `deepseek-flash`, thinking enabled, reasoning effort high.
  Do not call Pro. Default context window is 1,000,000 tokens, configurable in `.env`.
- Retrieval, answer support, host simulation, vendor compilation and hardware results
  are distinct. Preserve historical measurements with their date and scope.

## Development

Use each module's virtual environment and set `PYTHONNOUSERSITE=1`.
Run affected tests; avoid unrelated refactoring. Root `npm run dev` starts the Agent
and Vue, `npm run build` builds Vue, and `npm start` serves the built UI.

Keep the public repository focused on current source, configuration examples,
core tests and concise product/technical documentation. Plans, research, acceptance
logs, private evidence and resume materials belong outside the repository or under
ignored `output/`. Back up documents externally before rewriting or retiring them.
Never commit `.env`, keys, databases, sessions, uploaded documents or runtime caches.
