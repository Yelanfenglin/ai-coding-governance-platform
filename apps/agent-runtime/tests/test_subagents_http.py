"""Default HTTP/SSE transports through CLI, delegation, RAG, file tools and scoring.

The server is scripted: this tests wiring, not model quality. SSE frames follow
the existing wiring_acceptance_20261021 stub, including a trailing usage frame.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from app import cli
from app.contracts import Message, MessageEntry, TurnEndEntry
from app.session import JsonlSessionStore
from evaluation.runner import score


@pytest.mark.parametrize("mode", ["foreground", "background", "continuable"])
async def test_cli_delegation_and_delivery_through_default_http_transports(
    tmp_path, monkeypatch, capsys, mode,
):
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    requests = []
    searches = []

    def tool(name, arguments, cid):
        return {"tool_calls": [{"index": 0, "id": cid, "type": "function",
                                "function": {"name": name, "arguments": json.dumps(arguments)}}]}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):  # noqa: N802
            searches.append(self.path)
            payload = json.dumps([{
                "document_id": "doc-clock", "chunk_id": "clock-12", "product": "M9P6362",
                "version": "V1.16", "kind": "TEXT", "page_start": 12, "page_end": 12,
                "section_path": "clock", "text": "Clock is 16MHz.", "score": 0.9,
            }]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            messages = body["messages"]
            child = "research/review subagent" in messages[0]["content"]
            results = [m for m in messages if m["role"] == "tool"]
            if child:
                delta = ({"content": "16MHz [document=doc-clock chunk=clock-12 page=12]"}
                         if results else tool("search_spec", {"question": "clock"}, "lookup"))
            elif not results:
                delta = tool("subagent", {"prompt": "Research the clock; cite the manual.",
                             "context_mode": "fork" if mode == "continuable" else "spawn",
                             "run_in_background": mode != "foreground",
                             "continuable": mode == "continuable"},
                             "child")
            elif not any(m.get("name") == "write_file" for m in results):
                delta = tool("write_file", {"path": "config.json", "content": '{"duty":40}'},
                             "edit")
            elif not any(m.get("name") == "run_command" for m in results):
                delta = tool("run_command", {"command": ["python", "-c",
                    "import json; assert json.load(open('config.json'))['duty']==40; print('ok')",
                ]}, "verify")
            elif mode != "foreground" and not any(m.get("name") == "wait_agent" for m in results):
                agent_id = json.loads(next(m["content"] for m in results
                                          if m.get("name") == "subagent"))["agent_id"]
                delta = tool("wait_agent", {"agent_id": agent_id}, "wait-first")
            elif (mode == "continuable"
                  and not any(m.get("name") == "send_message" for m in results)):
                agent_id = json.loads(next(m["content"] for m in results
                                          if m.get("name") == "subagent"))["agent_id"]
                delta = tool("send_message", {"agent_id": agent_id,
                             "message": "Confirm the clock again using your previous findings."},
                             "follow-up")
            elif mode == "continuable" and sum(m.get("name") == "wait_agent" for m in results) == 1:
                agent_id = json.loads(next(m["content"] for m in results
                                          if m.get("name") == "subagent"))["agent_id"]
                delta = tool("wait_agent", {"agent_id": agent_id}, "wait-second")
            else:
                delta = {"content": "Updated duty to 40. Clock: doc-clock/clock-12 p12."}
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            frames = [
                {"choices": [{"index": 0, "delta": delta}]},
                {"choices": [{"index": 0, "delta": {},
                              "finish_reason": "tool_calls" if "tool_calls" in delta else "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 6, "completion_tokens": 4,
                                           "total_tokens": 10}},
            ]
            for frame in frames:
                self.wfile.write(f"data: {json.dumps(frame)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root, seed = tmp_path / "work", tmp_path / "seed"
    root.mkdir()
    seed.mkdir()
    for path in (root, seed):
        (path / "config.json").write_text('{"duty":30}', encoding="utf-8")
    base = f"http://127.0.0.1:{server.server_port}"
    delivery_dir = tmp_path / "delivery"
    if mode == "continuable":
        prior = JsonlSessionStore(tmp_path / "session.jsonl")
        for message in [Message("system", "parent host"), Message("user", "PRIOR_CONTEXT"),
                        Message("assistant", "prior answer", reasoning_content="prior reasoning")]:
            await prior.append(MessageEntry(0, 0, message, "now"))
        await prior.append(TurnEndEntry(1, "completed", "now"))
    try:
        code = await cli.run_once(cli.build_parser().parse_args([
            "Research the clock, then change duty to 40", "--base-url", base + "/v1",
            "--api-key", "offline", "--rag-url", base + "/rag", "--rag-key", "offline",
            "--workspace", str(root), "--session", str(tmp_path / "session.jsonl"),
            "--subagents", "--delivery-dir", str(delivery_dir), "--max-steps", "12",
            "--no-compaction", "--json",
        ]))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    response = json.loads(capsys.readouterr().out)
    assert code == 0 and response["status"] == "completed"
    parent_steps, total = {"foreground": (4, 6), "background": (5, 7),
                           "continuable": (7, 10)}[mode]
    assert response["steps"] == parent_steps and response["total_steps"] == total
    assert response["usage"]["total_tokens"] == total * 10 and len(requests) == total
    assert searches and "clock-12" in response["subagents"][0]["answer"]
    assert all(r["model"] == "deepseek-flash" and r["thinking"] == {"type": "enabled"}
               and r["reasoning_effort"] == "high" and r["stream"] for r in requests)
    if mode == "continuable":
        child_requests = [r for r in requests
                          if "research/review subagent" in r["messages"][0]["content"]]
        assert any(m.get("content") == "PRIOR_CONTEXT" for m in child_requests[0]["messages"])
        assert any(m.get("reasoning_content") == "prior reasoning"
                   for m in child_requests[0]["messages"])
        assert any("previous findings" in str(m.get("content"))
                   for m in child_requests[-1]["messages"])
    history = await JsonlSessionStore(response["subagents"][0]["session"]).load_messages()
    assert any(m.role == "tool" and "clock-12" in m.content for m in history)
    assert json.loads((root / "config.json").read_text())["duty"] == 40
    delivery = json.loads(Path(response["delivery"]).read_text(encoding="utf-8"))
    assert delivery["commands"][0]["matches_delivery"]
    assert '"exit_code": 0' in delivery["commands"][0]["content"]
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"id": "http-wiring", "seed": str(seed), "checks": [
        {"id": "duty", "kind": "json_value", "path": "config.json",
         "pointer": "/duty", "expected": 40},
    ]}), encoding="utf-8")
    assert score(case, Path(response["delivery"]), root, tmp_path / "score")["passed"]
