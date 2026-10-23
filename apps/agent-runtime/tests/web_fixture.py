"""Real HTTP model/RAG fixtures for the browser acceptance test; no paid endpoints."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

import uvicorn

from app import cli
from app.web import create_app

DOCUMENT = "6b1f0f2a-3c4d-4e5f-8a9b-0c1d2e3f4a5b"
CHUNK = "0dd2c1de-1f8c-4d3a-9a10-2f6b8a1c9e11"


def call(name, arguments):
    return {"id": f"call-{uuid4().hex}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


def handler(workspace):
    class Stub(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def record(self, value):
            path = workspace / ".agent-runtime" / "fixture-requests.jsonl"
            path.parent.mkdir(exist_ok=True)
            with path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(value, ensure_ascii=False) + "\n")

        def do_GET(self):
            self.record({"method": "GET", "path": self.path})
            hit = {"document_id": DOCUMENT, "chunk_id": CHUNK, "product": "M9F6820",
                   "version": "V1.7", "kind": "TEXT", "page_start": 113, "page_end": 113,
                   "section_path": "12.1 ADC", "text": "测试手册：模块配置必须保留来源。",
                   "score": 0.8}
            payload = [hit] if self.path.startswith("/search?") else hit
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload, ensure_ascii=False).encode())

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.record({"method": "POST", "body": body})
            assert self.path == "/chat/completions"
            assert body["stream"] is True
            messages = body["messages"]
            user = max(index for index, item in enumerate(messages) if item["role"] == "user")
            question = messages[user]["content"]
            results = [item for item in messages[user + 1:] if item["role"] == "tool"]
            count = len(results)
            delta, reason = {}, "tool_calls"
            if "等待取消" in question:
                delta, reason = {"content": "正在等待取消"}, "stop"
            elif "取消后继续" in question:
                delta, reason = {"content": "已在原会话续接，未重复执行工具。"}, "stop"
            elif "继续把" in question:
                assert any("已创建 generated.c" in str(m.get("content")) for m in messages[:user])
                if count == 0:
                    delta = {"tool_calls": [call("edit_file", {"path": "generated.c",
                             "old_text": "value = 2", "new_text": "value = 3"})]}
                elif count == 1:
                    delta = {"tool_calls": [call("run_command", {"command": [
                        "python", "-c", "from pathlib import Path; "
                        "assert 'value = 3' in Path('generated.c').read_text()",
                    ]})]}
                else:
                    delta, reason = {
                        "content": "已沿用上一轮工程，将 value 改为 3，检查通过。",
                    }, "stop"
            elif count == 0:
                delta = {"tool_calls": [call("read_file", {"path": "seed.c"}), call(
                    "search_spec", {"question": "M9F6820 的 ADC 配置", "product": "M9F6820",
                                    "version": "V1.7", "top_k": 1})]}
            elif count == 2:
                evidence = next(m["content"] for m in results if "document=" in m["content"])
                document = re.search(r"document=([\w-]+)", evidence).group(1)
                chunk = re.search(r"chunk=([\w-]+)", evidence).group(1)
                delta = {"tool_calls": [call("read_context", {
                    "document_id": document, "chunk_id": chunk,
                })]}
            elif count == 3:
                delta = {"tool_calls": [call("write_file", {
                    "path": "generated.c", "content": "int value = 2;\n",
                }), call("remember_fact", {"key": "confirmed.unit", "value": "HOST_ONLY",
                                          "source": "user confirmed fixture environment"})]}
            elif count == 5:
                check = call("run_command", {"command": [
                    "python", "-c", "from pathlib import Path; "
                    "assert 'value = 2' in Path('generated.c').read_text()",
                ]})
                delta = {"tool_calls": [check, call("record_implementation", {"decisions": [{
                    "id": "module", "title": "模块示例的配置与接入",
                    "requirement": "读取手册与工程，创建模块并检查。",
                    "rationale": "将示例值写入 `generated.c`，保留规格来源，运行主机断言。",
                    "code_refs": [{"path": "generated.c", "start_line": 1, "end_line": 1}],
                    "evidence_refs": [{"document_id": DOCUMENT, "chunk_id": CHUNK}],
                    "check_call_ids": [check["id"]],
                    "limitations": ["仅做主机断言，未做厂商编译或板卡验证。"],
                }]})]}
            else:
                delta, reason = {"content": f"## 已完成\n\n已创建 generated.c，主机检查通过。\n\n"
                                 f"```c\nint value = 2;\n```\n\n"
                                 f"来源 document={DOCUMENT} chunk={CHUNK} page=113"}, "stop"
            for index, item in enumerate(delta.get("tool_calls", [])):
                item["index"] = index
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                self.event({"choices": [{"delta": {"reasoning_content":
                    "先检查当前工程和规格依据，再根据工具结果继续。"}, "finish_reason": None}]})
                self.event({"choices": [{"delta": delta, "finish_reason": None}]})
                if "等待取消" in question:
                    for _ in range(300):
                        time.sleep(0.1)
                        self.wfile.write(b": waiting\n\n")
                        self.wfile.flush()
                self.event({"choices": [{"delta": {}, "finish_reason": reason}]})
                self.event({"choices": [], "usage": {"prompt_tokens": 30,
                            "completion_tokens": 5, "total_tokens": 35}})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # Expected when the real client cancels its HTTP stream.

        def event(self, payload):
            self.wfile.write(("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode())
            self.wfile.flush()

    return Stub


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8082)
    port = parser.parse_args().port
    with tempfile.TemporaryDirectory(prefix="agent-web-browser-") as directory:
        workspace = Path(directory)
        (workspace / "seed.c").write_text("int seed = 2;\n")
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler(workspace))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_port}"
        options = cli.build_parser().parse_args([
            "--workspace", str(workspace), "--state-dir", str(workspace / ".agent-runtime"),
            "--base-url", base, "--rag-url", base, "--api-key", "offline",
            "--rag-key", "offline", "--model", "deepseek-flash", "--thinking", "enabled",
            "--reasoning-effort", "high",
        ])
        ui = Path(__file__).resolve().parents[2] / "web-next" / "dist"
        try:
            uvicorn.run(create_app(options, ui_directory=ui), host="127.0.0.1", port=port)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
