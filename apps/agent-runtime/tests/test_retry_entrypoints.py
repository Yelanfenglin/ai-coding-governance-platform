"""Exercise CLI/Web retry assembly through the default HTTP/SSE transport."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from app import cli
from app.contracts import AssistantReasoningDelta, AssistantTextDelta, ProviderRetryScheduled


@pytest.fixture
def model_server():
    script, requests = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            reply = script.pop(0) if script else {"status": 422}
            status = reply.get("status", 200)
            payload = reply.get("payload", {"error": {"message": "injected failure"}})
            data = json.dumps(payload)
            if status == 200:
                data = f"data: {data}\n\n"
                if not reply.get("truncate") and not reply.get("no_done"):
                    data += "data: [DONE]\n\n"
            encoded = data.encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream" if status == 200
                             else "application/json")
            self.send_header("Content-Length", str(len(encoded) + bool(reply.get("truncate"))))
            for name, value in reply.get("headers", {}).items():
                self.send_header(name, value)
            self.end_headers()
            if reply.get("stall"):
                if reply.get("send_before_stall"):
                    self.wfile.write(encoded)
                    self.wfile.flush()
                time.sleep(reply["stall"])
                return
            self.wfile.write(encoded)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01))
    thread.start()
    try:
        yield SimpleNamespace(url=f"http://127.0.0.1:{server.server_port}/v1",
                              script=script, requests=requests)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def short_model_timeout(monkeypatch):
    from app.providers import openai_compatible

    original = openai_compatible.httpx_stream_transport
    monkeypatch.setattr(openai_compatible, "httpx_stream_transport", lambda: original(timeout=0.1))


def completion(text="done", *, tool=None):
    delta = {"content": text}
    if tool is not None:
        name, arguments = tool
        delta = {"tool_calls": [{
            "index": 0, "id": "call-1", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}
    return {"payload": {"choices": [{
        "delta": delta, "finish_reason": "stop" if tool is None else "tool_calls",
    }], "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}}}


def options(server, tmp_path, *extra):
    return cli.build_parser().parse_args([
        "question", "--base-url", server.url, "--api-key", "local-fault-injection",
        "--model", "deepseek-flash", "--state-dir", str(tmp_path / "state"),
        "--no-memory", "--no-compaction", "--json", *extra,
    ])


@pytest.mark.parametrize("status", [429, 503])
async def test_cli_recovers_from_a_transient_http_failure(model_server, tmp_path, status):
    model_server.script.extend([{"status": status}, completion()])
    events, responses = [], []
    code = await cli.run_once(
        options(model_server, tmp_path), console=False, event_sink=events.append,
        result_sink=lambda response, _: responses.append(response),
    )
    assert code == cli.EXIT_OK
    assert responses[0]["answer"] == "done"
    assert responses[0]["usage"]["total_tokens"] == 14
    assert len(model_server.requests) == 2
    assert model_server.requests[0] == model_server.requests[1]
    assert len([e for e in events if isinstance(e, ProviderRetryScheduled)]) == 1


@pytest.mark.parametrize(("reply", "expected_code"), [
    ({"stall": 0.3}, "TIMEOUT"),
    ({"payload": {}, "truncate": True}, "TRANSPORT"),
    ({"payload": {}, "no_done": True}, "STREAM_CLOSED"),
    (completion(""), "EMPTY_RESPONSE"),
    ({"payload": {"choices": [{"delta": {"reasoning_content": "discarded reasoning"},
                               "finish_reason": "stop"}]}}, "EMPTY_RESPONSE"),
    ({"payload": {"choices": [{"delta": {"reasoning_content": "discarded reasoning"}}]},
      "truncate": True, "stall": 0.3, "send_before_stall": True}, "TIMEOUT"),
])
async def test_cli_recovers_from_empty_or_interrupted_stream(
    model_server, tmp_path, short_model_timeout, reply, expected_code,
):
    model_server.script.extend([reply, completion()])
    events, responses = [], []
    session = tmp_path / "session.jsonl"
    code = await cli.run_once(
        options(model_server, tmp_path, "--session", str(session)), console=False,
        event_sink=events.append, result_sink=lambda response, _: responses.append(response),
    )
    assert code == cli.EXIT_OK
    assert responses[0]["answer"] == "done"
    assert len(model_server.requests) == 2
    assert model_server.requests[0] == model_server.requests[1]
    assert [e.code for e in events if isinstance(e, ProviderRetryScheduled)] == [expected_code]
    assert "discarded reasoning" not in session.read_text(encoding="utf-8")
    if "reasoning_content" in json.dumps(reply):
        assert [e.text for e in events if isinstance(e, AssistantReasoningDelta)] == [
            "discarded reasoning",
        ]


@pytest.mark.parametrize("status", [401, 402, 422])
async def test_cli_does_not_retry_configuration_errors(model_server, tmp_path, status):
    model_server.script.extend([{"status": status}, completion("must not be requested")])
    code = await cli.run_once(options(model_server, tmp_path), console=False)
    assert code == cli.EXIT_FAILED
    assert len(model_server.requests) == 1


async def test_cli_cancels_during_retry_after_without_another_request(model_server, tmp_path):
    model_server.script.extend([
        {"status": 429, "headers": {"Retry-After": "2"}}, completion(),
    ])
    cancelled = asyncio.Event()
    scheduled, responses = [], []

    def receive(event):
        if isinstance(event, ProviderRetryScheduled):
            scheduled.append(event)
            cancelled.set()

    code = await asyncio.wait_for(cli.run_once(
        options(model_server, tmp_path), console=False, event_sink=receive,
        cancel_event=cancelled, result_sink=lambda response, _: responses.append(response),
    ), timeout=1)
    assert code == cli.EXIT_FAILED
    assert responses[0]["status"] == "aborted"
    assert len(scheduled) == 1 and scheduled[0].delay == 2
    assert len(model_server.requests) == 1


async def test_cli_does_not_replay_a_partially_visible_http_stream(model_server, tmp_path):
    model_server.script.extend([
        {**completion("partial answer"), "truncate": True}, completion("must not be requested"),
    ])
    events = []
    code = await cli.run_once(options(model_server, tmp_path), console=False,
                              event_sink=events.append)
    assert code == cli.EXIT_FAILED
    assert len(model_server.requests) == 1
    assert [e.text for e in events if isinstance(e, AssistantTextDelta)] == ["partial answer"]
    assert not any(isinstance(e, ProviderRetryScheduled) for e in events)


@pytest.mark.parametrize("fault", [
    {"status": 503}, {"stall": 0.3}, {"payload": {}, "no_done": True},
    {"payload": {"choices": [{"delta": {"reasoning_content": "discarded reasoning"}}]},
     "truncate": True, "stall": 0.3, "send_before_stall": True},
])
async def test_cli_retry_does_not_repeat_an_executed_command(
    model_server, tmp_path, short_model_timeout, fault,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    model_server.script.extend([
        completion(tool=("run_command", {"command": [sys.executable, "-c", (
            "from pathlib import Path; p=Path('effects.txt'); "
            "p.write_text(p.read_text()+'ran\\n' if p.exists() else 'ran\\n')"
        )]})),
        fault, completion(),
    ])
    code = await cli.run_once(options(model_server, tmp_path, "--workspace", str(workspace)),
                              console=False)
    assert code == cli.EXIT_OK
    assert (workspace / "effects.txt").read_text() == "ran\n"
    assert len(model_server.requests) == 3
    assert model_server.requests[1] == model_server.requests[2]
    results = [m for m in model_server.requests[2]["messages"] if m["role"] == "tool"]
    assert len(results) == 1 and results[0]["tool_call_id"] == "call-1"


async def test_cli_subagent_uses_the_same_retry_policy(model_server, tmp_path):
    model_server.script.extend([
        completion(tool=("subagent", {"prompt": "Return a short review"})),
        {"status": 503}, completion("child review"), completion("parent answer"),
    ])
    responses = []
    code = await cli.run_once(
        options(model_server, tmp_path, "--subagents"), console=False,
        result_sink=lambda response, _: responses.append(response),
    )
    assert code == cli.EXIT_OK
    assert len(model_server.requests) == 4
    assert model_server.requests[1] == model_server.requests[2]
    assert responses[0]["subagents"][0]["status"] == "completed"
    assert responses[0]["subagents"][0]["answer"] == "child review"


async def test_web_recovers_and_forwards_the_retry_event(model_server, tmp_path):
    pytest.importorskip("fastapi")
    from app.web import API_PREFIX, create_app

    model_server.script.extend([{"status": 503}, completion()])
    app = create_app(options(model_server, tmp_path, "--workspace", str(tmp_path)))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                         base_url="http://127.0.0.1") as client,
    ):
        conversation = await client.post(f"{API_PREFIX}/conversations", json={
            "title": "retry acceptance", "project_id": "local",
        })
        created = await client.post(f"{API_PREFIX}/tasks", json={
            "conversation_id": conversation.json()["id"], "prompt": "question",
            "idempotency_key": str(uuid4()),
        })
        assert created.status_code == 201, created.text
        identifier = created.json()["id"]
        await asyncio.wait_for(app.state.agent.jobs[identifier].runner, timeout=5)
        record = app.state.agent.jobs[identifier].record
        assert record["status"] == "completed" and record["response"]["answer"] == "done"
        assert len(model_server.requests) == 2
        stream = await client.get(f"{API_PREFIX}/tasks/{identifier}/events")
        events = [json.loads(line[6:]) for line in stream.text.splitlines()
                  if line.startswith("data: ")]
        names = [event["type"] for event in events]
        assert names.index("ProviderRetryScheduled") < names.index("ProviderRetryStarted")
        assert names[-1] == "TaskFinished"
