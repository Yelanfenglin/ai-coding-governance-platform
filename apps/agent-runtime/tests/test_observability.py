"""SDK-level checks of trace ownership, async stream closure and delivery outcomes."""

from __future__ import annotations

import asyncio
import json

import pytest
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from app import cli, observability
from app.contracts import (
    AssistantTurn,
    Message,
    ModelRequest,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Usage,
)
from app.errors import ErrorCode
from app.loop import RunOptions, run_agent, user_message
from app.providers.mock import MockProvider
from app.providers.openai_compatible import OpenAiCompatibleProvider, TransportError
from app.retry import RetryPolicy
from app.subagents import SubagentRunner
from app.tools import FunctionTool, ToolRegistry


@pytest.fixture
def spans(monkeypatch):
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(observability, "tracer", provider.get_tracer("test"))
    yield exporter
    provider.shutdown()


def sse(payload):
    return "data: " + json.dumps(payload) + "\n\n"


async def completion(_request):
    yield sse({"id": "response-1", "model": "actual-flash", "choices": [
        {"delta": {"reasoning_content": "PRIVATE_REASONING"}},
    ]})
    yield sse({"choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}]})
    yield sse({"choices": [], "usage": {
        "prompt_tokens": 100, "completion_tokens": 12, "total_tokens": 112,
        "prompt_cache_hit_tokens": 80, "prompt_cache_miss_tokens": 20,
        "completion_tokens_details": {"reasoning_tokens": 9},
    }})
    yield "data: [DONE]\n\n"


async def test_sse_usage_and_parent_survive_different_next_and_close_tasks(spans, caplog):
    provider = OpenAiCompatibleProvider("http://unused", "requested-flash",
                                          stream_transport=completion,
                                          thinking="enabled", reasoning_effort="high")
    with observability.observe("task") as parent:
        iterator = provider.generate_stream(ModelRequest([user_message("PRIVATE_PROMPT")], []))
        while True:
            try:
                await asyncio.create_task(anext(iterator))
            except StopAsyncIteration:
                break
            assert trace.get_current_span() is parent
        await asyncio.create_task(iterator.aclose())
    model = next(s for s in spans.get_finished_spans() if s.name == "llm.request")
    assert model.parent.span_id == parent.get_span_context().span_id
    assert model.attributes["gen_ai.response.model"] == "actual-flash"
    assert model.attributes["llm.token_count.prompt"] == 100
    assert model.attributes["llm.token_count.prompt_details.cache_read"] == 80
    assert model.attributes["gen_ai.usage.uncached_input_tokens"] == 20
    assert model.attributes["llm.token_count.completion_details.reasoning"] == 9
    assert model.attributes["llm.first_text_ms"] >= model.attributes["llm.first_chunk_ms"]
    assert "PRIVATE" not in str(model.attributes)
    assert "Failed to detach context" not in caplog.text


async def test_cancel_closes_the_provider_span_without_changing_parent_context(spans):
    entered, closed = asyncio.Event(), asyncio.Event()

    async def blocked(_request):
        try:
            entered.set()
            await asyncio.Event().wait()
            yield "unreachable"
        finally:
            closed.set()

    provider = OpenAiCompatibleProvider("http://unused", "flash", stream_transport=blocked)
    with observability.observe("task") as parent:
        iterator = provider.generate_stream(ModelRequest([], []))
        pending = asyncio.create_task(anext(iterator))
        await entered.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await asyncio.create_task(iterator.aclose())
        assert closed.is_set() and trace.get_current_span() is parent
    model = next(s for s in spans.get_finished_spans() if s.name == "llm.request")
    assert model.attributes["agent.outcome"] == "interrupted"


async def test_retry_attempts_are_separate_and_usage_is_not_duplicated(spans):
    calls = 0

    async def transport(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TransportError("busy", ErrorCode.RATE_LIMIT)
        async for part in completion(request):
            yield part

    provider = OpenAiCompatibleProvider("http://unused", "flash", stream_transport=transport)
    result = await run_agent(RunOptions(provider, ToolRegistry(), [user_message("task")],
                                       retry_policy=RetryPolicy(base_delay=0, jitter=False)))
    requests = [s for s in spans.get_finished_spans() if s.name == "llm.request"]
    assert result.status == "completed" and len(requests) == 2
    assert requests[0].status.status_code == StatusCode.ERROR
    assert requests[1].attributes["llm.token_count.total"] == 112
    assert result.usage.total_tokens == 112 and result.usage.cached_input_tokens == 80


async def test_parallel_tools_and_real_subagent_keep_their_parent_relationships(spans, tmp_path):
    entered = 0
    both = asyncio.Event()

    async def lookup(args, context):
        nonlocal entered
        entered += 1
        if entered == 2:
            both.set()
        await asyncio.wait_for(both.wait(), 1)
        return ToolResult("fact")

    registry = ToolRegistry().register(FunctionTool(
        "lookup", handler=lookup, execution_mode="parallel",
    ))
    child = MockProvider([
        AssistantTurn(Message("assistant", tool_calls=[
            ToolCall("a", "lookup", {}), ToolCall("b", "lookup", {}),
        ]), "tool_calls"),
        AssistantTurn(Message("assistant", "child done"), "stop"),
    ])
    runner = SubagentRunner(lambda: RunOptions(child, registry, []), tmp_path / "children")
    parent = MockProvider([
        AssistantTurn(Message("assistant", tool_calls=[
            ToolCall("delegate", "subagent", {"prompt": "lookup"}),
        ]), "tool_calls"),
        AssistantTurn(Message("assistant", "done"), "stop"),
    ])
    try:
        result = await run_agent(RunOptions(parent, ToolRegistry().register(runner.tool()),
                                           [user_message("task")]))
    finally:
        await runner.close()
    assert result.status == "completed"
    recorded = spans.get_finished_spans()
    delegate = next(s for s in recorded if s.name == "tool.subagent")
    child_run = next(s for s in recorded if s.name == "agent.run"
                     and s.parent and s.parent.span_id == delegate.context.span_id)
    tools = [s for s in recorded if s.name == "tool.lookup"]
    assert len(tools) == 2 and all(s.parent.span_id == child_run.context.span_id for s in tools)
    assert max(s.start_time for s in tools) < min(s.end_time for s in tools)


async def test_caller_supplied_registry_is_observed_at_the_execution_boundary(spans):
    class PluginRegistry:
        def list(self):
            return [ToolDefinition("plugin")]

        def execution_mode(self, name):
            return "parallel"

        async def execute(self, call, context):
            return ToolResult("plugin result")

    provider = MockProvider([
        AssistantTurn(Message("assistant", tool_calls=[ToolCall("p", "plugin", {})]),
                      "tool_calls"),
        AssistantTurn(Message("assistant", "done"), "stop"),
    ])
    result = await run_agent(RunOptions(provider, PluginRegistry(), [user_message("task")]))
    assert result.status == "completed"
    plugin = next(s for s in spans.get_finished_spans() if s.name == "tool.plugin")
    assert plugin.attributes["tool_call.id"] == "p" and not plugin.attributes["tool.is_error"]


async def test_task_trace_includes_host_checks_and_keeps_both_outcomes(
    spans, tmp_path, monkeypatch,
):
    config = tmp_path / "check.json"
    config.write_text(json.dumps({"command": ["{python}", "-c", "raise SystemExit(1)"]}))
    args = cli.build_parser().parse_args([
        "task", "--base-url", "http://unused", "--workspace", str(tmp_path),
        "--verification", str(config), "--no-memory", "--no-compaction",
        "--delivery-dir", str(tmp_path / ".agent-runtime" / "delivery"),
    ])
    provider = MockProvider([AssistantTurn(Message("assistant", "done"), "stop")])
    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **_: provider)
    responses = []
    code = await cli.run_once(args, console=False, result_sink=lambda row, _: responses.append(row),
                              task_id="task-1", conversation_id="conversation-1")
    task = next(s for s in spans.get_finished_spans() if s.name == "agent.task")
    check = next(s for s in spans.get_finished_spans() if s.name == "project.verify")
    assert code == cli.EXIT_VERIFICATION and task.status.status_code == StatusCode.ERROR
    assert task.attributes["agent.status"] == "completed"
    assert task.attributes["verification.status"] == "failed"
    assert task.start_time <= check.start_time < check.end_time <= task.end_time
    assert responses[0]["trace_id"] == f"{task.context.trace_id:032x}"
    delivery = json.loads((tmp_path / ".agent-runtime/delivery/delivery.json").read_text())
    assert delivery["response"]["trace_id"] == responses[0]["trace_id"]


def test_usage_keeps_missing_cache_data_unknown():
    usage = Usage()
    usage.add(Usage(input_tokens=5, total_tokens=5))
    assert usage.cached_input_tokens is None
    usage.add(Usage(input_tokens=10, total_tokens=10, cached_input_tokens=0))
    assert usage.cached_input_tokens == 0 and usage.total_tokens == 15


async def test_user_cancellation_is_separate_from_failure(spans, tmp_path, monkeypatch):
    cancel = asyncio.Event()
    cancel.set()
    args = cli.build_parser().parse_args([
        "task", "--base-url", "http://unused", "--workspace", str(tmp_path),
        "--no-memory", "--no-compaction",
    ])
    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **_: MockProvider([]))
    code = await cli.run_once(args, cancel_event=cancel, console=False)
    task = next(s for s in spans.get_finished_spans() if s.name == "agent.task")
    assert code == cli.EXIT_FAILED and task.attributes["agent.status"] == "aborted"
    assert task.status.status_code != StatusCode.ERROR


async def test_real_otlp_failure_does_not_fail_or_retry_the_agent(monkeypatch):
    pytest.importorskip("opentelemetry.sdk")
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    delivered = []

    class UnavailableCollector(BaseHTTPRequestHandler):
        def do_POST(self):
            delivered.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(503)
            self.end_headers()

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), UnavailableCollector)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    provider = TracerProvider(shutdown_on_exit=False)
    exporter = OTLPSpanExporter(endpoint=f"http://127.0.0.1:{server.server_port}/v1/traces",
                                timeout=0.2)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    monkeypatch.setattr(observability, "tracer", provider.get_tracer("test"))
    try:
        llm = MockProvider([AssistantTurn(Message("assistant", "done"), "stop")])
        result = await run_agent(RunOptions(llm, ToolRegistry(), [user_message("task")]))
        assert result.status == "completed" and len(llm.requests) == 1
        await asyncio.to_thread(provider.force_flush, 1000)
        assert delivered, "must exercise the real OTLP exporter over HTTP"
    finally:
        await asyncio.to_thread(provider.shutdown)
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        thread.join(timeout=1)


def test_global_sdk_does_not_turn_ui_polling_into_task_traces():
    pytest.importorskip("opentelemetry.sdk")
    import os
    import subprocess
    import sys

    # A separate process allows the SDK's real, once-only global installation.
    # FastAPI's native telemetry is activated by that global SDK, not by the
    # per-test tracer used above. This reproduces the real HTTP acceptance gap.
    script = '''
import asyncio, tempfile
import httpx
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from app import cli
from app.web import create_app
exporter = InMemorySpanExporter()
provider = TracerProvider(shutdown_on_exit=False)
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)
async def check(directory):
    args = cli.build_parser().parse_args([
        '--workspace', directory, '--base-url', 'http://unused',
    ])
    app = create_app(args)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app),
                                     base_url='http://127.0.0.1') as client:
            for _ in range(3):
                assert (await client.get('/api/agent/health')).json()['ready']
                assert (await client.get('/api/agent/projects')).status_code == 200
    assert not exporter.get_finished_spans(), [s.name for s in exporter.get_finished_spans()]
with tempfile.TemporaryDirectory() as directory:
    asyncio.run(check(directory))
provider.shutdown()
'''
    env = {key: value for key, value in os.environ.items() if not key.startswith("OTEL_")}
    env["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run([sys.executable, "-c", script], env=env,
                               capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stderr
