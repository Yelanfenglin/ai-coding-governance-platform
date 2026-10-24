from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from prometheus_client import REGISTRY

from app import cli
from app.contracts import Message, ModelRequest, ToolCall, ToolResult
from app.errors import ErrorCode
from app.loop import RunOptions, run_agent, user_message
from app.observability import record_event
from app.providers.openai_compatible import OpenAiCompatibleProvider, TransportError
from app.retry import RetryPolicy
from app.tools import FunctionTool, ToolRegistry
from app.web import create_app


def sample(name, labels=None):
    return REGISTRY.get_sample_value(name, labels or {}) or 0


async def test_request_metrics_count_retries_and_final_usage_without_exporter():
    model = "metrics-fixture"
    calls = 0

    async def transport(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TransportError("PRIVATE_RESPONSE", ErrorCode.RATE_LIMIT, status_code=429)
        payloads = [
            {"choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
            {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 12,
                                      "prompt_cache_hit_tokens": 80,
                                      "prompt_cache_miss_tokens": 20}},
        ]
        for payload in payloads:
            yield "data: " + json.dumps(payload) + "\n\n"
        yield "data: [DONE]\n\n"

    labels = {"model": model, "purpose": "response", "type": "input"}
    before = sample("agent_model_tokens_total", labels)
    retry_before = sample("agent_model_retries_total", {"code": "RATE_LIMIT"})
    result = await run_agent(RunOptions(
        provider=OpenAiCompatibleProvider("http://unused", model, stream_transport=transport),
        tools=ToolRegistry(), messages=[user_message("PRIVATE_PROMPT")],
        on_event=record_event, retry_policy=RetryPolicy(base_delay=0, jitter=False),
    ))
    assert result.status == "completed" and calls == 2
    assert sample("agent_model_tokens_total", labels) - before == 100
    assert sample("agent_model_retries_total", {"code": "RATE_LIMIT"}) - retry_before == 1
    assert sample("agent_model_requests_total", {
        "model": model, "purpose": "response", "outcome": "error", "code": "RATE_LIMIT",
    }) == 1
    assert REGISTRY.get_sample_value("agent_model_tokens_total", {
        "model": model, "purpose": "response", "type": "reasoning",
    }) is None
    assert sample("agent_model_requests_active") == 0


async def test_cancelled_model_request_releases_active_gauge():
    entered = asyncio.Event()

    async def transport(_):
        entered.set()
        await asyncio.Event().wait()
        yield "unreachable"

    provider = OpenAiCompatibleProvider("http://unused", "cancel-fixture",
                                        stream_transport=transport)
    iterator = provider.generate_stream(ModelRequest([Message("user", "test")], []))
    pending = asyncio.create_task(anext(iterator))
    await entered.wait()
    assert sample("agent_model_requests_active") == 1
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await iterator.aclose()
    assert sample("agent_model_requests_active") == 0
    assert sample("agent_model_requests_total", {
        "model": "cancel-fixture", "purpose": "response", "outcome": "cancelled", "code": "",
    }) == 1


async def test_actual_tool_error_has_one_dispatch_metric():
    async def fail(_, context):
        return ToolResult("not supported", is_error=True)

    registry = ToolRegistry().register(FunctionTool("metrics_failure", handler=fail))
    from app.contracts import AssistantTurn
    from app.providers.mock import MockProvider

    result = await run_agent(RunOptions(
        provider=MockProvider([
            AssistantTurn(Message("assistant", tool_calls=[
                ToolCall("PRIVATE_CALL_ID", "metrics_failure", {}),
            ]), "tool_calls"),
            AssistantTurn(Message("assistant", "reported failure"), "stop"),
        ]), tools=registry, messages=[user_message("test")],
    ))
    assert result.status == "completed"
    assert sample("agent_tool_calls_total", {"tool": "metrics_failure", "outcome": "error"}) == 1
    assert sample("agent_tools_active", {"tool": "metrics_failure"}) == 0


async def test_metrics_endpoint_preserves_host_guard_and_app_factory_isolation(tmp_path):
    args = cli.build_parser().parse_args([
        "--workspace", str(tmp_path), "--base-url", "http://unused",
    ])
    apps = [create_app(args), create_app(args)]
    for application in apps:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(application),
                                     base_url="http://127.0.0.1:8081") as client:
            assert (await client.get("/api/agent/projects")).status_code == 200
            scraped = await client.get("/metrics", headers={"host": "host.docker.internal:8081"})
            assert scraped.status_code == 200
            assert "agent_tasks_active" in scraped.text
            assert 'handler="/api/agent/projects"' in scraped.text
            forbidden = await client.get("/api/agent/projects", headers={
                "host": "host.docker.internal:8081",
            })
            assert forbidden.status_code == 403
            assert (await client.get("/metrics", headers={
                "host": "host.docker.internal:8081", "origin": "https://foreign.invalid",
            })).status_code == 403
            assert (await client.get("/api/agent/health")).status_code == 200
            body = (await client.get("/metrics")).text
            assert 'handler="/metrics"' not in body and 'handler="/api/agent/health"' not in body


async def test_task_completion_and_failed_delivery_are_independent(monkeypatch):
    args = cli.build_parser().parse_args(["test", "--base-url", "http://unused"])
    labels = {"status": "completed", "verification": "failed"}
    before = sample("agent_tasks_total", labels)

    async def rejected(_args, *, result_sink, **kwargs):
        result_sink({"status": "completed", "verification": {"status": "failed"}}, 5)
        return 5

    monkeypatch.setattr(cli, "_run_once", rejected)
    assert await cli.run_once(args, console=False) == 5
    assert sample("agent_tasks_total", labels) - before == 1
    assert sample("agent_tasks_active") == sample("agent_task_started_timestamp_seconds") == 0
