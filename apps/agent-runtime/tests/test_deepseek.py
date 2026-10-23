"""DeepSeek request settings and reasoning replay through tools/session/compaction."""

from __future__ import annotations

import asyncio
import json

import pytest

from app import cli
from app.compaction import ModelSummarizer, estimate_context, render_transcript
from app.contracts import (
    AssistantTurn,
    Message,
    MessageEntry,
    ModelRequest,
    ReasoningDelta,
    TextDelta,
    ToolResult,
)
from app.errors import ProviderError
from app.loop import RunOptions, run_agent
from app.providers.mock import MockProvider
from app.providers.openai_compatible import OpenAiCompatibleProvider, to_wire_message
from app.session import JsonlSessionStore, message_from_dict
from app.tools import FunctionTool, ToolRegistry


@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_deepseek_settings_are_sent_as_top_level_fields(effort) -> None:
    provider = OpenAiCompatibleProvider(
        "https://api.deepseek.com", "deepseek-flash",
        thinking="enabled", reasoning_effort=effort,
    )
    body = provider._body(ModelRequest([Message("user", "hi")], []))
    assert body["model"] == "deepseek-flash"
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == effort
    assert "extra_body" not in body


def test_disabled_thinking_does_not_send_an_enabling_effort() -> None:
    provider = OpenAiCompatibleProvider(
        "https://api.deepseek.com", "deepseek-flash",
        thinking="disabled", reasoning_effort="high",
    )
    body = provider._body(ModelRequest([Message("user", "hi")], []))
    assert body["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in body


async def test_reasoning_survives_tools_session_reload_and_summary_request(tmp_path) -> None:
    requests = []
    responses = [
        {"reasoning_content": "reason before tool\n", "tool_calls": [{
            "index": 0, "id": "echo-1",
            "function": {"name": "echo", "arguments": '{"text":"ok"}'},
        }]},
        {"reasoning_content": "reason after tool\n", "content": "final answer"},
        {"reasoning_content": "summary reasoning", "content": "checkpoint"},
    ]

    async def transport(request):
        body = json.loads(request.body)
        requests.append(body)
        delta = responses[len(requests) - 1]
        reason = "tool_calls" if "tool_calls" in delta else "stop"
        yield "data: " + json.dumps({"choices": [{
            "delta": delta, "finish_reason": reason,
        }]}) + "\n\n"
        yield "data: [DONE]\n\n"

    provider = OpenAiCompatibleProvider(
        "https://api.deepseek.com", "deepseek-flash", stream_transport=transport,
        thinking="enabled", reasoning_effort="high",
    )
    registry = ToolRegistry().register(FunctionTool(
        name="echo", handler=lambda args, context: ToolResult(args["text"]),
    ))
    session = JsonlSessionStore(tmp_path / "session.jsonl")
    initial = [Message("system", "system head"), Message("user", "echo ok")]
    for message in initial:
        await session.append(MessageEntry(1, 1, message, "t"))
    result = await run_agent(RunOptions(
        provider=provider, tools=registry, messages=initial, session=session,
    ))
    assert result.status == "completed"
    assert result.messages[-1].content == "final answer"
    assert result.messages[-1].reasoning_content == "reason after tool\n"
    call = requests[1]["messages"][-2]
    assert call["content"] == ""
    assert call["reasoning_content"] == "reason before tool\n"
    assert call["tool_calls"][0]["id"] == "echo-1"
    assert requests[1]["messages"][-1]["content"] == "ok"

    restored = await JsonlSessionStore(session.path).load_messages()
    assert restored == result.messages
    summary = await ModelSummarizer(provider)(restored, tools=registry.list())
    assert summary == "checkpoint", "reasoning is not checkpoint text"
    assert requests[2]["messages"][:-1] == [to_wire_message(m) for m in restored]
    assert requests[2]["messages"][-2]["reasoning_content"] == "reason after tool\n"


async def test_cli_wires_flash_thinking_and_effort_to_the_request(
    monkeypatch, tmp_path, capsys,
) -> None:
    for key in ("OPENAI_MODEL", "OPENAI_THINKING", "OPENAI_REASONING_EFFORT"):
        monkeypatch.delenv(key, raising=False)
    seen = []

    async def transport(request):
        seen.append(json.loads(request.body))
        yield (
            'data: {"choices":[{"delta":{"reasoning_content":"private",'
            '"content":"done"},"finish_reason":"stop"}]}\n\n'
        )
        yield "data: [DONE]\n\n"

    original = OpenAiCompatibleProvider
    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **kwargs: original(
        **kwargs, stream_transport=transport,
    ))
    args = cli.build_parser().parse_args([
        "hi", "--base-url", "https://api.deepseek.com", "--api-key", "offline-test",
        "--state-dir", str(tmp_path), "--no-memory", "--no-compaction", "--json",
    ])
    assert await cli.run_once(args) == 0
    assert json.loads(capsys.readouterr().out)["answer"] == "done"
    assert seen[0]["model"] == "deepseek-flash"
    assert seen[0]["thinking"] == {"type": "enabled"}
    assert seen[0]["reasoning_effort"] == "high"

    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "max")
    args = cli.build_parser().parse_args(["hi"])
    assert args.reasoning_effort == "max"
    explicit = cli.build_parser().parse_args(["hi", "--reasoning-effort", "low"])
    assert explicit.reasoning_effort == "low"


def test_reasoning_is_counted_and_legacy_sessions_still_load() -> None:
    message = Message("assistant", "answer", reasoning_content="r" * 400)
    plain = Message("assistant", "answer")
    assert estimate_context([message]) == estimate_context([plain]) + 104
    assert "assistant reasoning: " + "r" * 400 in render_transcript([message])
    assert message_from_dict({"role": "assistant", "content": "old answer"}) == Message(
        "assistant", "old answer",
    )


async def test_cancellation_keeps_partial_reasoning_separate(tmp_path) -> None:
    cancelled = asyncio.Event()

    class Provider:
        async def generate_stream(self, request):
            yield ReasoningDelta(0, "partial reasoning")
            yield TextDelta(1, "partial answer")
            cancelled.set()
            await asyncio.Event().wait()

    session = JsonlSessionStore(tmp_path / "cancel.jsonl")
    result = await run_agent(RunOptions(
        Provider(), ToolRegistry(), [Message("user", "hi")],
        cancel_event=cancelled, session=session,
    ))
    assert result.status == "aborted"
    partial = result.messages[-1]
    assert partial.content == "partial answer"
    assert partial.reasoning_content == "partial reasoning"
    assert partial.interrupted
    assert (await session.load_messages())[-1] == partial


async def test_reasoning_only_output_is_not_a_text_checkpoint() -> None:
    provider = MockProvider([AssistantTurn(
        Message("assistant", reasoning_content="reasoning without a summary"), "stop",
    )])
    with pytest.raises(ProviderError, match="no text summary content"):
        await ModelSummarizer(provider)([Message("user", "conversation")])
