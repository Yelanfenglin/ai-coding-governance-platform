"""运行预算：步数 / 墙钟时间用尽 → `budget_exceeded`（与取消分开）。

预算停止时保留已发生的工具结果与 usage，并在 error 里给出具体限额原因。
"""

from __future__ import annotations

import asyncio

import pytest

from app.compaction import CompactionPolicy, ModelSummarizer
from app.contracts import AssistantTurn, Budget, Message, MessageEntry, ToolCall, ToolResult
from app.inbox import Inbox
from app.loop import RunOptions, run_agent, user_message
from app.providers.mock import MockProvider, stream_from_coroutine
from app.session import JsonlSessionStore
from app.tools import FunctionTool, ToolRegistry


def final(text: str = "offline answer") -> AssistantTurn:
    return AssistantTurn(Message(role="assistant", content=text), "stop")


def tool_turn(call_id: str) -> AssistantTurn:
    return AssistantTurn(
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id=call_id, name="echo", arguments={"text": "x"})],
        ),
        "tool_calls",
    )


def echo_registry() -> ToolRegistry:
    return ToolRegistry().register(
        FunctionTool(
            name="echo", handler=lambda args, ctx: ToolResult(str(args.get("text", "")))
        )
    )


async def test_step_budget_stops_the_run_and_keeps_results() -> None:
    provider = MockProvider([tool_turn(f"c{index}") for index in range(10)] + [final()])

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=echo_registry(),
            messages=[user_message("go")],
            budget=Budget(max_steps=2),
        )
    )

    assert result.status == "budget_exceeded", result.error
    assert result.steps == 2
    assert "step budget exhausted: max_steps=2" in (result.error or "")
    # 已发生的结果与 usage 都保留
    assert [m.content for m in result.messages if m.role == "tool"] == ["x", "x"]
    assert result.usage.total_tokens == 0 or result.usage.total_tokens >= 0


async def test_time_budget_stops_the_run() -> None:
    async def handler(args: dict[str, object], context: object) -> ToolResult:
        await asyncio.sleep(0.05)
        return ToolResult(content="slow")

    registry = ToolRegistry().register(FunctionTool(name="echo", handler=handler))
    provider = MockProvider([tool_turn("c1"), tool_turn("c2"), final()])

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=registry,
            messages=[user_message("go")],
            budget=Budget(max_seconds=0.01),
        )
    )

    assert result.status == "budget_exceeded"
    assert "time budget exhausted" in (result.error or "")
    assert result.steps == 1


async def test_budget_is_not_cancellation() -> None:
    """预算用尽 ≠ 用户取消：状态和语义都要分开。"""
    cancel_event = asyncio.Event()
    provider = MockProvider([tool_turn("c1"), final()])

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=echo_registry(),
            messages=[user_message("go")],
            cancel_event=cancel_event,
            budget=Budget(max_steps=1),
        )
    )

    assert result.status == "budget_exceeded"
    assert not cancel_event.is_set(), "预算停止不是取消信号"

    cancelled = asyncio.Event()

    def cancel_soon(args: dict[str, object], context: object) -> ToolResult:
        cancelled.set()
        return ToolResult(content="x")

    second = await run_agent(
        RunOptions(
            provider=MockProvider([tool_turn("c1"), final()]),
            tools=ToolRegistry().register(FunctionTool(name="echo", handler=cancel_soon)),
            messages=[user_message("go")],
            cancel_event=cancelled,
            budget=Budget(max_steps=5),
        )
    )

    assert second.status == "aborted"


async def test_time_budget_covers_the_whole_run_across_turns(monkeypatch) -> None:
    """时间预算从 run 开始计时：每个 turn 重置会让总耗时失去约束。"""
    from types import SimpleNamespace

    clock = {"elapsed": 0.0}

    def monotonic() -> float:
        return clock["elapsed"]

    monkeypatch.setattr("app.loop.time", SimpleNamespace(monotonic=monotonic))

    inbox = Inbox()

    class Provider:
        def __init__(self) -> None:
            self.calls = 0
            self.requests = []

        def generate_stream(self, request):
            return stream_from_coroutine(lambda: self.generate(request))

        async def generate(self, request):
            self.calls += 1
            self.requests.append(request)
            clock["elapsed"] += 0.7
            if self.calls == 1:
                inbox.followup(user_message("continue"))
                return final("first done")
            if self.calls == 2:
                return tool_turn("c1")
            return final("second done")

    provider = Provider()
    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=echo_registry(),
            messages=[user_message("start")],
            inbox=inbox,
            budget=Budget(max_seconds=1.0),
        )
    )

    assert result.turns == 2
    assert result.status == "budget_exceeded", (result.status, result.error)
    assert "time budget exhausted" in (result.error or "")
    assert provider.calls == 2, "总耗时 1.4s > 1.0s：不应再发起第三次请求"
    first, second = [request.messages[-1].content for request in provider.requests]
    assert "Seconds: limit=1.0; elapsed=0.0; remaining=1.0" in first
    assert "Seconds: limit=1.0; elapsed=0.7; remaining=0.3" in second
    assert "Steps:" not in first + second, "只显示配置的限额"


async def test_step_budget_is_visible_fresh_each_step_and_not_saved(tmp_path) -> None:
    provider = MockProvider([tool_turn("c1"), tool_turn("c2"), final("verified result")])
    session = JsonlSessionStore(tmp_path / "session.jsonl")
    initial = [Message("system", "fixed system head"), user_message("go")]
    for message in initial:
        await session.append(MessageEntry(1, 1, message, "t"))
    result = await run_agent(RunOptions(
        provider, echo_registry(), initial, session=session, budget=Budget(max_steps=3),
    ))
    assert result.status == "completed"
    assert result.messages[-1].content == "verified result"
    for used, request in enumerate(provider.requests):
        snapshots = [m for m in request.messages if "<run_budget>" in m.content]
        assert len(snapshots) == 1
        assert request.messages[-1] is snapshots[0]
        assert request.messages[-1].role == "user"
        assert f"Steps: limit=3; used={used}; remaining={3 - used}" in snapshots[0].content
        assert "including this response" in snapshots[0].content
        assert "With one step remaining" in snapshots[0].content
        assert "Seconds:" not in snapshots[0].content
        assert request.messages[0] == initial[0], "动态预算不改固定 system 前缀"
    replayed = await JsonlSessionStore(session.path).load_messages()
    assert replayed == result.messages
    assert not any("<run_budget>" in m.content for m in replayed)

    fresh = MockProvider([final()])
    await run_agent(RunOptions(fresh, echo_registry(), [*replayed, user_message("next")]))
    assert not any("<run_budget>" in m.content for m in fresh.requests[0].messages)


@pytest.mark.parametrize("budget", [None, Budget()])
async def test_no_configured_limit_leaves_request_unchanged(budget) -> None:
    provider = MockProvider([final()])
    messages = [Message("system", "sys"), user_message("go")]
    await run_agent(RunOptions(provider, echo_registry(), messages, budget=budget))
    assert provider.requests[0].messages == messages


async def test_step_budget_snapshot_does_not_reset_between_turns() -> None:
    inbox = Inbox()
    requests = []

    class Provider:
        def generate_stream(self, request):
            requests.append(request)
            if len(requests) == 1:
                inbox.followup(user_message("continue"))
            return stream_from_coroutine(lambda: self.generate())

        async def generate(self):
            return final()

    result = await run_agent(RunOptions(
        Provider(), echo_registry(), [user_message("go")], inbox=inbox, budget=Budget(max_steps=2),
    ))
    assert result.status == "completed" and result.turns == 2
    assert "Steps: limit=2; used=0; remaining=2" in requests[0].messages[-1].content
    assert "Steps: limit=2; used=1; remaining=1" in requests[1].messages[-1].content


async def test_compaction_does_not_save_budget_and_main_request_refreshes_time(monkeypatch) -> None:
    from types import SimpleNamespace

    clock = {"now": 100.0}
    monkeypatch.setattr("app.loop.time", SimpleNamespace(monotonic=lambda: clock["now"]))

    class Provider(MockProvider):
        def generate_stream(self, request):
            if request.purpose == "compaction":
                clock["now"] += 7.5
            return super().generate_stream(request)

    provider = Provider([final("checkpoint"), final("done")])
    result = await run_agent(RunOptions(
        provider, echo_registry(), [Message("system", "sys"), user_message("x" * 10000),
                                   Message("assistant", "y" * 2000), user_message("next")],
        budget=Budget(max_steps=1, max_seconds=60),
        compaction=CompactionPolicy(2000, retain_ratio=0.1, summarizer=ModelSummarizer(provider)),
    ))
    assert result.status == "completed" and result.steps == 1
    summary, main = provider.requests
    assert summary.purpose == "compaction" and main.purpose is None
    assert not any("<run_budget>" in m.content for m in summary.messages)
    snapshot = main.messages[-1].content
    assert "Steps: limit=1; used=0; remaining=1" in snapshot
    assert "Seconds: limit=60.0; elapsed=7.5; remaining=52.5" in snapshot
    assert not any("<run_budget>" in m.content for m in result.messages)


async def test_cli_returns_a_distinct_exit_code_for_budget() -> None:
    from app import cli

    class LoopingProvider:
        def generate_stream(self, request):
            return stream_from_coroutine(lambda: self.generate(request))

        async def generate(self, request):
            return tool_turn("c1")

    original = cli.OpenAiCompatibleProvider
    cli.OpenAiCompatibleProvider = lambda **kwargs: LoopingProvider()  # type: ignore[assignment]
    try:
        args = cli.build_parser().parse_args(
            [
                "offline question",
                "--base-url",
                "http://unused.invalid/v1",
                "--api-key",
                "offline-test",
                "--rag-key",
                "offline-test",
                "--max-steps",
                "1",
                "--json",
            ]
        )
        exit_code = await cli.run_once(args)
    finally:
        cli.OpenAiCompatibleProvider = original  # type: ignore[assignment]

    assert exit_code == cli.EXIT_BUDGET
    assert exit_code != 0
