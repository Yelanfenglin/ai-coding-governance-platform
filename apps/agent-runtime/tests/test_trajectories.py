"""Phase G：Trajectory 评测。

用**会校验 tool-call / result 配对**的 strict provider 替身跑固定脚本，
对每条轨迹断言事件顺序、工具次数与顺序、终止原因、usage 累加。

覆盖：多步工具、parallel/exclusive 屏障、retry、两种取消入口、压缩后继续、记忆范围与预算。
不调用真实模型，也不把离线替身的结果写成真实模型效果。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.compaction import CompactionPolicy
from app.contracts import (
    AssistantTurn,
    Message,
    ModelRequest,
    ToolCall,
    ToolResult,
    Usage,
)
from app.loop import RunOptions, run_agent, system_message, user_message
from app.memory import JsonMemoryStore, MemoryMetadata
from app.providers.mock import stream_from_coroutine
from app.tools import FunctionTool, ToolRegistry
from app.trajectory import trajectory_recorder


class StrictProvider:
    """固定脚本 provider，并校验每次请求的消息结构。

    校验规则（OpenAI/DeepSeek 的硬性要求）：
    - 每条 `tool` 消息必须能对应到它前面某条 assistant 消息的 tool_call id；
    - 除最后一条 assistant 消息外，每个 tool_call 都必须在请求里有对应结果。
    """

    def __init__(self, turns: list[AssistantTurn]) -> None:
        self._turns = list(turns)
        self.requests: list[ModelRequest] = []

    def generate_stream(self, request):
        return stream_from_coroutine(lambda: self.generate(request))

    async def generate(self, request: ModelRequest) -> AssistantTurn:
        self.requests.append(request)
        self.validate(request.messages)
        index = len(self.requests) - 1
        if index >= len(self._turns):
            raise AssertionError(f"strict provider exhausted after {len(self._turns)} turn(s)")
        return self._turns[index]

    @staticmethod
    def validate(messages: list[Message]) -> None:
        issued: dict[str, str] = {}
        answered: set[str] = set()
        for message in messages:
            for call in message.tool_calls:
                issued[call.id] = call.name
            if message.role == "tool":
                if message.tool_call_id is None:
                    raise AssertionError("tool message without tool_call_id")
                if message.tool_call_id not in issued:
                    raise AssertionError(
                        f"tool result {message.tool_call_id!r} has no matching assistant call"
                    )
                answered.add(message.tool_call_id)
        # 只有「请求里最后一条消息发起的、且后面没有任何消息」的调用允许还没有结果
        # （正要去执行）。一旦后面又出现了别的消息，未回答的旧调用就是结构错误。
        owner_index: dict[str, int] = {}
        for index, message in enumerate(messages):
            for call in message.tool_calls:
                owner_index[call.id] = index
        last_index = len(messages) - 1
        for call_id in set(issued) - answered:
            if owner_index.get(call_id) != last_index:
                raise AssertionError(
                    f"tool call {call_id!r} has no result in the request "
                    f"(issued at {owner_index.get(call_id)}, last message at {last_index})"
                )


def tool_turn(call_id: str, name: str, **arguments: object) -> AssistantTurn:
    return AssistantTurn(
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id=call_id, name=name, arguments=dict(arguments))],
        ),
        "tool_calls",
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )


def multi_turn(calls: list[tuple[str, str]]) -> AssistantTurn:
    return AssistantTurn(
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id=cid, name=name, arguments={}) for cid, name in calls],
        ),
        "tool_calls",
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )


def final(text: str = "done") -> AssistantTurn:
    return AssistantTurn(
        Message(role="assistant", content=text),
        "stop",
        usage=Usage(input_tokens=20, output_tokens=5, total_tokens=25),
    )


def echo_registry(timeline: list[str] | None = None) -> ToolRegistry:
    def make(name: str, mode: str = "exclusive", delay: float = 0.0) -> FunctionTool:
        async def handler(args: dict[str, object], context: object) -> ToolResult:
            if timeline is not None:
                timeline.append(f"{name}:start")
            if delay:
                await asyncio.sleep(delay)
            if timeline is not None:
                timeline.append(f"{name}:end")
            return ToolResult(content=name)

        return FunctionTool(name=name, handler=handler, execution_mode=mode)

    return (
        ToolRegistry()
        .register(make("echo"))
        .register(make("fast", mode="parallel", delay=0.01))
        .register(make("slow", mode="parallel", delay=0.05))
    )


def test_strict_provider_rejects_an_old_unanswered_call() -> None:
    """旧调用没有结果、后面又有别的消息 —— 必须判为结构错误。"""
    transcript = [
        Message("assistant", tool_calls=[ToolCall("missing-result", "echo")]),
        Message("assistant", content="done"),
        user_message("continue"),
    ]
    with pytest.raises(AssertionError):
        StrictProvider.validate(transcript)


def test_strict_provider_allows_the_trailing_call() -> None:
    """最后一条消息刚发起的调用允许还没有结果（正要去执行）。"""
    transcript = [
        user_message("go"),
        Message("assistant", tool_calls=[ToolCall("c1", "echo")]),
    ]
    StrictProvider.validate(transcript)


async def test_multi_step_tool_trajectory() -> None:
    provider = StrictProvider([tool_turn("c1", "echo"), tool_turn("c2", "echo"), final()])
    trajectory, on_event = trajectory_recorder()

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=echo_registry(),
            messages=[system_message("sys"), user_message("go")],
            on_event=on_event,
        )
    )

    trajectory.finish(result)
    assert result.status == "completed"
    assert trajectory.turns == 1
    assert trajectory.steps == 3
    assert trajectory.tool_names == ["echo", "echo"]
    assert not any(record.is_error for record in trajectory.tool_calls)
    assert trajectory.event_types[:4] == [
        "TurnStarted",
        "StepStarted",
        "AssistantTurnEvent",
        "ToolStarted",
    ]
    # usage 累加：15 + 15 + 25（两个 tool step + 一个 final step）
    assert trajectory.usage.total_tokens == 55
    assert result.usage.total_tokens == 55


async def test_parallel_and_exclusive_barrier_trajectory() -> None:
    timeline: list[str] = []
    provider = StrictProvider(
        [multi_turn([("c1", "fast"), ("c2", "slow"), ("c3", "echo")]), final()]
    )

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=echo_registry(timeline),
            messages=[user_message("go")],
        )
    )

    assert result.status == "completed"
    assert len(provider.requests) == 2
    # parallel 先行且并发：slow 尚未结束前 fast 已开始
    assert timeline.index("fast:start") < timeline.index("slow:end")
    # exclusive 是屏障：它必须在两个 parallel 都结束后才开始
    assert timeline.index("echo:start") > max(
        timeline.index("fast:end"), timeline.index("slow:end")
    )


async def test_retry_trajectory() -> None:
    from app.errors import ErrorCode, ProviderError

    class FlakyStrict(StrictProvider):
        def __init__(self) -> None:
            super().__init__([final()])
            self.failed = False

        def generate_stream(self, request):
            return stream_from_coroutine(lambda: self.generate(request))

        async def generate(self, request: ModelRequest) -> AssistantTurn:
            if not self.failed:
                self.failed = True
                raise ProviderError("slow down", ErrorCode.RATE_LIMIT, status_code=429)
            return await super().generate(request)

    provider = FlakyStrict()
    trajectory, on_event = trajectory_recorder()
    from app.retry import RetryPolicy

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=echo_registry(),
            messages=[user_message("go")],
            retry_policy=RetryPolicy(max_retries=2, base_delay=0.001, max_delay=0.01, jitter=False),
            on_event=on_event,
        )
    )

    assert result.status == "completed"
    assert trajectory.retries == 1
    assert "ProviderRetryScheduled" in trajectory.event_types
    assert "ProviderRetryStarted" in trajectory.event_types


async def test_cancel_event_trajectory_has_complete_results() -> None:
    cancel_event = asyncio.Event()

    def handler(args: dict[str, object], context: object) -> ToolResult:
        cancel_event.set()
        return ToolResult(content="probed")

    registry = ToolRegistry().register(FunctionTool(name="echo", handler=handler))
    provider = StrictProvider([multi_turn([("c1", "echo"), ("c2", "echo")]), final()])
    trajectory, on_event = trajectory_recorder()

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=registry,
            messages=[user_message("go")],
            cancel_event=cancel_event,
            on_event=on_event,
        )
    )

    assert result.status == "aborted"
    assert len(trajectory.tool_calls) == 1, "取消后未派发的 call 不产生 ToolStarted"
    assert [m.content for m in result.messages if m.role == "tool"] == [
        "probed",
        "Error: tool call aborted before dispatch",
    ]


async def test_usage_accounting_and_compaction_then_continue(tmp_path: Path) -> None:
    async def summarize(messages: list[Message]) -> str:
        return "SUMMARY: PWM 配置已确认"

    session_path = tmp_path / "session.jsonl"
    from app.session import JsonlSessionStore, MessageEntry

    store = JsonlSessionStore(session_path)
    initial = [system_message("sys"), user_message("goal " * 200)]
    for message in initial:
        await store.append(MessageEntry(0, 0, message, "t"))

    provider = StrictProvider([tool_turn("c1", "echo", text="x" * 400), final("A")])
    trajectory, on_event = trajectory_recorder()

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=echo_registry(),
            messages=initial,
            session=store,
            compaction=CompactionPolicy(100, retain_ratio=0.01, summarizer=summarize),
            on_event=on_event,
        )
    )

    assert result.status == "completed"
    assert trajectory.compactions >= 1
    assert result.usage.total_tokens >= 15 + 25, "压缩请求也计入 usage"

    # 压缩后继续：重放得到的活动上下文 == 运行中的上下文
    replayed = await JsonlSessionStore(session_path).load_messages()
    assert replayed == result.messages


async def test_memory_scope_and_budget_trajectory(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "facts.json", max_chars=200)
    await store.remember(
        "pwm.period", "PR2=0xFF", MemoryMetadata(applies_to={"chip": "PIC16F877A"})
    )
    await store.remember("language", "中文")

    trajectory, on_event = trajectory_recorder()
    result = await run_agent(
        RunOptions(
            provider=StrictProvider([final()]),
            tools=echo_registry(),
            messages=[system_message("sys"), user_message("go")],
            memory=store,
            applies_to={"chip": "PIC16F877A"},
            on_event=on_event,
        )
    )

    assert result.status == "completed"
    assert trajectory.memory_recalls == 1
    assert await store.recall(applies_to={"chip": "PIC18F4550"}) != [], "无关型号仍可召回全局事实"
    scoped = await store.recall(applies_to={"chip": "PIC18F4550"}, query="pwm.period")
    assert [item.key for item in scoped] == [], "带型号的事实不该跨型号召回"
