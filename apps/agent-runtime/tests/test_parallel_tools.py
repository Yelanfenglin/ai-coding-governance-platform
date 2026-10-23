"""Phase C：并行工具 + exclusive barrier。

调度语义对齐 DSH `packages/core/agent-loop/src/tool-calls.ts`：
parallel-safe 的连续 call 进有界并发池，exclusive call 是屏障，结果按模型顺序写回，
取消时已开始的 call 结果必须提交、未派发的补 synthetic 结果。
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app import loop as loop_module
from app.cli import _continuation
from app.contracts import (
    AssistantTurn,
    Message,
    MessageEntry,
    SessionEvent,
    StepEndEntry,
    StepStartEntry,
    ToolCall,
    ToolFinished,
    ToolResult,
    TurnEndEntry,
)
from app.loop import (
    ABORTED_BEFORE_DISPATCH,
    ABORTED_IN_FLIGHT,
    RunOptions,
    run_agent,
    system_message,
    user_message,
)
from app.providers.mock import MockProvider
from app.session import JsonlSessionStore
from app.tools import FunctionTool, ToolRegistry


def turns_with(calls: list[ToolCall]) -> list[AssistantTurn]:
    return [
        AssistantTurn(
            message=Message(role="assistant", content="", tool_calls=calls),
            stop_reason="tool_calls",
        ),
        AssistantTurn(message=Message(role="assistant", content="done"), stop_reason="stop"),
    ]


def tool_messages(result: object) -> list[str]:
    assert hasattr(result, "messages")
    return [m.content for m in result.messages if m.role == "tool"]


async def test_parallel_tools_actually_overlap() -> None:
    """两个 parallel 工具必须真的并发：互相等待对方进入，串行会超时。"""
    first_entered = asyncio.Event()
    second_entered = asyncio.Event()

    async def first(args: dict[str, object], context: object) -> ToolResult:
        first_entered.set()
        await asyncio.wait_for(second_entered.wait(), timeout=1.0)
        return ToolResult(content="first")

    async def second(args: dict[str, object], context: object) -> ToolResult:
        second_entered.set()
        await asyncio.wait_for(first_entered.wait(), timeout=1.0)
        return ToolResult(content="second")

    registry = (
        ToolRegistry()
        .register(FunctionTool(name="first", handler=first, execution_mode="parallel"))
        .register(FunctionTool(name="second", handler=second, execution_mode="parallel"))
    )
    provider = MockProvider(
        turns_with(
            [
                ToolCall(id="c1", name="first", arguments={}),
                ToolCall(id="c2", name="second", arguments={}),
            ]
        )
    )

    result = await run_agent(
        RunOptions(provider=provider, tools=registry, messages=[user_message("go")])
    )

    assert result.status == "completed"
    assert tool_messages(result) == ["first", "second"]


async def test_exclusive_tools_do_not_overlap() -> None:
    async def slow(args: dict[str, object], context: object) -> ToolResult:
        await asyncio.sleep(0.03)
        return ToolResult(content=str(args.get("name", "?")))

    registry = (
        ToolRegistry()
        .register(FunctionTool(name="a", handler=slow, execution_mode="exclusive"))
        .register(FunctionTool(name="b", handler=slow, execution_mode="exclusive"))
    )
    provider = MockProvider(
        turns_with(
            [
                ToolCall(id="c1", name="a", arguments={"name": "a"}),
                ToolCall(id="c2", name="b", arguments={"name": "b"}),
            ]
        )
    )

    start = time.monotonic()
    result = await run_agent(
        RunOptions(provider=provider, tools=registry, messages=[user_message("go")])
    )
    elapsed = time.monotonic() - start

    assert result.status == "completed"
    assert elapsed >= 0.05, "exclusive 工具必须串行执行"


async def test_default_execution_mode_is_exclusive() -> None:
    """未声明 execution_mode 的工具不能被并行（DSH 默认 exclusive）。"""
    async def slow(args: dict[str, object], context: object) -> ToolResult:
        await asyncio.sleep(0.03)
        return ToolResult(content=str(args.get("name", "?")))

    registry = (
        ToolRegistry()
        .register(FunctionTool(name="a", handler=slow))
        .register(FunctionTool(name="b", handler=slow))
    )
    provider = MockProvider(
        turns_with(
            [
                ToolCall(id="c1", name="a", arguments={"name": "a"}),
                ToolCall(id="c2", name="b", arguments={"name": "b"}),
            ]
        )
    )

    start = time.monotonic()
    await run_agent(
        RunOptions(provider=provider, tools=registry, messages=[user_message("go")])
    )
    elapsed = time.monotonic() - start

    assert elapsed >= 0.05


async def test_results_are_committed_in_model_order() -> None:
    """先完成的工具也要等前面的 slot 就绪，结果顺序 = 模型顺序。"""

    async def slow(args: dict[str, object], context: object) -> ToolResult:
        await asyncio.sleep(0.05)
        return ToolResult(content="slow")

    async def fast(args: dict[str, object], context: object) -> ToolResult:
        await asyncio.sleep(0.01)
        return ToolResult(content="fast")

    registry = (
        ToolRegistry()
        .register(FunctionTool(name="slow", handler=slow, execution_mode="parallel"))
        .register(FunctionTool(name="fast", handler=fast, execution_mode="parallel"))
    )
    provider = MockProvider(
        turns_with(
            [
                ToolCall(id="c1", name="slow", arguments={}),
                ToolCall(id="c2", name="fast", arguments={}),
            ]
        )
    )

    result = await run_agent(
        RunOptions(provider=provider, tools=registry, messages=[user_message("go")])
    )

    assert tool_messages(result) == ["slow", "fast"]


async def test_exclusive_call_is_a_barrier_between_parallel_groups() -> None:
    timeline: list[str] = []

    def make(name: str, delay: float, mode: str) -> FunctionTool:
        async def handler(args: dict[str, object], context: object) -> ToolResult:
            timeline.append(f"{name}:start")
            await asyncio.sleep(delay)
            timeline.append(f"{name}:end")
            return ToolResult(content=name)

        return FunctionTool(name=name, handler=handler, execution_mode=mode)

    registry = (
        ToolRegistry()
        .register(make("p1", 0.02, "parallel"))
        .register(make("p2", 0.02, "parallel"))
        .register(make("ex", 0.01, "exclusive"))
        .register(make("p3", 0.02, "parallel"))
        .register(make("p4", 0.02, "parallel"))
    )
    calls = [
        ToolCall(id="c1", name="p1", arguments={}),
        ToolCall(id="c2", name="p2", arguments={}),
        ToolCall(id="c3", name="ex", arguments={}),
        ToolCall(id="c4", name="p3", arguments={}),
        ToolCall(id="c5", name="p4", arguments={}),
    ]

    result = await run_agent(
        RunOptions(
            provider=MockProvider(turns_with(calls)), tools=registry, messages=[user_message("go")]
        )
    )

    assert result.status == "completed"
    assert timeline.index("ex:start") > max(timeline.index("p1:end"), timeline.index("p2:end"))
    assert min(timeline.index("p3:start"), timeline.index("p4:start")) > timeline.index("ex:end")


async def test_max_parallel_tool_calls_bounds_the_pool() -> None:
    active = 0
    peak = 0

    async def handler(args: dict[str, object], context: object) -> ToolResult:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return ToolResult(content="ok")

    registry = ToolRegistry().register(
        FunctionTool(name="work", handler=handler, execution_mode="parallel")
    )
    calls = [ToolCall(id=f"c{index}", name="work", arguments={}) for index in range(4)]

    await run_agent(
        RunOptions(
            provider=MockProvider(turns_with(calls)),
            tools=registry,
            messages=[user_message("go")],
            max_parallel_tool_calls=2,
        )
    )

    assert peak == 2


async def test_cancel_commits_started_results_and_skips_the_rest() -> None:
    """取消后：已开始的 call 结果必须提交，未派发的补 synthetic 结果。"""
    cancel_event = asyncio.Event()

    async def trigger(args: dict[str, object], context: object) -> ToolResult:
        cancel_event.set()
        return ToolResult(content="started")

    async def never(args: dict[str, object], context: object) -> ToolResult:
        return ToolResult(content="never")

    registry = (
        ToolRegistry()
        .register(FunctionTool(name="trigger", handler=trigger, execution_mode="parallel"))
        .register(FunctionTool(name="never", handler=never, execution_mode="parallel"))
        .register(FunctionTool(name="never2", handler=never, execution_mode="parallel"))
    )
    calls = [
        ToolCall(id="c1", name="trigger", arguments={}),
        ToolCall(id="c2", name="never", arguments={}),
        ToolCall(id="c3", name="never2", arguments={}),
    ]

    result = await run_agent(
        RunOptions(
            provider=MockProvider(turns_with(calls)),
            tools=registry,
            messages=[user_message("go")],
            cancel_event=cancel_event,
            max_parallel_tool_calls=1,
        )
    )

    assert result.status == "aborted"
    assert tool_messages(result) == ["started", ABORTED_BEFORE_DISPATCH, ABORTED_BEFORE_DISPATCH]


async def test_outer_cancel_keeps_completed_sibling_results(tmp_path: Path) -> None:
    """外层 `task.cancel()`：前面的调用被取消，后面已经跑完的结果不能跟着消失。

    反例（RC01）：`commit_ready` 只提交「模型顺序上连续」的槽位。0 号被取消后永远
    没有槽位，1 号已经完成的结果就被堵住丢掉 —— 会话里留下两个未配对调用，
    续聊时把这种历史直接送给模型。
    """
    session = JsonlSessionStore(tmp_path / "cancel.jsonl")
    seed = [system_message("sys"), user_message("run both")]
    for message in seed:
        await session.append(MessageEntry(0, 0, message, "t"))

    slow_started = asyncio.Event()
    fast_finished = asyncio.Event()
    effects: list[str] = []

    async def slow(args: dict[str, object], context: object) -> ToolResult:
        slow_started.set()
        await asyncio.Event().wait()
        return ToolResult(content="slow-done")

    async def fast(args: dict[str, object], context: object) -> ToolResult:
        effects.append("fast ran")
        fast_finished.set()
        return ToolResult(content="fast-result")

    registry = (
        ToolRegistry()
        .register(FunctionTool(name="slow", handler=slow, execution_mode="parallel"))
        .register(FunctionTool(name="fast", handler=fast, execution_mode="parallel"))
    )
    provider = MockProvider(
        [
            AssistantTurn(
                message=Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall(id="slow-id", name="slow", arguments={}),
                        ToolCall(id="fast-id", name="fast", arguments={}),
                    ],
                ),
                stop_reason="tool_calls",
            )
        ]
    )
    parent = asyncio.create_task(
        run_agent(RunOptions(provider=provider, tools=registry, messages=seed, session=session))
    )
    await asyncio.wait_for(slow_started.wait(), 2)
    await asyncio.wait_for(fast_finished.wait(), 2)
    for _ in range(3):
        await asyncio.sleep(0)
    parent.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await parent

    persisted = await session.load_messages()
    results = {m.tool_call_id: m.content for m in persisted if m.role == "tool"}
    assert effects == ["fast ran"], "已完成的结果不能靠重跑工具来补"
    assert results["fast-id"] == "fast-result", "后一个工具已完成的真实结果必须保留"
    assert results["slow-id"] == ABORTED_IN_FLIGHT, "没结果的调用补一条中断说明"
    # 顺序仍按模型顺序：slow 在前、fast 在后。
    assert [m.tool_call_id for m in persisted if m.role == "tool"] == ["slow-id", "fast-id"]

    calls = {call.id for m in persisted for call in m.tool_calls}
    assert calls == set(results), "中断历史不能留下未配对的调用"
    timeline = await session.load_timeline()
    assert timeline.last_turn_open is False, "运行边界必须收尾，日志里不能留下永远开着的 turn"


async def test_cancel_during_result_commit_leaves_one_result_and_closes_the_step(
    tmp_path: Path,
) -> None:
    """反例（F01 / F02）：取消恰好落在「结果已写入、提交还没返回」的窗口里。

    F01：`commit_ready` 先把 slot 取走再 await 落盘，`committed` 却要等 await 之后才
    递增。取消卡在这个缝里，收尾只看到「slot 没了、committed 没动」，于是给同一个
    `call-1` 又补一条 synthetic —— 一次调用留下两条互相矛盾的结果，续接也两条都带着。

    F02：同一路径上 `step/end` 从来没写（取消绕过了 `_end_step`），只补了 `turn/end`，
    `last_turn_open=False` 并不代表阶段事件闭合。

    用合法的异步 SessionSink 制造交错：真实结果写完、append 尚未返回时取消父运行。
    """
    entered = asyncio.Event()
    release = asyncio.Event()
    effects: list[str] = []

    class PausingStore(JsonlSessionStore):
        paused = False

        async def append(self, event: SessionEvent) -> str:
            node = await super().append(event)
            if (
                not self.paused
                and isinstance(event, MessageEntry)
                and event.message.role == "tool"
                and event.message.content == "real tool result"
            ):
                self.paused = True
                entered.set()
                await release.wait()
            return node

    session = PausingStore(tmp_path / "commit-window.jsonl")
    seed = [system_message("sys"), user_message("perform the tool once")]
    for message in seed:
        await session.append(MessageEntry(0, 0, message, "t"))

    def work(args: dict[str, object], context: object) -> ToolResult:
        effects.append("tool ran once")
        return ToolResult(content="real tool result")

    registry = ToolRegistry().register(FunctionTool(name="work", handler=work))
    provider = MockProvider(
        [
            AssistantTurn(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="call-1", name="work", arguments={})],
                ),
                stop_reason="tool_calls",
            )
        ]
    )
    parent = asyncio.create_task(
        run_agent(RunOptions(provider=provider, tools=registry, messages=seed, session=session))
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        parent.cancel()
        # 取消会等待这次提交收尾，sink 必须独立释放，不能等 parent 结束后才释放。
        release.set()
        with contextlib.suppress(asyncio.CancelledError):
            await parent
    finally:
        release.set()

    persisted = await session.load_messages()
    results = [m for m in persisted if m.role == "tool"]
    assert effects == ["tool ran once"], "取消不能靠重跑工具来补日志"
    assert [m.tool_call_id for m in results] == ["call-1"], "同一次调用只能有一条结果"
    assert results[0].content == "real tool result", "已经产生的真实结果不能被 synthetic 覆盖"

    # 续接侧：配对后仍然只有一条结果（不能靠去重掩盖执行器重复提交）。
    continued = [m for m in _continuation(persisted, "continue") if m.role == "tool"]
    assert [m.tool_call_id for m in continued] == ["call-1"]
    assert continued[0].content == "real tool result"

    events = await session.load_events()
    starts = [e for e in events if isinstance(e, StepStartEntry)]
    ends = [e for e in events if isinstance(e, StepEndEntry)]
    assert len(starts) == 1 and len(ends) == 1, "每个已开始的 step 必须闭合一次"
    assert [e.step for e in ends] == [e.step for e in starts]
    assert [e.status for e in events if isinstance(e, TurnEndEntry)] == ["aborted"]
    timeline = await session.load_timeline()
    assert timeline.last_turn_open is False


async def test_cancel_waits_for_a_queued_result_write(tmp_path: Path) -> None:
    """默认 JSONL 写入仍在排队时取消，也必须保留已完成工具的唯一真实结果。"""
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
    busy = asyncio.Event()
    writing = asyncio.Event()
    release = threading.Event()
    effects: list[str] = []
    events: list[object] = []

    class ObservedStore(JsonlSessionStore):
        async def append(self, event: SessionEvent) -> str:
            if isinstance(event, MessageEntry) and event.message.role == "tool":
                writing.set()
            return await super().append(event)

    session = ObservedStore(tmp_path / "queued-write.jsonl")
    seed = [user_message("run once")]
    await session.append(MessageEntry(0, 0, seed[0], "t"))

    def other_io() -> None:
        loop.call_soon_threadsafe(busy.set)
        release.wait()

    def work(args: dict[str, object], context: object) -> ToolResult:
        effects.append("ran once")
        loop.run_in_executor(None, other_io)
        return ToolResult(content="real result")

    registry = ToolRegistry().register(FunctionTool(name="work", handler=work))
    parent = asyncio.create_task(
        run_agent(
            RunOptions(
                provider=MockProvider(turns_with([ToolCall(id="c1", name="work")])),
                tools=registry,
                messages=seed,
                session=session,
                on_event=events.append,
            )
        )
    )
    try:
        await asyncio.wait_for(busy.wait(), 2)
        await asyncio.wait_for(writing.wait(), 2)
        parent.cancel()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        release.set()
        with contextlib.suppress(asyncio.CancelledError):
            await parent
    finally:
        release.set()
        if not parent.done():
            parent.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await parent

    results = [m for m in await session.load_messages() if m.role == "tool"]
    assert parent.cancelled()
    assert effects == ["ran once"]
    assert [(m.tool_call_id, m.content) for m in results] == [("c1", "real result")]
    assert sum(isinstance(event, ToolFinished) for event in events) == 1


async def test_cancel_is_bounded_when_a_result_write_is_stuck(
    tmp_path: Path, monkeypatch
) -> None:
    """反例（F01 收尾）：sink 永不返回时，取消也必须在有界时间内结束。

    「收完同一次写入再传播取消」不能变成无上限的等待 —— 网络盘、磁盘满、被锁的文件
    都会让 `cancel()` 永不返回，外层 `wait_for` 的超时随之失效，进程也退不出去。
    上限之内收完就收完；超过上限，结果已经在上下文里，就不再用 synthetic 重复提交它。
    """
    monkeypatch.setattr(loop_module, "COMMIT_GRACE_SECONDS", 0.05)
    entered = asyncio.Event()
    release = asyncio.Event()
    effects: list[str] = []

    class StuckStore(JsonlSessionStore):
        paused = False

        async def append(self, event: SessionEvent) -> str:
            node = await super().append(event)
            if (
                not self.paused
                and isinstance(event, MessageEntry)
                and event.message.role == "tool"
            ):
                self.paused = True
                entered.set()
                await release.wait()  # 永不返回，直到测试收尾
            return node

    session = StuckStore(tmp_path / "stuck-write.jsonl")
    seed = [system_message("sys"), user_message("run once")]
    for message in seed:
        await session.append(MessageEntry(0, 0, message, "t"))

    def work(args: dict[str, object], context: object) -> ToolResult:
        effects.append("ran once")
        return ToolResult(content="real tool result")

    registry = ToolRegistry().register(FunctionTool(name="work", handler=work))
    provider = MockProvider(
        [
            AssistantTurn(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="call-1", name="work", arguments={})],
                ),
                stop_reason="tool_calls",
            )
        ]
    )
    parent = asyncio.create_task(
        run_agent(RunOptions(provider=provider, tools=registry, messages=seed, session=session))
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        parent.cancel()
        # 超时保护不是断言，是失败时让测试结束而不是挂住：等待上限远小于 2s。
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(parent, 2)
    finally:
        release.set()
        if not parent.done():
            parent.cancel()
            with contextlib.suppress(BaseException):
                await parent

    assert parent.done(), "取消不能被卡住的写入无限期拖住"
    results = [m for m in await session.load_messages() if m.role == "tool"]
    assert effects == ["ran once"]
    assert [(m.tool_call_id, m.content) for m in results] == [("call-1", "real tool result")]


async def test_concludes_turn_keeps_sibling_results() -> None:
    """DSH：concludes_turn 不丢同 step 内其余 call 的结果，只保证不再请求模型。"""

    async def submit(args: dict[str, object], context: object) -> ToolResult:
        return ToolResult(content="submitted", concludes_turn=True)

    registry = (
        ToolRegistry()
        .register(FunctionTool(name="submit", handler=submit, execution_mode="parallel"))
        .register(
            FunctionTool(
                name="echo",
                handler=lambda args, context: ToolResult(content="echoed"),
                execution_mode="parallel",
            )
        )
    )
    provider = MockProvider(
        [
            AssistantTurn(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(id="c1", name="submit", arguments={}),
                        ToolCall(id="c2", name="echo", arguments={}),
                    ],
                ),
                stop_reason="tool_calls",
            )
        ]
    )

    result = await run_agent(
        RunOptions(provider=provider, tools=registry, messages=[user_message("go")])
    )

    assert result.status == "completed"
    assert len(provider.requests) == 1, "concludes_turn 之后不再请求模型"
    assert tool_messages(result) == ["submitted", "echoed"]


def test_registry_reports_execution_modes() -> None:
    registry = (
        ToolRegistry()
        .register(FunctionTool(name="safe", handler=lambda a, c: ToolResult(content="")))
        .register(
            FunctionTool(
                name="solo",
                handler=lambda a, c: ToolResult(content=""),
                execution_mode="exclusive",
            )
        )
    )

    assert registry.execution_mode("safe") == "exclusive", "默认 exclusive"
    assert registry.execution_mode("solo") == "exclusive"
    assert registry.execution_mode("missing") == "exclusive", "未知工具也不并行"
