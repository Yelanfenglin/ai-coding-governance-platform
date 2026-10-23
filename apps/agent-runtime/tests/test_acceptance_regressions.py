"""验收反例回归（2026-10-20 独立验收提出，已修复并迁入正式测试）。

来源：`output/acceptance_review_20261020/report.md` 与 `review_probes.py`。
覆盖：SSE 行协议 / usage trailer / 同块多字段、compaction 分组与滚动摘要与会话重放、
记忆的适用条件 / 跨 run 快照 / 预算 / 并发写入、挂起 provider 的取消、中文多词兜底。

不调用真实模型或外部网络。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.assembler import BlockAssembler
from app.compaction import (
    CompactionPolicy,
    compact_context,
    is_compaction_summary,
)
from app.contracts import (
    AssistantTurn,
    Message,
    MessageEntry,
    ModelRequest,
    StreamFinished,
    ToolCall,
    ToolResult,
)
from app.loop import RunOptions, run_agent, system_message, user_message
from app.memory import JsonMemoryStore, MemoryFullError
from app.memory_tools import memory_tools
from app.providers.mock import MockProvider, chunks_for_turn
from app.providers.openai_compatible import OpenAiCompatibleProvider
from app.session import JsonlSessionStore
from app.session_search import SessionSearch
from app.tools import FunctionTool, ToolRegistry


def final(text: str = "done") -> AssistantTurn:
    return AssistantTurn(Message(role="assistant", content=text), "stop")


def sse(payload: dict) -> str:
    return "data: " + json.dumps(payload) + "\n\n"


def tool_turn(text: str = "x" * 400) -> AssistantTurn:
    return AssistantTurn(
        Message(
            role="assistant",
            tool_calls=[ToolCall("call-1", "echo", {"text": text})],
        ),
        "tool_calls",
    )


async def summarize(messages: list[Message]) -> str:
    return "SUMMARY: preserve the requested PWM configuration"


async def test_default_httpx_line_shape_is_accepted() -> None:
    """httpx aiter_lines() removes line terminators; the default adapter yields those lines."""
    async def lines(request):
        yield sse({"choices": [{"delta": {"content": "OK"}, "finish_reason": None}]}).strip()
        yield ""
        yield sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}).strip()
        yield ""
        yield "data: [DONE]"
        yield ""

    provider = OpenAiCompatibleProvider("http://unused.invalid/v1", "test", stream_transport=lines)
    result = await run_agent(
        RunOptions(provider=provider, tools=ToolRegistry(), messages=[user_message("go")])
    )
    assert result.status == "completed", result.error
    assert result.messages[-1].content == "OK"


async def test_default_transport_consumes_a_successful_response() -> None:
    """默认 transport 必须真的读响应体（读取循环不能被缩进到错误分支后面）。"""
    from types import SimpleNamespace
    from unittest.mock import patch

    class Response:
        status_code = 200
        headers: dict[str, str] = {}
        iterated = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def aiter_lines(self):
            self.iterated = True
            for line in [
                'data: {"choices":[{"delta":{"content":"OK"},"finish_reason":"stop"}]}',
                "",
                "data: [DONE]",
                "",
            ]:
                yield line

    response = Response()

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, *args, **kwargs):
            return response

    with patch.dict("sys.modules", {"httpx": SimpleNamespace(AsyncClient=Client)}):
        provider = OpenAiCompatibleProvider("http://unused.invalid/v1", "test")
        result = await run_agent(
            RunOptions(provider=provider, tools=ToolRegistry(), messages=[user_message("go")])
        )
    assert result.status == "completed", {
        "error": result.error,
        "response_lines_iterated": response.iterated,
    }
    assert result.messages[-1].content == "OK"


async def test_chunk_starting_data_inside_content_is_not_a_line_boundary() -> None:
    """正文里的 `data:`（`metadata: PWM`）不能被当成新行起点。"""
    wire = (
        "data: "
        + json.dumps(
            {"choices": [{"delta": {"content": "metadata: PWM"}, "finish_reason": "stop"}]}
        )
        + "\n\ndata: [DONE]\n\n"
    )
    cut = wire.index("data: PWM")

    async def chunks(request):
        yield wire[:cut]
        yield wire[cut:]

    provider = OpenAiCompatibleProvider(
        "http://unused.invalid/v1", "test", stream_transport=chunks
    )
    result = await run_agent(
        RunOptions(provider=provider, tools=ToolRegistry(), messages=[user_message("go")])
    )
    assert result.status == "completed", result.error
    assert result.messages[-1].content == "metadata: PWM"


async def test_usage_trailer_is_included_in_the_result() -> None:
    async def chunks(request):
        yield sse({"choices": [{"delta": {"content": "OK"}, "finish_reason": None}]})
        yield sse({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        yield sse({"choices": [], "usage": {
            "prompt_tokens": 6, "completion_tokens": 3, "total_tokens": 9,
        }})
        yield "data: [DONE]\n\n"

    provider = OpenAiCompatibleProvider("http://unused.invalid/v1", "test", stream_transport=chunks)
    result = await run_agent(
        RunOptions(provider=provider, tools=ToolRegistry(), messages=[user_message("go")])
    )
    assert result.status == "completed"
    assert result.usage.total_tokens == 9, result.usage


async def test_compaction_preserves_tool_call_result_pairs() -> None:
    messages = [
        system_message("sys"),
        user_message("goal " * 100),
        tool_turn().message,
        Message(role="tool", content="evidence " * 100, tool_call_id="call-1", name="echo"),
    ]
    outcome = await compact_context(
        messages, CompactionPolicy(100, retain_ratio=0.01, summarizer=summarize), 1000
    )
    assert outcome is not None
    available = {call.id for message in outcome.messages for call in message.tool_calls}
    orphaned = [
        message.tool_call_id for message in outcome.messages
        if message.role == "tool" and message.tool_call_id not in available
    ]
    assert orphaned == [], f"tool results retained without their assistant call: {orphaned}"


async def test_compacted_session_can_reconstruct_the_live_context(tmp_path: Path) -> None:
    store = JsonlSessionStore(tmp_path / "session.jsonl")
    initial = [system_message("sys"), user_message("goal " * 100)]
    # Seed input explicitly to keep this probe focused on compaction replay.
    for message in initial:
        await store.append(MessageEntry(0, 0, message, "t"))
    provider = MockProvider([tool_turn(), final()])
    registry = ToolRegistry().register(FunctionTool(
        name="echo", handler=lambda args, ctx: ToolResult(str(args["text"]))
    ))
    result = await run_agent(RunOptions(
        provider=provider, tools=registry, messages=initial, session=store,
        compaction=CompactionPolicy(100, retain_ratio=0.01, summarizer=summarize),
    ))
    assert result.status == "completed"
    assert any("SUMMARY:" in message.content for message in result.messages)
    replayed = await JsonlSessionStore(store.path).load_messages()
    assert replayed == result.messages, "session replay ignores the CompactionEntry"


async def test_replace_respects_the_memory_budget(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memory.json", max_chars=100)
    await store.remember("clock", "20MHz")
    with pytest.raises(MemoryFullError):
        await store.replace("clock", "x" * 1000)


async def test_agent_memory_writes_preserve_applicability(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memory.json")
    registry = ToolRegistry()
    for tool in memory_tools(store):
        registry.register(tool)
    result = await registry.execute(ToolCall("m1", "remember_fact", {
        "key": "pwm.period", "value": "PR2=0xFF", "source": "chunk-42",
        "applies_to": {"chip": "PIC16F877A", "version": "v1"},
    }))
    assert not result.is_error
    wrong_chip = await store.recall(applies_to={"chip": "PIC18F4550", "version": "v2"})
    assert wrong_chip == [], f"Agent wrote an unscoped chip fact: {wrong_chip}"


async def test_memory_snapshot_does_not_survive_invalidation_across_runs(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memory.json")
    await store.remember("clock", "20MHz")
    first = await run_agent(RunOptions(
        provider=MockProvider([final("A")]), tools=ToolRegistry(),
        messages=[system_message("sys"), user_message("go")], memory=store,
    ))
    await store.invalidate("clock")
    provider = MockProvider([final("B")])
    await run_agent(RunOptions(
        provider=provider, tools=ToolRegistry(),
        messages=[*first.messages, user_message("continue")], memory=store,
    ))
    stale = any("clock: 20MHz" in message.content for message in provider.requests[0].messages)
    assert not stale, (
        "the prior run's ephemeral snapshot is retained as an ordinary system message"
    )


async def test_simultaneous_external_memory_writes_preserve_both_updates(tmp_path: Path) -> None:
    class SynchronizedReads(JsonMemoryStore):
        """Arrange the valid interleaving where both clients finish reading before writing."""
        def __init__(self, path):
            super().__init__(path)
            self.read_count = 0
            self.ready = asyncio.Event()

        async def _load(self):
            items = await super()._load()
            self.read_count += 1
            if self.read_count == 2:
                self.ready.set()
            await self.ready.wait()
            return items

    store = SynchronizedReads(tmp_path / "memory.json")
    await asyncio.gather(store.remember("clock", "20MHz"), store.remember("language", "zh"))
    saved = await JsonMemoryStore(store.path).all()
    assert {item.key for item in saved} == {"clock", "language"}, saved


async def test_cancel_event_stops_a_stalled_provider() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    cancel = asyncio.Event()

    class StalledProvider:
        def generate_stream(self, request):
            return self.stream()

        async def stream(self):
            started.set()
            await release.wait()
            for chunk in chunks_for_turn(final()):
                yield chunk

    task = asyncio.create_task(run_agent(RunOptions(
        provider=StalledProvider(), tools=ToolRegistry(),
        messages=[user_message("go")], cancel_event=cancel,
    )))
    await started.wait()
    cancel.set()
    done, _ = await asyncio.wait([task], timeout=0.1)
    returned_while_blocked = bool(done)
    release.set()
    result = await task
    assert result.status == "aborted"
    assert returned_while_blocked, "cancellation is only observed after the next provider chunk"


async def test_history_substring_fallback_preserves_multiple_query_terms(tmp_path: Path) -> None:
    session = JsonlSessionStore(tmp_path / "s1.jsonl")
    await session.append(MessageEntry(
        1, 1, Message(role="tool", content="通道选择位由 ADCON0 配置"), "t"
    ))
    search = SessionSearch(tmp_path / "search.db")
    await search.index_file(session.path)
    hits = await search.search("通道 ADCON0")
    assert hits, "FTS uses AND terms but LIKE fallback requires a contiguous phrase"


async def test_compaction_merges_the_prior_summary_instead_of_pinning_it() -> None:
    seen: list[list[Message]] = []

    async def summarize(messages: list[Message]) -> str:
        seen.append(list(messages))
        return "SUMMARY: preserve the requested PWM configuration"

    policy = CompactionPolicy(100, retain_ratio=0.01, summarizer=summarize)
    first = await compact_context([
        system_message("sys"), user_message("old " * 100), user_message("tail")
    ], policy, 1000)
    assert first is not None
    assert is_compaction_summary(first.messages[1]), "checkpoint 是 user 消息，不是 system"
    second = await compact_context([
        *first.messages, user_message("next " * 100), user_message("tail-2")
    ], policy, 1000)
    assert second is not None
    # DSH region.ts: 旧 checkpoint 是区域里的普通 user 消息，会随区域一起进入新摘要（滚动合并）。
    assert any(is_compaction_summary(m) for m in seen[-1]), "旧 checkpoint 没有进入新摘要"
    checkpoints = [m for m in second.messages if is_compaction_summary(m)]
    assert len(checkpoints) == 1, "旧摘要被保留成固定 system prompt 就会不断累积"
    assert checkpoints[0].role == "user"
    assert not any(
        message.role == "system" and "checkpoint" in message.content.lower()
        for message in second.messages
    )


async def test_combined_final_delta_records_both_content_and_finish_reason() -> None:
    async def chunks(request):
        yield sse({"choices": [{"delta": {"content": "OK"}, "finish_reason": "stop"}]})
        yield "data: [DONE]\n\n"

    provider = OpenAiCompatibleProvider("http://unused.invalid/v1", "test", stream_transport=chunks)
    result = await run_agent(
        RunOptions(provider=provider, tools=ToolRegistry(), messages=[user_message("go")])
    )
    assert result.status == "completed", result.error


async def test_same_delta_preserves_text_and_tool_calls() -> None:
    async def chunks(request):
        yield sse({"choices": [{"delta": {
            "content": "checking",
            "tool_calls": [{"index": 0, "id": "call-1", "function": {
                "name": "echo", "arguments": "{}",
            }}],
        }, "finish_reason": None}]})
        yield sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        yield "data: [DONE]\n\n"

    provider = OpenAiCompatibleProvider("http://unused.invalid/v1", "test", stream_transport=chunks)
    events = [event async for event in provider.generate_stream(ModelRequest([], []))]
    assert isinstance(events[-1], StreamFinished)
    turn = assemble(events).turn()
    assert turn.message.content == "checking"
    assert [call.id for call in turn.message.tool_calls] == ["call-1"], (
        "the text early return discards other fields from the same delta"
    )


def assemble(chunks):  # noqa: ANN001, ANN201
    assembler = BlockAssembler()
    for chunk in chunks:
        assembler.push(chunk)
    return assembler
