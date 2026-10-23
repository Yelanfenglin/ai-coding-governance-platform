"""Phase A：provider 流式输出。

中断语义对齐 DSH `packages/core/agent-loop/src/agent.ts`：
取消中断 → 提交可见前缀（`interrupted: true`）；错误中断 → 不把半成品写成消息。
"""

from __future__ import annotations

import asyncio

from app.contracts import (
    AssistantReasoningDelta,
    AssistantTextDelta,
    AssistantTurn,
    AssistantTurnEvent,
    FinishReason,
    Message,
    ModelRequest,
    ReasoningDelta,
    StreamChunk,
    StreamFinished,
    TextDelta,
    ToolCall,
    ToolResult,
)
from app.errors import ErrorCode, ProviderError
from app.loop import RunOptions, run_agent, user_message
from app.providers.mock import MockProvider
from app.tools import FunctionTool, ToolRegistry


def echo_tool() -> FunctionTool:
    return FunctionTool(
        name="echo",
        description="Echo text back",
        handler=lambda args, context: ToolResult(content=str(args.get("text", ""))),
    )


def final_turn(text: str) -> AssistantTurn:
    return AssistantTurn(message=Message(role="assistant", content=text), stop_reason="stop")


class ScriptedStreamProvider:
    """只实现 generate_stream 的 provider（DSH 里这是唯一入口）。"""

    def __init__(self, chunks: list[StreamChunk]) -> None:
        self.chunks = chunks
        self.requests: list[ModelRequest] = []

    def generate_stream(self, request: ModelRequest):  # noqa: ANN201
        self.requests.append(request)
        return self._stream()

    async def _stream(self):  # noqa: ANN201
        for chunk in self.chunks:
            yield chunk


async def test_text_deltas_are_emitted_before_the_turn_is_committed() -> None:
    provider = MockProvider([final_turn("hello world")])
    events: list[object] = []

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry(),
            messages=[user_message("go")],
            on_event=events.append,
        )
    )

    assert result.status == "completed"
    assert result.messages[-1].content == "hello world"

    deltas = [event for event in events if isinstance(event, AssistantTextDelta)]
    assert [delta.text for delta in deltas] == ["hello wo", "rld"]
    assert deltas[0].turn == 1 and deltas[0].step == 1

    committed = [event for event in events if isinstance(event, AssistantTurnEvent)]
    assert len(committed) == 1
    delta_index = events.index(deltas[0])
    assert events.index(committed[0]) > delta_index, "增量必须先于落盘事件发出"


async def test_streamed_tool_calls_are_executed_and_the_turn_continues() -> None:
    provider = MockProvider(
        [
            AssistantTurn(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "hi"})],
                ),
                stop_reason="tool_calls",
            ),
            final_turn("done"),
        ]
    )

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry().register(echo_tool()),
            messages=[user_message("go")],
        )
    )

    assert result.status == "completed"
    assert result.steps == 2
    assert [m.content for m in result.messages if m.role == "tool"] == ["hi"]
    assert result.messages[-1].content == "done"


async def test_reasoning_deltas_are_forwarded() -> None:
    provider = ScriptedStreamProvider(
        [
            ReasoningDelta(index=0, text="thinking"),
            TextDelta(index=1, text="answer"),
            StreamFinished(reason=FinishReason(kind="stop")),
        ]
    )
    events: list[object] = []

    await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry(),
            messages=[user_message("go")],
            on_event=events.append,
        )
    )

    reasoning = [event for event in events if isinstance(event, AssistantReasoningDelta)]
    assert [event.text for event in reasoning] == ["thinking"]


async def test_cancel_mid_stream_commits_the_visible_prefix() -> None:
    """DSH agent.ts：取消中断时把已输出文本作为 interrupted message 落盘。"""
    cancel_event = asyncio.Event()
    provider = ScriptedStreamProvider([])

    async def stream():  # noqa: ANN202
        yield TextDelta(index=0, text="partial answer")
        cancel_event.set()
        yield TextDelta(index=0, text="never delivered")

    provider.generate_stream = lambda request: stream()  # type: ignore[method-assign]

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry(),
            messages=[user_message("go")],
            cancel_event=cancel_event,
        )
    )

    assert result.status == "aborted"
    assistant = [m for m in result.messages if m.role == "assistant"]
    assert len(assistant) == 1
    assert assistant[0].content == "partial answer"
    assert assistant[0].interrupted is True


async def test_error_mid_stream_does_not_commit_a_partial_message() -> None:
    """DSH agent.ts：非取消的失败只留 attempt，不把半成品写成 assistant message。"""
    provider = ScriptedStreamProvider([])

    async def stream():  # noqa: ANN202
        yield TextDelta(index=0, text="partial")
        raise ProviderError("stream died", ErrorCode.SERVER)

    provider.generate_stream = lambda request: stream()  # type: ignore[method-assign]

    result = await run_agent(
        RunOptions(
            provider=provider, tools=ToolRegistry(), messages=[user_message("go")]
        )
    )

    assert result.status == "error"
    assert result.error == "stream died"
    assert [m.role for m in result.messages] == ["user"]


async def test_stream_without_a_terminal_event_is_stream_closed() -> None:
    provider = ScriptedStreamProvider([TextDelta(index=0, text="dangling")])

    result = await run_agent(
        RunOptions(
            provider=provider, tools=ToolRegistry(), messages=[user_message("go")]
        )
    )

    assert result.status == "error"
    assert result.error == "provider stream ended without a terminal event"


async def test_provider_without_stream_is_rejected() -> None:
    """契约只认 `generate_stream`：只有 `generate` 的 provider 直接失败。

    DSH `LlmAdapter.stream()` 是 the only required method（"Stream one model call
    as raw chunks. The only required method."）。本地不再保留非流式回退，所以这类
    provider 不会被静默接受。
    """

    class NonStreamingProvider:
        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def generate(self, request: ModelRequest) -> AssistantTurn:
            self.requests.append(request)
            return final_turn("done")

    provider = NonStreamingProvider()

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry(),
            messages=[user_message("go")],
        )
    )

    # 没有 `generate_stream` 就没有任何可用的入口：run 以 error 结束，
    # 也不会退回去调用 `generate`。
    assert result.status == "error"
    assert provider.requests == []


async def test_provider_stream_is_closed_after_the_turn_finishes() -> None:
    """反例（真 HTTP 接线时发现）：流必须显式关，不能等 GC。

    之前 `StreamFinished` 分支直接 return，SSE 生成器没人关 —— 单元测试里的假 transport
    不在乎，真 httpx 会一直占着连接，直到 GC 才抛
    `aclose(): asynchronous generator is already running`（unraisable）。
    """
    import json

    from app.providers.openai_compatible import OpenAiCompatibleProvider, TransportRequest

    closed: list[str] = []

    def transport(http_request: TransportRequest):  # type: ignore[no-untyped-def]
        async def source():  # type: ignore[no-untyped-def]
            try:
                yield (
                    "data: "
                    + json.dumps({"choices": [{"index": 0, "delta": {"content": "hi"}}]})
                    + "\n\n"
                )
                yield (
                    "data: "
                    + json.dumps(
                        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                    )
                    + "\n\n"
                )
                yield "data: [DONE]\n\n"
            finally:
                closed.append("closed")

        return source()

    provider = OpenAiCompatibleProvider(
        base_url="http://stub/v1", model="stub", stream_transport=transport
    )

    result = await run_agent(
        RunOptions(provider=provider, tools=ToolRegistry(), messages=[user_message("go")])
    )

    assert result.status == "completed"
    assert closed == ["closed"], "跑完一轮就必须把 provider 的流关掉"
