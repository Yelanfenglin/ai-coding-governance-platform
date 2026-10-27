"""按脚本产出完整模型流并记录请求，供离线测试使用。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from ..contracts import (
    AssistantTurn,
    BlockEnd,
    BlockStart,
    FinishReason,
    LlmFailure,
    ModelProvider,
    ModelRequest,
    ReasoningBlock,
    ReasoningDelta,
    StreamChunk,
    StreamFinished,
    TextBlock,
    TextDelta,
    ToolCallBlock,
    ToolCallDelta,
    UsageChunk,
)

#: 文本切块大小，保证 delta 顺序可断言。
TEXT_CHUNK_SIZE = 8


class MockProviderExhausted(RuntimeError):
    """脚本用完还想再要一个 turn，说明 loop 没有按预期终止。"""


@dataclass(slots=True)
class MockProvider:
    turns: Sequence[AssistantTurn]
    requests: list[ModelRequest] = field(default_factory=list)

    def generate_stream(self, request: ModelRequest) -> AsyncIterator[StreamChunk]:
        return _scripted_stream(self._take(request))

    def _take(self, request: ModelRequest) -> AssistantTurn:
        self.requests.append(request)
        index = len(self.requests) - 1
        if index >= len(self.turns):
            raise MockProviderExhausted(f"mock provider exhausted after {len(self.turns)} turn(s)")
        return self.turns[index]


def chunks_for_turn(turn: AssistantTurn) -> list[StreamChunk]:
    """把一个已完成的 turn 展开成完整 chunk 序列（含 block-end / usage / finish）。"""
    chunks: list[StreamChunk] = []
    index = 0

    if turn.message.reasoning_content:
        chunks.append(BlockStart(index=index, block_type="reasoning"))
        for piece in _chunk_text(turn.message.reasoning_content):
            chunks.append(ReasoningDelta(index=index, text=piece))
        chunks.append(BlockEnd(index=index, block=ReasoningBlock(turn.message.reasoning_content)))
        index += 1

    if turn.message.content:
        chunks.append(BlockStart(index=index, block_type="text"))
        for piece in _chunk_text(turn.message.content):
            chunks.append(TextDelta(index=index, text=piece))
        chunks.append(BlockEnd(index=index, block=TextBlock(text=turn.message.content)))
        index += 1

    for call in turn.message.tool_calls:
        raw = (
            call.arguments
            if isinstance(call.arguments, str)
            else json.dumps(call.arguments, ensure_ascii=False)
        )
        chunks.append(BlockStart(index=index, block_type="tool-call"))
        chunks.append(
            ToolCallDelta(index=index, call_id=call.id, name=call.name, arguments_delta=raw)
        )
        chunks.append(
            BlockEnd(
                index=index,
                block=ToolCallBlock(id=call.id, name=call.name, arguments=raw),
            )
        )
        index += 1

    if turn.usage is not None:
        chunks.append(UsageChunk(usage=turn.usage))
    chunks.append(StreamFinished(reason=finish_reason_of(turn.stop_reason)))
    return chunks


def finish_reason_of(stop_reason: str) -> FinishReason:
    """内部 `StopReason` → 协议 `FinishReason`（`app.assembler` 里映射的反向）。"""
    if stop_reason == "tool_calls":
        return FinishReason(kind="tool-calls")
    if stop_reason == "length":
        return FinishReason(kind="max-tokens")
    if stop_reason == "error":
        return FinishReason(
            kind="error",
            failure=LlmFailure(message="model returned an error stop reason", code="PROVIDER"),
        )
    if stop_reason == "aborted":
        return FinishReason(
            kind="aborted",
            failure=LlmFailure(message="model request cancelled", code="ABORTED"),
        )
    return FinishReason(kind="stop")


async def _scripted_stream(turn: AssistantTurn) -> AsyncIterator[StreamChunk]:
    for chunk in chunks_for_turn(turn):
        yield chunk


def _chunk_text(text: str) -> list[str]:
    if text == "":
        return []
    return [
        text[offset : offset + TEXT_CHUNK_SIZE] for offset in range(0, len(text), TEXT_CHUNK_SIZE)
    ]


def script_provider(turns: Sequence[AssistantTurn]) -> ModelProvider:
    return MockProvider(turns)


def stream_from_coroutine(
    factory: Callable[[], Awaitable[AssistantTurn]],
) -> AsyncIterator[StreamChunk]:
    """将返回 AssistantTurn 的协程包装为测试用 StreamChunk 流。"""

    async def _run() -> AsyncIterator[StreamChunk]:
        for chunk in chunks_for_turn(await factory()):
            yield chunk

    return _run()
