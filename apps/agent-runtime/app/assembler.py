"""组装流式消息块。

按 index 保存增量，以首次出现顺序输出；支持从 delta 建块。
block-end 关闭后忽略后续增量；缺少 finish 时使用 stop。
长度截断时丢弃工具调用，中断时仅保留非空的文本与推理块。
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import (
    AssistantTurn,
    BlockEnd,
    BlockStart,
    BlockType,
    ContentBlock,
    FinishReason,
    Message,
    ReasoningBlock,
    ReasoningDelta,
    StopReason,
    StreamChunk,
    StreamFinished,
    TextBlock,
    TextDelta,
    ToolCallBlock,
    ToolCallDelta,
    Usage,
    UsageChunk,
)


@dataclass(slots=True)
class _Partial:
    block_type: str
    text: str = ""
    tool_call_id: str | None = None
    tool_call_name: str | None = None
    tool_call_arguments: str = ""
    #: `block-end` 设置，权威且冻结该 partial。
    block: ContentBlock | None = None


#: 将协议终止原因映射为内部 StopReason；仅在组装完成时转换。
_STOP_REASON: dict[str, StopReason] = {
    "stop": "stop",
    "tool-calls": "tool_calls",
    "max-tokens": "length",
    "error": "error",
    "aborted": "aborted",
}


class BlockAssembler:
    """把 `StreamChunk` 折成完整的 `ContentBlock` 列表与终止状态。"""

    def __init__(self) -> None:
        self._partials: dict[int, _Partial] = {}
        self._order: list[int] = []
        self._usage: Usage | None = None
        self._finish: FinishReason | None = None

    # --- 喂入 -----------------------------------------------------------------

    def push(self, chunk: StreamChunk) -> None:
        if isinstance(chunk, BlockStart):
            if chunk.index not in self._partials:
                self._order.append(chunk.index)
                self._partials[chunk.index] = _Partial(block_type=chunk.block_type)
            return

        if isinstance(chunk, TextDelta):
            partial = self._ensure(chunk.index, "text")
            if partial.block is not None:
                return  # 已被 block-end 关闭：忽略迟到增量
            partial.text += chunk.text
            return

        if isinstance(chunk, ReasoningDelta):
            partial = self._ensure(chunk.index, "reasoning")
            if partial.block is not None:
                return
            partial.text += chunk.text
            return

        if isinstance(chunk, ToolCallDelta):
            partial = self._ensure(chunk.index, "tool-call")
            if partial.block is not None:
                return
            partial.tool_call_id = chunk.call_id
            if chunk.name:
                partial.tool_call_name = chunk.name
            partial.tool_call_arguments += chunk.arguments_delta
            return

        if isinstance(chunk, BlockEnd):
            partial = self._ensure(chunk.index, _block_kind(chunk.block))
            # 第一次关闭为准：重复关闭忽略，保证流式输出与最终 block 一致。
            if partial.block is not None:
                return
            partial.block = chunk.block
            return

        if isinstance(chunk, UsageChunk):
            self._usage = chunk.usage
            return

        if isinstance(chunk, StreamFinished):
            self._finish = chunk.reason
            return

    def _ensure(self, index: int, block_type: str) -> _Partial:
        partial = self._partials.get(index)
        if partial is None:
            partial = _Partial(block_type=block_type)
            self._partials[index] = partial
            self._order.append(index)
        return partial

    # --- 读取 -----------------------------------------------------------------

    def _assemble(self, partial: _Partial, index: int) -> ContentBlock:
        if partial.block is not None:
            return partial.block
        if partial.block_type == "text":
            return TextBlock(text=partial.text)
        if partial.block_type == "reasoning":
            return ReasoningBlock(text=partial.text)
        if partial.block_type == "tool-call":
            return ToolCallBlock(
                id=partial.tool_call_id or f"call-{index}",
                name=partial.tool_call_name or "",
                arguments=partial.tool_call_arguments,
            )
        raise ValueError(f'cannot assemble incomplete block of type "{partial.block_type}"')

    def blocks(self) -> list[ContentBlock]:
        """按流顺序组装所有已见 block。

        `max-tokens` 截断时丢弃 tool-call：被截断的调用参数不完整，执行它等于让模型
        用半句话触发副作用。
        """
        assembled = [
            self._assemble(self._partials[index], index) for index in self._order
        ]
        if self.finish.kind == "max-tokens":
            return [block for block in assembled if not isinstance(block, ToolCallBlock)]
        return assembled

    def interrupted_blocks(self) -> list[ContentBlock]:
        """中断时可以安全收尾的前缀：已关闭或仍开着的 text / reasoning，且非空白。

        丢弃 tool-call —— 中断发生在派发之前，保留它就必须伪造一个结果。
        """
        kept: list[ContentBlock] = []
        for index in self._order:
            partial = self._partials[index]
            block_type = (
                _block_kind(partial.block) if partial.block is not None else partial.block_type
            )
            if block_type not in ("text", "reasoning"):
                continue
            block = self._assemble(partial, index)
            if isinstance(block, (TextBlock, ReasoningBlock)) and block.text.strip() != "":
                kept.append(block)
        return kept

    @property
    def usage(self) -> Usage | None:
        return self._usage

    @property
    def finish(self) -> FinishReason:
        """返回终止原因；流未提供 finish 时使用 stop。"""
        return self._finish or FinishReason(kind="stop")

    def message(self) -> Message:
        blocks = self.blocks()
        return Message(
            role="assistant", content=_text_of(blocks), reasoning_content=_reasoning_of(blocks)
        )

    def turn(self) -> AssistantTurn:
        """组装结果。tool-call block 的原始 JSON 在这里解析成 arguments。"""
        blocks = self.blocks()
        return AssistantTurn(
            message=Message(
                role="assistant",
                content=_text_of(blocks),
                reasoning_content=_reasoning_of(blocks),
                tool_calls=[
                    _to_tool_call(block) for block in blocks if isinstance(block, ToolCallBlock)
                ],
            ),
            stop_reason=_STOP_REASON.get(self.finish.kind, "stop"),
            usage=self._usage,
        )


def _text_of(blocks: list[ContentBlock]) -> str:
    return "".join(block.text for block in blocks if isinstance(block, TextBlock))


def _reasoning_of(blocks: list[ContentBlock]) -> str:
    return "".join(block.text for block in blocks if isinstance(block, ReasoningBlock))


def _block_kind(block: ContentBlock) -> BlockType:
    """通过块类型识别已关闭消息块的类别。"""
    if isinstance(block, TextBlock):
        return "text"
    if isinstance(block, ReasoningBlock):
        return "reasoning"
    return "tool-call"


def _to_tool_call(block: ToolCallBlock):
    from .contracts import ToolCall

    return ToolCall(id=block.id, name=block.name, arguments=parse_arguments(block.arguments))


def parse_arguments(raw: str):
    """解析工具参数：空输入返回空字典，非法 JSON 保留原文供工具层报告错误。"""
    if raw == "":
        return {}
    import json

    try:
        parsed = json.loads(raw)
    except ValueError:
        return raw
    if isinstance(parsed, dict):
        return parsed
    return raw
