"""定义消息、工具、模型流、事件与终止状态，以及可选步骤和时间预算。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant", "tool"]
StopReason = Literal["stop", "tool_calls", "length", "error", "aborted"]
AgentStatus = Literal[
    "completed", "max_tokens", "aborted", "error", "budget_exceeded"
]
#: 工具并发模式，默认 exclusive。
ExecutionMode = Literal["parallel", "exclusive"]


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    #: 解析后的入参；非法 JSON 保留原文，由工具层报告入参错误。
    arguments: Any = field(default_factory=dict)


@dataclass(slots=True)
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    #: 标记因取消而截断的 assistant 可见前缀。
    interrupted: bool = False
    #: 宿主记录的消息归属信息，与发送给模型的内容分别保存。
    source: dict[str, Any] | None = None
    #: 推理内容独立于用户可见正文重放。
    reasoning_content: str = ""


@dataclass(frozen=True)
class Budget:
    """运行预算（与取消不同：配额用尽是正常终止，不是用户中断）。

    停止时保留已发生的工具结果与 usage，并在 error 里给出具体限额原因。
    """

    max_steps: int | None = None
    max_seconds: float | None = None

    def exceeded(self, steps: int, elapsed: float) -> str | None:
        """返回超限原因；未超限返回 `None`。"""
        if self.max_steps is not None and steps >= self.max_steps:
            return f"step budget exhausted: max_steps={self.max_steps}"
        if self.max_seconds is not None and elapsed >= self.max_seconds:
            return f"time budget exhausted: max_seconds={self.max_seconds}"
        return None


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int | None = None
    uncached_input_tokens: int | None = None
    reasoning_tokens: int | None = None

    def add(self, other: Usage | None) -> None:
        if other is None:
            return
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        for name in ("cached_input_tokens", "uncached_input_tokens", "reasoning_tokens"):
            value = getattr(other, name)
            if value is not None:
                setattr(self, name, (getattr(self, name) or 0) + value)


@dataclass(slots=True)
class RunControl:
    """维护任务预算与实际用量，并与工具调用的子任务共享用量对象。"""

    budget: Budget | None = None
    steps: int = 0
    started_at: float = field(default_factory=time.monotonic)
    usage: Usage = field(default_factory=Usage)


@dataclass(slots=True, frozen=True)
class ContextUsageAnchor:
    """保存请求与实际用量锚点，供后续请求和会话续接计算上下文增减。"""

    header: str
    surface_tokens: int
    tool_tokens: int
    usage_tokens: int


# --- 模型流协议 -----------------------------------------------------------
# block-start、delta、block-end 管理消息块生命周期；usage 和 finish 单独发出。

BlockType = Literal["text", "reasoning", "tool-call"]


@dataclass(slots=True, frozen=True)
class TextBlock:
    text: str


@dataclass(slots=True, frozen=True)
class ReasoningBlock:
    text: str


@dataclass(slots=True, frozen=True)
class ToolCallBlock:
    """模型请求的工具调用；参数保留原始 JSON 字符串，解析由工具层处理。"""

    id: str
    name: str
    arguments: str


ContentBlock = TextBlock | ReasoningBlock | ToolCallBlock


@dataclass(slots=True, frozen=True)
class LlmFailure:
    """可序列化的失败信息，供策略判断是否重试。"""

    message: str
    code: str
    status: int | None = None
    retry_after_ms: int | None = None


#: 通过 kind 区分终止原因的五类分支。
FinishKind = Literal["stop", "tool-calls", "max-tokens", "aborted", "error"]


@dataclass(slots=True, frozen=True)
class FinishReason:
    kind: FinishKind
    failure: LlmFailure | None = None


@dataclass(slots=True, frozen=True)
class BlockStart:
    """一个 block 开始。此后同 `index` 的 delta 属于它。"""

    index: int
    block_type: BlockType


@dataclass(slots=True, frozen=True)
class TextDelta:
    index: int
    text: str


@dataclass(slots=True, frozen=True)
class ReasoningDelta:
    index: int
    text: str


@dataclass(slots=True, frozen=True)
class ToolCallDelta:
    index: int
    call_id: str
    name: str | None = None
    arguments_delta: str = ""


@dataclass(slots=True, frozen=True)
class BlockEnd:
    """关闭消息块，并携带最终组装内容。"""

    index: int
    block: ContentBlock


@dataclass(slots=True, frozen=True)
class UsageChunk:
    """在流结束事件之前发出 token 用量。"""

    usage: Usage


@dataclass(slots=True, frozen=True)
class StreamFinished:
    """终止 chunk。正常结束、截断、取消、失败都走它，不是异常。"""

    reason: FinishReason


StreamChunk = (
    BlockStart
    | TextDelta
    | ReasoningDelta
    | ToolCallDelta
    | BlockEnd
    | UsageChunk
    | StreamFinished
)

#: BlockAssembler 生成的内部结果，用于循环控制与会话持久化。
@dataclass(slots=True)
class AssistantTurn:
    message: Message
    stop_reason: StopReason
    usage: Usage | None = None


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object"})


@dataclass(slots=True)
class ToolContext:
    """传给工具的执行上下文。cancelled 是取消探针，工具可在长任务里轮询。"""

    cancelled: Callable[[], bool] | None = None
    session_id: str | None = None
    agent: object | None = None
    defer_context: Callable[[Message], None] | None = None
    cancel_event: asyncio.Event | None = None
    run_control: RunControl | None = None
    tool_call_id: str | None = None


@dataclass(slots=True)
class ToolResult:
    content: str
    is_error: bool = False
    concludes_turn: bool = False
    #: 按工具结果顺序在下一 step 注入的附加上下文。
    additional_contexts: list[Message] = field(default_factory=list)


class Tool(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]
    #: 工具是否可安全重复执行。当前 loop 不重试工具，声明后才允许将来开启。
    idempotent: bool
    #: 并发模式；默认 exclusive，避免未声明的工具被并行。
    execution_mode: ExecutionMode

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult: ...


@dataclass(slots=True)
class ModelRequest:
    messages: list[Message]
    tools: list[ToolDefinition]
    cancelled: Callable[[], bool] | None = None
    #: 辅助调用用途，供适配器选择传输元数据或生成策略；普通对话留空。
    purpose: Literal["compaction", "session-title"] | None = None
    max_output_tokens: int | None = None


class ModelProvider(Protocol):
    """模型适配器的流式接口。

    通过 generate_stream 产出 StreamChunk；错误和取消统一表示为终止 finish 块。
    """

    def generate_stream(self, request: ModelRequest) -> AsyncIterator[StreamChunk]: ...


class ToolRegistryLike(Protocol):
    def list(self) -> list[ToolDefinition]: ...

    def execution_mode(self, name: str) -> ExecutionMode: ...

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult: ...


# --- 运行时事件（给调用方/UI 的流式输出）---------------------------------------


@dataclass(slots=True, frozen=True)
class TurnStarted:
    turn: int


@dataclass(slots=True, frozen=True)
class StepStarted:
    turn: int
    step: int


@dataclass(slots=True, frozen=True)
class AssistantTurnEvent:
    turn: int
    step: int
    message: Message


@dataclass(slots=True, frozen=True)
class AssistantTextDelta:
    """流式输出：模型文本增量，立即发给调用方。"""

    turn: int
    step: int
    text: str


@dataclass(slots=True, frozen=True)
class AssistantReasoningDelta:
    turn: int
    step: int
    text: str


@dataclass(slots=True, frozen=True)
class ProviderRetryScheduled:
    """记录已排定的重试及等待时间。"""

    turn: int
    step: int
    attempt: int
    delay: float
    code: str
    message: str


@dataclass(slots=True, frozen=True)
class ProviderRetryStarted:
    """下一次 attempt 开始；调用方重置本 turn/step 的临时推理预览。"""

    turn: int
    step: int
    attempt: int


@dataclass(slots=True, frozen=True)
class MemoryRecalled:
    """本 step 召回了哪些长期记忆（含来源，便于审计）。"""

    turn: int
    step: int
    items: list[MemoryItem]


@dataclass(slots=True, frozen=True)
class ContextCompacted:
    """记录摘要与保留原文条数。"""

    turn: int
    step: int
    summary: str
    compacted_messages: int
    context_tokens: int
    context_tokens_after: int | None = None


@dataclass(slots=True, frozen=True)
class CompactionWarning:
    turn: int
    step: int
    message: str
    context_tokens: int
    threshold_tokens: int
    attempts: int


@dataclass(slots=True, frozen=True)
class InboxClaimed:
    """记录本 step 认领的待处理输入。"""

    turn: int
    step: int
    messages: list[Message]


@dataclass(slots=True, frozen=True)
class ToolStarted:
    turn: int
    step: int
    call: ToolCall


@dataclass(slots=True, frozen=True)
class ToolFinished:
    turn: int
    step: int
    call: ToolCall
    result: ToolResult


@dataclass(slots=True, frozen=True)
class StepFinished:
    turn: int
    step: int


@dataclass(slots=True, frozen=True)
class TurnFinished:
    turn: int
    status: AgentStatus


@dataclass(slots=True, frozen=True)
class RunError:
    error: str


AgentEvent = (
    TurnStarted
    | StepStarted
    | AssistantTurnEvent
    | AssistantTextDelta
    | AssistantReasoningDelta
    | ProviderRetryScheduled
    | ProviderRetryStarted
    | MemoryRecalled
    | ContextCompacted
    | CompactionWarning
    | InboxClaimed
    | ToolStarted
    | ToolFinished
    | StepFinished
    | TurnFinished
    | RunError
)


# --- 长期记忆（Phase F）---------------------------------------------------------
# memory 与 compaction 的区别：compaction 压缩当前会话，memory 是跨会话的稳定事实。


@dataclass(slots=True)
class MemoryItem:
    key: str
    value: str
    #: 记忆来源（session id / 证据 id），召回时要标注。
    source: str = ""
    #: project / chip / user。
    scope: str = "project"
    updated_at: str = ""
    confidence: float = 1.0
    #: 适用条件（如 `{"chip": "PIC16F877A", "version": "v2"}`）：与当前上下文冲突时不召回。
    applies_to: dict[str, str] = field(default_factory=dict)
    #: 软失效时间戳；非空表示这条记忆已被纠正或配置变更推翻。
    invalidated_at: str | None = None

    @property
    def active(self) -> bool:
        return self.invalidated_at is None


# --- 会话事件（append-only 日志，可重放重建 messages）---------------------------


@dataclass(slots=True, frozen=True)
class TurnStartEntry:
    turn: int
    timestamp: str


@dataclass(slots=True, frozen=True)
class StepStartEntry:
    turn: int
    step: int
    timestamp: str


@dataclass(slots=True, frozen=True)
class MessageEntry:
    turn: int
    step: int
    message: Message
    timestamp: str


@dataclass(slots=True, frozen=True)
class StepEndEntry:
    turn: int
    step: int
    timestamp: str
    #: 本 step 累计的 token 使用（Phase E：每步记账）。
    usage: Usage | None = None
    context_anchor: ContextUsageAnchor | None = None


@dataclass(slots=True, frozen=True)
class CompactionEntry:
    """一次上下文压缩。原始事件不删除，重放时 = 摘要 + 最近原文。"""

    turn: int
    step: int
    summary: str
    first_kept_entry_id: str | None
    timestamp: str
    #: 保留区条数：重放时据此还原「摘要 + 最近原文」。
    kept_messages: int = 0


@dataclass(slots=True, frozen=True)
class TurnEndEntry:
    turn: int
    status: AgentStatus
    timestamp: str


@dataclass(slots=True, frozen=True)
class GoalChangeEntry:
    data: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass(slots=True, frozen=True)
class TodoWriteEntry:
    todos: list[dict[str, str]]
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


SessionEvent = (
    TurnStartEntry | StepStartEntry | MessageEntry | StepEndEntry | TurnEndEntry | CompactionEntry
    | GoalChangeEntry | TodoWriteEntry
)


class SessionSink(Protocol):
    """追加一个会话事件；实现可以返回节点 id（树形会话用它记录裁剪边界）。"""

    async def append(self, event: SessionEvent) -> str | None: ...
