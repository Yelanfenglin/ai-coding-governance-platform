"""运行时契约：消息、工具、provider、事件与终止状态。

终止语义以 DSH 的模型/工具终止、max-tokens、取消、错误为基础。
本地额外提供可选 Budget（step / wall time），配额用尽与取消分开。
"""

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
#: 工具的并发模式（DSH `ToolExecutionMode`）。默认 exclusive，未声明的工具不会被并行。
ExecutionMode = Literal["parallel", "exclusive"]


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    #: 解析后的入参。正常是 `dict`；模型给出的参数不是合法 JSON 时保留**原始文本**
    #: （DSH `parseArguments` 的 `catch { return raw }`），由工具层报入参错误。
    arguments: Any = field(default_factory=dict)


@dataclass(slots=True)
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    #: DSH 的 `interrupted`：这段 assistant 输出被取消截断，是可见前缀而非完整回合。
    interrupted: bool = False
    #: Host-attested DSH message attribution; not part of the wire prompt.
    source: dict[str, Any] | None = None
    #: DeepSeek/DSH reasoning is replayed separately from user-visible text.
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
    """One task's budget and observed usage, shared with tool-invoked child runs.

    Step/time Budget is a local adaptation. The shared usage object follows
    OpenAI Agents SDK Agent.as_tool's nested ToolContext(usage=context.usage).
    """

    budget: Budget | None = None
    steps: int = 0
    started_at: float = field(default_factory=time.monotonic)
    usage: Usage = field(default_factory=Usage)


@dataclass(slots=True, frozen=True)
class ContextUsageAnchor:
    """DSH token-meter 的请求/用量锚点，供后续请求和会话续接计算有符号增量。"""

    header: str
    surface_tokens: int
    tool_tokens: int
    usage_tokens: int


# --- LLM 流协议（对齐 DSH `dsh-llm` 的 `StreamChunk`）---------------------------
#
# 上游把「一个 block 的生命周期」建模成显式事件：`block-start` 开块，若干带 `index`
# 的 delta 增量，`block-end` 关闭并携带组装好的 block；`usage` 与 `finish` 是独立
# chunk，终止原因（含 aborted / error）是**流里的数据**而不是异常。
#
# 这里逐个对齐，不做简化：见 DSH `packages/llm/llm/src/types.ts` 的 `StreamChunk`。

BlockType = Literal["text", "reasoning", "tool-call"]


@dataclass(slots=True, frozen=True)
class TextBlock:
    text: str


@dataclass(slots=True, frozen=True)
class ReasoningBlock:
    text: str


@dataclass(slots=True, frozen=True)
class ToolCallBlock:
    """模型请求的工具调用。

    `arguments` 是**原始 JSON 字符串**（DSH `ToolCallBlock.arguments`），不是解析后的
    dict。坏 JSON 保留原文交给工具层处理（DSH `parseArguments` 的 `catch { return raw }`），
    而不是让整个响应失败。
    """

    id: str
    name: str
    arguments: str


ContentBlock = TextBlock | ReasoningBlock | ToolCallBlock


@dataclass(slots=True, frozen=True)
class LlmFailure:
    """可序列化的失败事实；是否可重试由策略决定（DSH `LlmFailure`）。"""

    message: str
    code: str
    status: int | None = None
    retry_after_ms: int | None = None


#: DSH `FinishReasonMap` 的五个分支。本地用 `kind` 判别，等价于 TS 的 discriminated union。
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
    """关闭一个 block，携带组装好的 block（DSH：`block-end` 是权威值）。"""

    index: int
    block: ContentBlock


@dataclass(slots=True, frozen=True)
class UsageChunk:
    """token 记账。DSH 要求它在 terminal `finish` 之前发出。"""

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

#: 组装结果的内部表示。
#:
#: 只在 adapter 层之外使用（loop 判断是否继续、session 落盘）。**不是流协议的一部分** ——
#: DSH 没有这个结构，内容都在 `block-end` / `usage` / `finish` 里，用 `BlockAssembler` 折。
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
    #: DSH additionalContexts, accepted in tool-result order at the next step.
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
    #: 辅助调用的用途（DSH `GenerateOptions.purpose`）。适配器可映射成模型不可见的
    #: 传输元数据，或据此套用专门的生成策略；普通对话请求留空。
    purpose: Literal["compaction", "session-title"] | None = None
    max_output_tokens: int | None = None


class ModelProvider(Protocol):
    """模型适配器：**唯一入口是一条 `StreamChunk` 流**。

    对齐 DSH `LlmAdapter.stream()`：上游注释原文是 "Stream one model call as raw
    chunks. The only required method."。没有 `generate()` / `complete()` 之类的
    第二入口，也没有「一次性完成」的 fallback。

    终止（含 aborted / error）是流里的 `finish` chunk，**不是异常**。适配器内部可以
    抛错，但那由适配器自己（或调用方归一化层）翻译成 terminal `finish`，不会以
    异常的形式穿过消费方。
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
    """一次重试已排定、正在等待（DSH 的 `llm/retry`）。"""

    turn: int
    step: int
    attempt: int
    delay: float
    code: str
    message: str


@dataclass(slots=True, frozen=True)
class ProviderRetryStarted:
    """等待结束，下一次 attempt 开始（DSH 的 `llm/retry-started`）。"""

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
    """上下文被压缩：摘要 + 保留的原文条数（DSH 的 compaction 事件）。"""

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
    """从 Inbox 取走了一批待处理输入，即将进入本 step（DSH `agent/inbox/claimed`）。"""

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
