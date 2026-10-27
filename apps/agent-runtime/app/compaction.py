"""在 step 边界按上下文压力压缩历史。

默认触发比例为 0.8，原文保留比例为 0.16；摘要采用八段结构。
原始事件持续保存，重放时组合摘要与最近原文；压缩失败时沿用原上下文。
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field

from .assembler import BlockAssembler
from .contracts import (
    AssistantTurn,
    ContextUsageAnchor,
    Message,
    ModelProvider,
    ModelRequest,
    ToolDefinition,
    Usage,
)
from .errors import ErrorCode, ProviderError, has_code
from .observability import observed

# --- 摘要检查点包装 -------------------------------------------------------
# 摘要以带标签的 user 消息写入，保留原 system 指令的优先级。

SUMMARY_OPEN_TAG = "<compacted-summary>"
SUMMARY_CLOSE_TAG = "</compacted-summary>"

CHECKPOINT_PREAMBLE = (
    "This is an automatically generated checkpoint condensing an earlier span of the "
    "conversation to free up context. Treat the captured context as established background "
    "and build on it without restating it. Continue the task directly from the messages "
    "that follow, without acknowledging this checkpoint."
)

#: 摘要器消费流时的结束哨兵（`StopAsyncIteration` 不能直接当值传递）。
_STREAM_DONE = object()

#: 摘要请求与取消竞争时的轮询间隔（摘要器只拿到探针，拿不到 event）。
_CANCEL_POLL_SECONDS = 0.01

#: 摘要指令放在区域消息之后，保留 system、tools 与记忆组成的请求前缀。
COMPACTION_INSTRUCTION = "\n".join(
    [
        (
            "You are now acting as a compaction engine for this AI coding assistant. "
            "Condense the conversation ABOVE into a structured checkpoint that lets another "
            "model resume the work with no loss of essential context."
        ),
        "",
        (
            "Output EXACTLY the Markdown structure below: keep every section, in order. "
            "Use terse bullets, not prose paragraphs. Write \"(none)\" for an empty "
            "section — never drop a section."
        ),
        "",
        "## Primary Request and Intent",
        (
            "- [the user's original and evolving goals; quote verbatim where the exact "
            "wording matters]"
        ),
        "",
        "## Key Technical Concepts",
        "- [technologies, frameworks, patterns, and conventions in play]",
        "",
        "## Files and Code",
        "- [exact path: why it matters, key changes or snippets]",
        "",
        "## Errors and Fixes",
        "- [error: how it was resolved, plus any related user feedback]",
        "",
        "## Pending Jobs",
        "- [explicitly requested work not yet completed]",
        "",
        "## Current Work",
        "- [precisely what was in progress at this checkpoint]",
        "",
        "## Next Step",
        "- [the single next action, directly in line with the most recent request, or \"(none)\"]",
        "",
        "## Critical Context",
        "- [decisions and their rationale, constraints, user preferences, open questions, "
        "data needed to continue]",
        "",
        "Rules:",
        (
            "- Write concise English engineering prose. Preserve exact file paths, commands, "
            "error strings, identifiers, numeric values, function signatures, and syntax "
            "fragments."
        ),
        "- Capture user feedback and explicit instructions faithfully, especially corrections.",
        "- Do NOT mention this summarization request or that the context was compacted.",
        "- Output only the checkpoint text: do not call any tool or take any other action.",
        (
            f"- If the conversation already contains a {SUMMARY_OPEN_TAG} block, it is a "
            "PRIOR checkpoint. Do not copy it forward verbatim: preserve still-true facts, "
            "drop stale ones, and merge newer information into a single consolidated "
            "summary under the same structure."
        ),
    ]
)

Summarizer = Callable[[list[Message]], Awaitable[str]]

#: 摘要结构（八段）。从 `COMPACTION_INSTRUCTION` 派生标题，避免两处漂移。
SUMMARY_SECTIONS: tuple[str, ...] = tuple(
    line for line in COMPACTION_INSTRUCTION.splitlines() if line.startswith("## ")
)


@dataclass(frozen=True)
class CompactionPolicy:
    """配置压缩触发比例、原文保留比例及摘要预算。"""

    context_window: int
    threshold_ratio: float = 0.8
    retain_ratio: float = 0.16
    summarizer: Summarizer | None = None
    compaction_retries: int = 1

    @property
    def threshold_tokens(self) -> int:
        return int(self.context_window * self.threshold_ratio)

    @property
    def retain_tokens(self) -> int:
        return int(self.context_window * self.retain_ratio)


@dataclass(slots=True)
class CompactionOutcome:
    summary: str
    compacted_messages: int
    messages: list[Message]
    context_tokens: int
    #: 保留区（原样留下的尾部），供会话记录裁剪边界与重放。
    kept: list[Message] = field(default_factory=list)
    usage: Usage | None = None

    @property
    def first_kept(self) -> Message | None:
        return self.kept[0] if self.kept else None


def estimate_tokens(message: Message) -> int:
    """按字符数除以 4 向上取整，并计入角色、块及工具调用字段的结构开销。"""
    if message.role == "system":
        return math.ceil(len(message.content) / 4) + 4 if message.content else 0
    tokens = 4
    if message.content:
        tokens += math.ceil(len(message.content) / 4) + 4
    if message.reasoning_content:
        tokens += math.ceil(len(message.reasoning_content) / 4) + 4
    for call in message.tool_calls:
        arguments = json.dumps(call.arguments, ensure_ascii=False, separators=(",", ":"))
        tokens += math.ceil(len(call.name) / 4) + math.ceil(len(arguments) / 4) + 4
        tokens += math.ceil(len(call.id) / 4)
    if message.role == "tool":
        tokens += 4 + math.ceil(len(message.tool_call_id or "") / 4)
        tokens += math.ceil(len(message.name or "") / 4)
    return tokens


def estimate_context(messages: list[Message]) -> int:
    return sum(estimate_tokens(message) for message in messages)


def _tool_json(tools: list[ToolDefinition]) -> str:
    return json.dumps([asdict(tool) for tool in tools], ensure_ascii=False, sort_keys=True)


def estimate_request(request: ModelRequest) -> int:
    """完整请求表面：含临时记忆和工具 schema。"""
    return estimate_context(request.messages) + _tool_tokens(request.tools)


def _tool_tokens(tools: list[ToolDefinition]) -> int:
    return math.ceil(len(_tool_json(tools)) / 4) + 4 if tools else 0


def _request_header(request: ModelRequest, provider: ModelProvider) -> str:
    # 路由、模型或工具变化时重新建立 usage 校准锚点。
    return json.dumps({
        "provider": type(provider).__qualname__,
        "base_url": getattr(provider, "base_url", None),
        "model": getattr(provider, "model", None),
        "tools": _tool_json(request.tools),
    }, sort_keys=True)


@dataclass(slots=True)
class ContextMeter:
    """结合实际请求用量锚点和后续上下文增减，估算当前 token 占用。"""

    anchor: ContextUsageAnchor | None = None

    def measure(self, request: ModelRequest, provider: ModelProvider) -> int:
        surface = estimate_context(request.messages)
        anchor = self.anchor
        if anchor is None or anchor.header != _request_header(request, provider):
            return surface + _tool_tokens(request.tools)
        baseline = max(anchor.usage_tokens, anchor.surface_tokens + anchor.tool_tokens)
        return max(0, baseline + surface - anchor.surface_tokens)

    def observe(
        self, request: ModelRequest, provider: ModelProvider, turn: AssistantTurn
    ) -> None:
        usage = turn.usage
        self.anchor = ContextUsageAnchor(
            header=_request_header(request, provider),
            surface_tokens=estimate_context([*request.messages, turn.message]),
            tool_tokens=_tool_tokens(request.tools),
            usage_tokens=usage.input_tokens + usage.output_tokens if usage is not None else 0,
        )


def leading_system_count(messages: list[Message]) -> int:
    """统计前导 system 消息，作为压缩时保留的固定前缀。"""
    count = 0
    for message in messages:
        if message.role != "system":
            break
        count += 1
    return count


def is_compaction_summary(message: Message) -> bool:
    """识别带摘要标签的 user 检查点，供后续压缩滚动合并。"""
    return message.role == "user" and SUMMARY_OPEN_TAG in message.content


def frame_checkpoint(summary: str) -> str:
    """用前导说明和摘要标签包装摘要，生成 user 检查点消息。"""
    return f"{CHECKPOINT_PREAMBLE}\n\n{SUMMARY_OPEN_TAG}{summary}{SUMMARY_CLOSE_TAG}"


def message_groups(messages: list[Message]) -> list[list[Message]]:
    """把一条 assistant tool call 与它的全部 tool result 绑成一组。

    压缩按组切，避免「保留结果、摘要掉发起调用」这种破坏请求结构的组合。
    """
    groups: list[list[Message]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role == "assistant" and message.tool_calls:
            expected = {call.id for call in message.tool_calls}
            group = [message]
            seen: set[str] = set()
            cursor = index + 1
            while cursor < len(messages) and messages[cursor].role == "tool":
                group.append(messages[cursor])
                if messages[cursor].tool_call_id is not None:
                    seen.add(messages[cursor].tool_call_id)
                cursor += 1
                if expected.issubset(seen):
                    break
            index = cursor
        else:
            group = [message]
            index += 1
        groups.append(group)
    return groups


def split_region(
    messages: list[Message], retain_tokens: int
) -> tuple[list[Message], list[Message]]:
    """切分待摘要区域与保留原文。

    从尾部按消息组累计至 retainTokens，允许最后一组超出预算。
    工具调用与结果按组保留，使切点落在完整配对边界。
    """
    groups = message_groups(messages)

    kept: list[Message] = []
    used = 0
    for group in reversed(groups):
        kept[:0] = group
        used += sum(estimate_tokens(message) for message in group)
        if used >= retain_tokens:
            break

    cut = len(messages) - len(kept)
    return messages[:cut], kept


def summarization_messages(
    transcript: list[Message],
    *,
    prefix: Sequence[Message] = (),
    flatten: bool = False,
) -> list[Message]:
    """组合摘要请求：实际固定前缀、待压缩消息和末尾摘要指令。

    prefix 包含会话 system 与本回合记忆，保持请求前缀一致以复用缓存。
    flatten 模式将工具信息序列化为文本，供不接受工具消息的适配器使用。
    """
    head = list(prefix)
    if flatten:
        body = f"{render_transcript(transcript)}\n\n{COMPACTION_INSTRUCTION}"
        return [*head, Message(role="user", content=body)]
    return [*head, *transcript, Message(role="user", content=COMPACTION_INSTRUCTION)]


def render_transcript(messages: list[Message]) -> str:
    """把消息渲染成文本兜底 transcript：工具名称、参数、call id 都要序列化出来。"""
    lines: list[str] = []
    for message in messages:
        if message.role == "tool":
            target = message.tool_call_id or "?"
            name = f"/{message.name}" if message.name else ""
            lines.append(f"tool{name}[id={target}]: {message.content}")
            continue
        if message.content:
            lines.append(f"{message.role}: {message.content}")
        if message.reasoning_content:
            lines.append(f"{message.role} reasoning: {message.reasoning_content}")
        for call in message.tool_calls:
            arguments = json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)
            lines.append(f"{message.role}: tool_call {call.name}({arguments}) [id={call.id}]")
        if not message.content and not message.reasoning_content and not message.tool_calls:
            lines.append(f"{message.role}:")
    return "\n".join(lines)


class ModelSummarizer:
    """通过同一流式 provider 生成摘要。

    沿用主请求的固定前缀与工具定义，收集增量并组装摘要结果。
    摘要请求接入运行取消信号，在等待读取期间也可中断。
    """

    def __init__(self, provider: ModelProvider, max_output_tokens: int = 8192) -> None:
        self._provider = provider
        self.max_output_tokens = max_output_tokens
        #: **本次**调用的 usage；进入 `__call__` 时先清空，避免把上一次调用（或被拒绝
        #: 的调用）的用量重复记账。
        self.last_usage: Usage | None = None
        self._stream_usage: Usage | None = None

    async def __call__(
        self,
        transcript: list[Message],
        cancelled: Callable[[], bool] | None = None,
        tools: list[ToolDefinition] | None = None,
        prefix: Sequence[Message] = (),
    ) -> str:
        self.last_usage = None
        self._stream_usage = None
        request = ModelRequest(
            messages=summarization_messages(transcript, prefix=prefix),
            tools=list(tools or ()),
            cancelled=cancelled,
            purpose="compaction",
            max_output_tokens=self.max_output_tokens,
        )
        try:
            turn = await self._collect(request, cancelled)
        except BaseException:
            # 失败/截断/取消同样消耗了 token：留下本次流里报出的 usage，
            # 由调用方统一记账，而不是因为它没被采用就整笔丢掉。
            self.last_usage = self._stream_usage
            raise
        self.last_usage = turn.usage
        return turn.message.content

    async def _collect(
        self, request: ModelRequest, cancelled: Callable[[], bool] | None
    ) -> AssistantTurn:
        """消费同一条 chunk 流，折成最终 `AssistantTurn`（不转发增量）。"""
        assembler = BlockAssembler()
        iterator = self._provider.generate_stream(request).__aiter__()
        try:
            while True:
                pending = asyncio.ensure_future(self._anext(iterator))
                value = await self._settle(pending, cancelled)
                if value is _STREAM_DONE:
                    break
                assembler.push(value)  # type: ignore[arg-type]
        finally:
            await self._close(iterator)
            self._stream_usage = assembler.usage

        reason = assembler.finish
        # error、aborted 或 max-tokens 终止均回退，避免采用截断的摘要。
        if reason.kind in ("error", "aborted", "max-tokens"):
            failure = reason.failure
            if reason.kind == "max-tokens":
                raise ProviderError(
                    "summarization truncated at the token cap (incomplete checkpoint)",
                    ErrorCode.PROVIDER,
                )
            raise ProviderError(
                failure.message if failure is not None else "summarization failed",
                ErrorCode.ABORTED if reason.kind == "aborted" else ErrorCode.PROVIDER,
            )
        turn = assembler.turn()
        # 空白摘要保留原上下文。
        if not turn.message.content.strip():
            raise ProviderError(
                "summarization produced no text summary content", ErrorCode.PROVIDER
            )
        return turn

    @staticmethod
    async def _anext(iterator: object) -> object:
        try:
            return await iterator.__anext__()  # type: ignore[attr-defined]
        except StopAsyncIteration:
            return _STREAM_DONE

    @staticmethod
    async def _close(iterator: object) -> None:
        aclose = getattr(iterator, "aclose", None)
        if aclose is None:
            return
        with contextlib.suppress(BaseException):
            await aclose()

    @staticmethod
    async def _settle(
        pending: asyncio.Future[object], cancelled: Callable[[], bool] | None
    ) -> object:
        if cancelled is None:
            return await pending
        # 摘要器只拿到 cancel 探针（拿不到 event），所以用短轮询与取消竞争。
        while True:
            try:
                await asyncio.wait_for(asyncio.shield(pending), timeout=_CANCEL_POLL_SECONDS)
            except TimeoutError:
                if cancelled():
                    pending.cancel()
                    with contextlib.suppress(BaseException):
                        await pending
                    raise ProviderError("summarization cancelled", ErrorCode.ABORTED) from None
                continue
            except BaseException:
                pending.cancel()
                with contextlib.suppress(BaseException):
                    await pending
                raise
            return pending.result()


async def _call_summarizer(
    summarizer: Callable[..., Awaitable[str]],
    transcript: list[Message],
    cancelled: Callable[[], bool] | None,
    tools: list[ToolDefinition] | None = None,
    prefix: Sequence[Message] = (),
) -> str:
    """调用摘要器，并向支持相应参数的实现传入工具定义与实际请求前缀。"""
    kwargs: dict[str, object] = {}
    if cancelled is not None and _accepts_keyword(summarizer, "cancelled"):
        kwargs["cancelled"] = cancelled
    if tools and _accepts_keyword(summarizer, "tools"):
        kwargs["tools"] = tools
    if prefix and _accepts_keyword(summarizer, "prefix"):
        kwargs["prefix"] = prefix
    return await summarizer(transcript, **kwargs)  # type: ignore[call-arg]


def _accepts_keyword(summarizer: Callable[..., object], name: str) -> bool:
    try:
        parameters = inspect.signature(summarizer).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _reset_summary_usage(summarizer: Callable[..., Awaitable[str]]) -> None:
    """调用前清空摘要器自报的 usage，保证 `last_usage` 只表示**本次**调用。

    自定义摘要器不会自己重置，若上次调用失败，`last_usage` 会留着上一笔 ——
    在 `finally` 里直接读就会重复记账。
    """
    if hasattr(summarizer, "last_usage"):
        summarizer.last_usage = None  # type: ignore[attr-defined]


def _report_summary_usage(
    summarizer: Callable[..., Awaitable[str]],
    on_usage: Callable[[Usage], None] | None,
) -> None:
    """把**本次**摘要调用的 usage 交给调用方记账。

    记账与「是否采用 checkpoint」解耦：摘要被非空/缩短验收拒绝时，这次调用已经真实
    计费，不能因为它没被采用就消失。`ModelSummarizer` 在进入 `__call__` 时清空
    `last_usage`，所以这里读到的一定是本次调用的值，不会重复累加上一次的。
    """
    if on_usage is None:
        return
    usage = getattr(summarizer, "last_usage", None)
    if isinstance(usage, Usage):
        on_usage(usage)


@observed("context.compaction")
async def compact_context(
    messages: list[Message],
    policy: CompactionPolicy,
    context_tokens: int,
    cancelled: Callable[[], bool] | None = None,
    tools: list[ToolDefinition] | None = None,
    request_prefix: Sequence[Message] | None = None,
    on_usage: Callable[[Usage], None] | None = None,
    on_failure: Callable[[str], None] | None = None,
) -> CompactionOutcome | None:
    """将可压缩消息替换为非空且更短的 user 摘要检查点，并保留最近原文。

    未达阈值或摘要无效时返回 None；旧检查点随区域滚动合并。
    请求携带实际固定前缀与取消信号，每次实际摘要调用均回报 usage。
    """
    if context_tokens < policy.threshold_tokens:
        return None
    if policy.summarizer is None:
        return None

    prefix_end = leading_system_count(messages)
    fixed_prefix = messages[:prefix_end]
    summarize_prefix = list(fixed_prefix if request_prefix is None else request_prefix)

    region, kept = split_region(messages[prefix_end:], policy.retain_tokens)
    # 仅处理存在可压缩区域的上下文；旧检查点参与滚动合并。
    if not region:
        return None

    try:
        _reset_summary_usage(policy.summarizer)
        summary = await _call_summarizer(
            policy.summarizer, region, cancelled, tools, summarize_prefix
        )
    except (asyncio.CancelledError, ProviderError) as error:
        # 取消 / 中断必须向上传播：压缩失败可以回退，但"停不下来"不能靠回退掩盖。
        # 按 code 的**值**判断：`ProviderError.code` 是 str，用 `is` 比枚举永远为 False。
        if isinstance(error, ProviderError) and not has_code(error, ErrorCode.ABORTED):
            if on_failure is not None:
                on_failure(str(error))
            return None
        raise
    except Exception as error:
        if on_failure is not None:
            on_failure(str(error))
        return None  # 压缩失败时沿用原上下文。
    finally:
        _report_summary_usage(policy.summarizer, on_usage)

    # C06 / summarizer.ts:167：非空验收。空白摘要不能替换有效历史。
    if not summary.strip():
        if on_failure is not None:
            on_failure("summarization produced no text summary content")
        return None

    checkpoint = Message(role="user", content=frame_checkpoint(summary))

    # C07 / region.ts:398-406：缩短验收。摘要必须比被替换区域更小，否则压缩反而加压。
    if estimate_tokens(checkpoint) >= sum(estimate_tokens(message) for message in region):
        if on_failure is not None:
            on_failure("summary is not smaller than the replaced region")
        return None

    usage = getattr(policy.summarizer, "last_usage", None)
    return CompactionOutcome(
        summary=summary,
        compacted_messages=len(region),
        messages=[*fixed_prefix, checkpoint, *kept],
        context_tokens=context_tokens,
        kept=kept,
        usage=usage if isinstance(usage, Usage) else None,
    )
