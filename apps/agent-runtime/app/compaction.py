"""上下文压缩（Phase E）。

参照 DSH `packages/compaction/compaction-basic`：

- 触发阈值 `thresholdRatio=0.8`、保留尾部 `retainRatio=0.16`（DSH 默认）；
- 只在 step 边界触发（DSH 挂在 `agent/pre-step`）；
- 摘要结构沿用 DSH `summarizer.ts` 的八段；
- **原始事件不删除**：重放结果 = 摘要 + 最近原文；
- 压缩失败回退到未压缩上下文，不中断请求。
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

# --- checkpoint framing（DSH `summarizer.ts`）---------------------------------
#
# 上游把摘要落成一条 **user** 消息，用 tag 包裹，而不是替换 system prompt。
# 摘要内容由模型产生，变成 system 会改变指令优先级 —— 这不是命名差异。

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

#: DSH `summarizer.ts::COMPACTION_INSTRUCTION`。
#:
#: 它作为区域之后**最后一条 user 消息**发出，不作为另一个 system prompt ——
#: 前面的 system head / tools / 区域消息因此构成上次请求的真实前缀，prefix cache 可复用。
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
    """压缩策略。比例与默认值取自 DSH `compaction-basic/config.ts`。"""

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
    """DSH estimate.ts：字符/4 向上取整，加 role/block 开销；仍非精确分词。

    Python 的 tool result 是单独的消息，补上本地 name/call-id 的结构字段。
    """
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
    # DSH canonicalHeader 的本地适配：路由/模型/工具不同就不用旧 usage 校准。
    return json.dumps({
        "provider": type(provider).__qualname__,
        "base_url": getattr(provider, "base_url", None),
        "model": getattr(provider, "model", None),
        "tools": _tool_json(request.tools),
    }, sort_keys=True)


@dataclass(slots=True)
class ContextMeter:
    """DSH TokenMeter.measure 的薄适配：用量锚点 + 有符号的消息表面变化。

    锚点计入请求和生成结果；后续工具结果、用户输入以及压缩删除都形成增减量。
    不能把 max(上次 input, 当前估算) 当作校准，也不能将估算当成硬窗口保证。
    """

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
    """DSH：system prompt 不参与压缩。"""
    count = 0
    for message in messages:
        if message.role != "system":
            break
        count += 1
    return count


def is_compaction_summary(message: Message) -> bool:
    """识别 checkpoint。

    DSH 把 checkpoint 落成 **user** 消息并用 tag 包裹（`summarizer.ts::frameSummary`）。
    它必须能被下一次压缩再次吸收，所以不能当成固定 system prompt。
    """
    return message.role == "user" and SUMMARY_OPEN_TAG in message.content


def frame_checkpoint(summary: str) -> str:
    """DSH `frameSummary`：preamble + 开 tag + 摘要 + 闭 tag，落成一条 user 消息。"""
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
    """切出「要摘要的前段」和「保留的尾部原文」。

    语义对齐 DSH `region.ts:132-146`：`retainTokens` 是**最少**保留量，不是上限 ——

    1. 从尾向前累加，累计到 `>= retainTokens` 才停（允许最后一组超额）；
    2. 再向前扩展直到工具配对平衡（上游的 `toolPairingBalancedBefore` 循环）。

    本地 `message_groups()` 已经按「assistant call + 它的全部 result」成组，所以按组
    移动天然满足第 2 步：切点永远落在组边界上。

    与旧实现的差别：不再因为「下一组放不下」就停止保留。旧写法在最后一组超额时可以
    一条原文都不留，那与上游「至少保留这么多原文」是反的。
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
    """拼出摘要请求（DSH `region.ts:528-546` + `summarizer.ts:144-159`）。

    上游的形状是：**实际请求的固定前缀 + 区域消息 + 末尾一条 `COMPACTION_INSTRUCTION`
    的 user 消息**。不新增摘要专用 system prompt —— 这样辅助调用是上次请求的真实前缀，
    provider 的 prefix cache 可复用，只有末尾指令是新增输入。

    `prefix` 必须传**实际发出去的那几段固定前缀**：会话的 system head **以及本 turn
    注入的 memory block**（`loop._context_messages` 的位置）。只取 system head 会在
    开启记忆时让前缀与原请求不一致 —— 那等于没说前缀对齐。

    `flatten=True` 只用于无法接受 tool 消息的 provider（文本兜底，仍然序列化工具信息）。
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
    """用同一个 provider 生成摘要（DSH `summarizer.ts::summarizeWithLlm`）。

    DSH 把 `ctx.llm.stream()` 直接当摘要入口：replay 会话前缀（system prompt、
    **tools**、前导消息），把压缩指令作为最后一条 user 消息 —— 这样辅助调用是上次
    请求的真实前缀，provider 的前缀缓存可以复用，只有末尾指令是新增输入。

    本地保持同一形状：走 `generate_stream`（不是另一条 `generate` 路径），tools 与
    主请求一致，只把增量收进来拼成最终 `AssistantTurn`，不转发给调用方。

    DSH 还把 agent signal 继续传给摘要模型；这里同样接收 `cancelled` 探针：既放进
    `ModelRequest`（provider 适配器可在读取前放弃），也在等待时竞争取消，否则压缩
    期间卡住的摘要请求会让整个 run 停不下来。
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
        # DSH `finishError`：error / aborted / max-tokens 都是失败终止，压缩要回退。
        # max-tokens 单独处理，因为截断的 checkpoint 看起来「非空且更短」，会通过
        # 后面两条验收 —— 但它丢了后半段事实，必须当成失败。
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
        # DSH: "summarization produced no text summary content"
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
    """调用摘要器。

    DSH 的摘要请求带上会话的 tools 与实际请求前缀（system head + 注入的记忆块）。
    支持这些关键字的实现（`ModelSummarizer`）会收到它们；只接受 `transcript` 的
    自定义摘要器仍按位置参数调用。
    """
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
    """未达阈值或压缩失败时返回 `None`（调用方保持原上下文）。

    对应 DSH `region.ts::compactRegion` + `summarizer.ts`：

    - 落地结果的固定前缀只有会话原本的 system head；
    - **摘要请求的前缀**是实际请求前缀（`request_prefix` = system head + 本 turn 的
      memory block），否则开启记忆时摘要调用并不是主请求的前缀；
    - 旧 checkpoint 是**区域里的普通 user 消息**，因此随区域一起进入新摘要（滚动合并）；
    - 摘要非空、且比被替换区域更短，否则整次压缩失败并保持原上下文；
    - 落地的是 user checkpoint（带 `<compacted-summary>` framing），不是 system。

    `cancelled` 让摘要请求也接进运行的取消链路（DSH 把 agent signal 传进摘要器）。
    `on_usage` 在每次**实际发生**的摘要调用后收到一次 usage，与是否采用无关。
    """
    if context_tokens < policy.threshold_tokens:
        return None
    if policy.summarizer is None:
        return None

    prefix_end = leading_system_count(messages)
    fixed_prefix = messages[:prefix_end]
    summarize_prefix = list(fixed_prefix if request_prefix is None else request_prefix)

    region, kept = split_region(messages[prefix_end:], policy.retain_tokens)
    # 没有可压缩区域就不做（DSH keepFromIdx <= firstIdx）；旧 checkpoint 可以滚动压缩。
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
        return None  # DSH：压缩失败回退到未压缩上下文，不中断请求
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
