"""按 turn 和 step 两层组织模型与工具执行。

turn 处理用户输入及插话，step 执行一次模型请求与一组工具调用。
根据模型、工具完成标记、长度、取消、错误及可选预算结束运行。
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from . import metrics
from .assembler import BlockAssembler
from .circuit import CircuitBreakers, CircuitOpen, Outcome, Permit, provider_outcome
from .compaction import (
    CompactionPolicy,
    ContextMeter,
    compact_context,
    leading_system_count,
)
from .contracts import (
    AgentEvent,
    AgentStatus,
    AssistantReasoningDelta,
    AssistantTextDelta,
    AssistantTurn,
    AssistantTurnEvent,
    Budget,
    CompactionEntry,
    CompactionWarning,
    ContextCompacted,
    ContextUsageAnchor,
    FinishReason,
    InboxClaimed,
    MemoryItem,
    MemoryRecalled,
    Message,
    MessageEntry,
    ModelProvider,
    ModelRequest,
    ProviderRetryScheduled,
    ProviderRetryStarted,
    ReasoningBlock,
    ReasoningDelta,
    RunControl,
    RunError,
    SessionEvent,
    SessionSink,
    StepEndEntry,
    StepFinished,
    StepStarted,
    StepStartEntry,
    StreamChunk,
    StreamFinished,
    TextBlock,
    TextDelta,
    ToolCall,
    ToolContext,
    ToolFinished,
    ToolRegistryLike,
    ToolResult,
    ToolStarted,
    TurnEndEntry,
    TurnFinished,
    TurnStarted,
    TurnStartEntry,
    Usage,
)
from .errors import ErrorCode, ProviderError, as_provider_error, has_code
from .goal_driver import TOOL_INITIATOR, GoalRoundDriver
from .goal_prompts import goal_guidance
from .inbox import Inbox
from .memory import JsonMemoryStore
from .observability import failure as record_failure
from .observability import observe
from .retry import RetryPolicy

ABORTED_BEFORE_DISPATCH = "Error: tool call aborted before dispatch"

#: 已派发但没等到结果（被取消）的调用，收尾时补的 synthetic 结果。
#: 不能重跑工具来补日志：副作用已经可能发生过，重跑等于重复执行。
ABORTED_IN_FLIGHT = "Error: tool call was cancelled before it returned a result"

#: 取消时等待在途结果写入完成的最长时间，单位为秒。
COMMIT_GRACE_SECONDS = 1.0

#: `_await_or_cancel` 用来表示「流已结束」的内部哨兵。
_STREAM_END = object()


@dataclass(slots=True)
class AgentRunResult:
    status: AgentStatus
    messages: list[Message]
    turns: int
    steps: int
    usage: Usage = field(default_factory=Usage)
    error: str | None = None
    total_steps: int = 0


@dataclass(slots=True)
class RunOptions:
    provider: ModelProvider
    tools: ToolRegistryLike
    messages: list[Message]
    cancel_event: asyncio.Event | None = None
    on_event: Callable[[AgentEvent], None] | None = None
    session: SessionSink | None = None
    #: 模型请求重试策略；None 表示不重试。
    retry_policy: RetryPolicy | None = None
    #: 依赖级熔断注册表（进程内按依赖键共享）；None 表示不启用。
    circuit: CircuitBreakers | None = None
    #: 本 provider 的依赖键（`model:{base_url}|{model}`）；熔断启用时必填。
    circuit_dependency: str | None = None
    #: 并行工具调用池上限。
    max_parallel_tool_calls: int = 10
    #: 运行中插话/追问的输入队列；None 表示不启用 steering。
    inbox: Inbox | None = None
    #: 取消时是否清空待处理输入队列，默认清空。
    clear_inbox_on_cancel: bool = True
    #: 上下文压缩策略；None 表示不压缩。
    compaction: CompactionPolicy | None = None
    #: 项目事实库，None 表示跳过记忆注入。
    memory: JsonMemoryStore | None = None
    #: 用户画像库。
    user_profile: JsonMemoryStore | None = None
    #: 当前适用条件（如 `{"chip": ...}`）；与之冲突的记忆不召回。
    applies_to: dict[str, str] | None = None
    #: 每个库最多注入多少条。
    memory_limit: int = 20
    #: 运行预算（步数 / 墙钟时间）。用尽 → `budget_exceeded`，保留已发生的结果与 usage。
    budget: Budget | None = None
    #: Host-owned task accounting; delegated runs inherit the same object.
    run_control: RunControl | None = None
    #: Local adaptation: a child activation may have a smaller allowance while
    #: sharing the task ledger. Reservation protects the parent's final response.
    local_max_steps: int | None = None
    reserved_parent_steps: int = 0
    #: 续接时从会话恢复的测量锚点；旧日志没有该字段时重新估算完整请求。
    context_anchor: ContextUsageAnchor | None = None
    #: 可选目标状态与轮次调度器。
    goal_driver: GoalRoundDriver | None = None


@dataclass(slots=True)
class _StepOutcome:
    status: AgentStatus
    steps: int
    error: str | None = None


# --- 公开入口 -----------------------------------------------------------------


async def run_agent(options: RunOptions) -> AgentRunResult:
    with observe("agent.run", "AGENT") as span:
        owns_control = options.run_control is None
        control = options.run_control or RunControl(
            budget=options.budget, started_at=time.monotonic(),
        )
        options = replace(options, inbox=options.inbox or Inbox(), run_control=control,
                          budget=control.budget)
        result = await _run_with_driver(options)
        result.total_steps = control.steps
        if owns_control:
            result.usage = control.usage
        span.set_attributes({"agent.status": result.status, "agent.steps": result.steps})
        return result


async def _run_with_driver(options: RunOptions) -> AgentRunResult:
    driver = options.goal_driver
    if driver is None:
        return await _run_agent(options)
    options = replace(options, cancel_event=options.cancel_event or asyncio.Event())
    driver.running = True
    driver.cancel = options.cancel_event.set
    driver.bind_inbox(options.inbox)
    try:
        return await _run_agent(options)
    finally:
        driver.close()


async def _run_agent(options: RunOptions) -> AgentRunResult:
    messages: list[Message] = list(options.messages)
    usage = Usage()
    turns = 0
    steps = 0

    # 当前回合结束后继续处理 next-turn 输入；步骤与时间预算跨回合累计。
    started_at = options.run_control.started_at
    entry_ids: dict[int, str] = {}
    meter = ContextMeter(options.context_anchor)

    while True:
        turns += 1
        _emit(options, TurnStarted(turns))
        await _append(options, TurnStartEntry(turns, _now()))
        if turns == 1 and options.goal_driver is not None and options.messages:
            # Python callers already supply the first accepted input in messages;
            # later human/goal inputs enter through the inbox and its source tag.
            options.goal_driver.note_input(options.messages[-1])

        # 每个回合重建记忆快照，仅用于当次请求上下文。
        memory_block = await _build_memory_block(options, turns)

        try:
            outcome = await _run_steps(
                options, messages, usage, turns, steps, entry_ids, memory_block, started_at, meter
            )
        except asyncio.CancelledError:
            # 取消时写入回合结束事件，保留可重放的运行边界。
            _emit(options, TurnFinished(turns, "aborted"))
            with contextlib.suppress(BaseException):
                await _append(options, TurnEndEntry(turns, "aborted", _now()))
                if options.goal_driver is not None:
                    await options.goal_driver.after_turn("aborted")
            raise
        steps = outcome.steps

        _emit(options, TurnFinished(turns, outcome.status))
        await _append(options, TurnEndEntry(turns, outcome.status, _now()))
        if options.goal_driver is not None:
            await options.goal_driver.after_turn(outcome.status)

        if outcome.status != "completed":
            if (
                options.inbox is not None
                and options.clear_inbox_on_cancel
                and outcome.status == "aborted"
            ):
                options.inbox.clear()
            return AgentRunResult(
                status=outcome.status,
                messages=messages,
                turns=turns,
                steps=steps,
                usage=usage,
                error=outcome.error,
            )

        if options.goal_driver is not None and options.inbox is not None:
            followup = await options.goal_driver.reserve(options.inbox)
            if followup is not None:
                options.inbox.followup(followup)
        if options.inbox is None or not options.inbox.has_pending:
            return AgentRunResult(
                status="completed", messages=messages, turns=turns, steps=steps, usage=usage
            )


# --- step 循环 ----------------------------------------------------------------


async def _build_memory_block(options: RunOptions, turn: int) -> Message | None:
    """为本回合构建固定记忆快照，并在组装请求时插入 system 之后。

    快照独立于会话历史；没有匹配条目时返回 None。
    """
    stores = [store for store in (options.memory, options.user_profile) if store is not None]
    if not stores:
        return None

    blocks: list[str] = []
    items: list[MemoryItem] = []
    for store in stores:
        block = await store.render(applies_to=options.applies_to, limit=options.memory_limit)
        if block:
            blocks.append(block)
            items.extend(
                await store.recall(applies_to=options.applies_to, limit=options.memory_limit)
            )

    if not blocks:
        return None

    _emit(options, MemoryRecalled(turn=turn, step=0, items=items))
    return Message(role="system", content="\n\n".join(blocks))


def _budget_reason(options: RunOptions, local_steps: int, started_at: float) -> str | None:
    budget = options.budget
    if budget is not None:
        reason = budget.exceeded(options.run_control.steps, time.monotonic() - started_at)
        if reason is not None:
            return reason
        if (budget.max_steps is not None and options.reserved_parent_steps
                and options.run_control.steps >= budget.max_steps - options.reserved_parent_steps):
            return (f"child stopped: {options.reserved_parent_steps} "
                    "parent response step(s) reserved")
    if options.local_max_steps is not None and local_steps >= options.local_max_steps:
        return f"child step allowance exhausted: max_steps={options.local_max_steps}"
    return None


def _budget_message(options: RunOptions, used_steps: int, started_at: float,
                    local_steps: int = 0) -> Message | None:
    """将剩余步骤与时间预算提供给模型上下文。"""
    budget = options.budget
    if budget is None and options.local_max_steps is None:
        return None
    limits = []
    if budget is not None and budget.max_steps is not None:
        limits.append(
            f"Steps: limit={budget.max_steps}; used={used_steps}; "
            f"remaining={max(0, budget.max_steps - used_steps)} (including this response)."
        )
        if options.reserved_parent_steps:
            limits.append(f"Parent response steps reserved: {options.reserved_parent_steps}.")
    if options.local_max_steps is not None:
        limits.append(
            f"Child activation steps: limit={options.local_max_steps}; "
            f"remaining={max(0, options.local_max_steps - local_steps)}."
        )
    if budget is not None and budget.max_seconds is not None:
        elapsed = time.monotonic() - started_at
        limits.append(
            f"Seconds: limit={budget.max_seconds:.1f}; elapsed={elapsed:.1f}; "
            f"remaining={max(0.0, budget.max_seconds - elapsed):.1f}."
        )
    if not limits:
        return None
    return Message(role="user", content=(
        "<run_budget>\nHost runtime budget snapshot; shared across all turns and delegated runs.\n"
        + "\n".join(limits)
        + "\nA step is one model response and its tool calls; retries and compaction do not "
        "consume extra steps. All elapsed time counts. Limits are checked between steps.\n"
        "Reserve a model response to deliver verified results. Tool results need a later "
        "response to inspect and report them. With one step remaining, use this response "
        "to give the final answer from established evidence and state any unfinished work. "
        "Fit work to the remaining time; avoid optional investigations near the limit. "
        "Do not claim unverified completion or change the task's acceptance criteria.\n"
        "</run_budget>"
    ))


def _context_messages(messages: list[Message], memory_block: Message | None,
                      goal_driver: GoalRoundDriver | None = None,
                      budget_message: Message | None = None) -> list[Message]:
    """Fixed prefix + history + current request-only runtime budget snapshot."""
    tail = [] if budget_message is None else [budget_message]
    if memory_block is None and goal_driver is None:
        return [*messages, *tail]
    prefix = leading_system_count(messages)
    head = list(messages[:prefix])
    if goal_driver is not None:
        policy = goal_guidance(goal_driver.blocked_after)
        if head:
            head[-1] = replace(head[-1], content=f"{head[-1].content}\n\n{policy}")
        else:
            head.append(Message(role="system", content=policy))
    return [*head,
            *([] if memory_block is None else [memory_block]), *messages[prefix:], *tail]


def _request_prefix(messages: list[Message], memory_block: Message | None,
                    goal_driver: GoalRoundDriver | None = None) -> list[Message]:
    """`_context_messages` 里那段固定前缀：system head + 本 turn 的记忆块。

    摘要请求要用它做前缀，才是**主请求的真实前缀**。记忆块不进入 `messages`，
    所以拼接只发生在请求组装层，不写回会话历史。
    """
    context = _context_messages(messages, memory_block, goal_driver)
    return context[:leading_system_count(context)]


async def _run_steps(
    options: RunOptions,
    messages: list[Message],
    usage: Usage,
    turn: int,
    start_step: int,
    entry_ids: dict[int, str],
    memory_block: Message | None,
    started_at: float,
    meter: ContextMeter,
) -> _StepOutcome:
    """`started_at` 由 `run_agent` 传入：时间预算覆盖整个 run，不随 turn 重置。"""
    steps = start_step

    while True:
        if _is_cancelled(options):
            return _StepOutcome("aborted", steps)
        if (options.goal_driver is not None and options.goal_driver.attempt is not None
                and options.goal_driver.attempt.cancelled):
            return _StepOutcome("aborted", steps)

        # 预算：步数 / 时间，先到先停（与取消分开：这是配额用尽，不是用户中断）。
        reason = _budget_reason(options, steps, started_at)
        if reason is not None:
            return _StepOutcome("budget_exceeded", steps, reason)

        # 回合首步认领 next-turn，后续步骤只认领 next-step。
        target = "next-turn" if steps == start_step else "next-step"
        if not await _claim_inbox(options, messages, turn, steps + 1, target, entry_ids):
            return _StepOutcome("completed", steps)

        # 在步骤边界压缩上下文，并传播摘要请求的取消信号。
        try:
            await _maybe_compact(
                options, messages, usage, turn, steps + 1, meter, entry_ids, memory_block,
                _budget_message(options, options.run_control.steps, started_at, steps),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if _is_cancelled(options):
                return _StepOutcome("aborted", steps)
            return _StepOutcome("error", steps, _error_text(exc))

        # Compaction can await while another child consumes the shared allowance.
        # Claim this response without an await between the check and increment.
        reason = _budget_reason(options, steps, started_at)
        if reason is not None:
            return _StepOutcome("budget_exceeded", steps, reason)
        budget_message = _budget_message(options, options.run_control.steps, started_at, steps)
        options.run_control.steps += 1
        steps += 1
        step_usage = Usage()
        step_anchor: ContextUsageAnchor | None = None
        _emit(options, StepStarted(turn, steps))
        await _append(options, StepStartEntry(turn, steps, _now()))

        # 在 finally 中关闭已开始的 step；收尾异常让位于原始异常。
        failure: BaseException | None = None
        try:
            request = ModelRequest(
                messages=_context_messages(
                    messages, memory_block, options.goal_driver,
                    budget_message,
                ),
                tools=options.tools.list(), cancelled=_cancel_probe(options),
            )
            try:
                turn_result = await _request_with_retry(
                    options, messages, turn, steps, request
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # 不吞异常：进事件 + 进返回结果
                if _is_cancelled(options):
                    return _StepOutcome("aborted", steps)
                text = _error_text(exc)
                _emit(options, RunError(text))
                return _StepOutcome("error", steps, text)

            if turn_result is None:  # 流式被取消：可见前缀已提交，结束本 step
                return _StepOutcome("aborted", steps)

            step_usage.add(turn_result.usage)
            usage.add(turn_result.usage)
            options.run_control.usage.add(turn_result.usage)
            meter.observe(request, options.provider, turn_result)
            step_anchor = meter.anchor
            await _commit_message(options, messages, turn, steps, turn_result.message, entry_ids)
            _emit(options, AssistantTurnEvent(turn, steps, turn_result.message))

            if turn_result.stop_reason == "aborted":
                # 取消已规范为 aborted；可见前缀持久化后直接结束 step。
                return _StepOutcome("aborted", steps)

            if turn_result.stop_reason == "error":
                text = "model returned an error stop reason"
                _emit(options, RunError(text))
                return _StepOutcome("error", steps, text)

            if turn_result.stop_reason == "length":
                return _StepOutcome("max_tokens", steps)

            tool_calls = turn_result.message.tool_calls
            if not tool_calls:
                # 存在待处理的 next-step 输入时，在当前回合继续处理插话。
                if _has_pending_steering(options):
                    continue
                return _StepOutcome("completed", steps)

            terminal = await _execute_tool_calls(options, messages, turn, steps, tool_calls)
            if terminal is not None:
                if terminal == "completed" and _has_pending_steering(options):
                    continue
                return _StepOutcome(terminal, steps)
        except BaseException as exc:
            failure = exc
            raise
        finally:
            if failure is None:
                await _end_step(options, turn, steps, step_usage, step_anchor)
            else:
                with contextlib.suppress(BaseException):
                    await _end_step(options, turn, steps, step_usage, step_anchor)


async def _maybe_compact(
    options: RunOptions,
    messages: list[Message],
    usage: Usage,
    turn: int,
    step: int,
    meter: ContextMeter,
    entry_ids: dict[int, str],
    memory_block: Message | None = None,
    budget_message: Message | None = None,
) -> None:
    """在 step 边界检查并压缩上下文，失败时保持原上下文。"""
    policy = options.compaction
    if policy is None:
        return

    def pressure() -> int:
        return meter.measure(ModelRequest(
            messages=_context_messages(messages, memory_block, options.goal_driver, budget_message),
            tools=options.tools.list(),
        ), options.provider)

    context_tokens = pressure()
    if context_tokens < policy.threshold_tokens:
        return
    attempts = 0
    failures: list[str] = []

    def record_usage(value: Usage) -> None:
        usage.add(value)
        options.run_control.usage.add(value)

    # 压缩最多执行初次尝试和一次复查；每次替换后重新计量上下文。
    for _ in range(policy.compaction_retries + 1):
        outcome = await compact_context(
            messages, policy, context_tokens, cancelled=_cancel_probe(options),
            tools=options.tools.list(),
            request_prefix=_request_prefix(messages, memory_block, options.goal_driver),
            on_usage=record_usage,
            on_failure=failures.append,
        )
        if outcome is None:
            break
        attempts += 1
        messages[:] = outcome.messages
        context_tokens = pressure()
        _emit(options, ContextCompacted(
            turn=turn, step=step, summary=outcome.summary,
            compacted_messages=outcome.compacted_messages,
            context_tokens=outcome.context_tokens, context_tokens_after=context_tokens,
        ))
        first_kept = outcome.first_kept
        await _append(options, CompactionEntry(
            turn=turn, step=step, summary=outcome.summary,
            first_kept_entry_id=(
                entry_ids.get(id(first_kept)) if first_kept is not None else None
            ), timestamp=_now(), kept_messages=len(outcome.kept),
        ))
        if context_tokens < policy.threshold_tokens:
            return

    if attempts or failures:
        attempts += bool(failures)
        _emit(options, CompactionWarning(
            turn=turn, step=step, context_tokens=context_tokens,
            threshold_tokens=policy.threshold_tokens, attempts=attempts,
            message=(
                f"compaction failed: {failures[-1]}; continuing turn" if failures else
                f"compaction still above threshold after {attempts} attempts "
                f"({context_tokens} estimated tokens >= {policy.threshold_tokens}); continuing turn"
            ),
        ))


async def _claim_inbox(
    options: RunOptions,
    messages: list[Message],
    turn: int,
    step: int,
    target: str,
    entry_ids: dict[int, str],
) -> bool:
    """在 step 边界认领待处理输入并写入上下文。"""
    inbox = options.inbox
    if inbox is None:
        return True
    claimed = inbox.claim(target)  # type: ignore[arg-type]
    if not claimed:
        return True
    _emit(options, InboxClaimed(turn=turn, step=step, messages=list(claimed)))
    for message in claimed:
        if (options.goal_driver is not None
                and (message.source or {}).get("kind") == "goal"):
            async def commit(message: Message = message) -> None:
                await _commit_message(options, messages, turn, step, message, entry_ids)

            competing = bool(inbox.next_turn) or any(
                item is not message and (item.source or {"kind": "user"}).get("kind") == "user"
                for item in claimed
            )
            if not await options.goal_driver.admit(message, commit, competing_input=competing):
                for retained in reversed(claimed):
                    if retained is not message and retained not in messages:
                        inbox.prepend("next-step", retained)
                return False
            continue
        await _commit_message(options, messages, turn, step, message, entry_ids)
    return True


def _has_pending_steering(options: RunOptions) -> bool:
    return options.inbox is not None and bool(options.inbox.next_step)


# --- provider 请求：单一 event 流 ----------------------------------------------


async def _request_with_retry(
    options: RunOptions,
    messages: list[Message],
    turn: int,
    step: int,
    request: ModelRequest,
) -> AssistantTurn | None:
    """请求模型输出，并在当前 step 的有限预算内重试可恢复错误。

    每次实际请求前检查熔断状态；被拒绝的调用直接返回错误。
    """
    ledger = _RetryLedger(options)
    try:
        answer = await _request_attempts(options, messages, turn, step, request, ledger)
    except BaseException as exc:
        ledger.finish(exc)
        raise
    # `None` 表示取消（可见前缀已提交，见 `_request_assistant_turn`）：
    # 它不抛异常，所以不能把“没有异常”当成 recovered。
    ledger.finish(None, cancelled=answer is None)
    return answer


async def _request_attempts(
    options: RunOptions,
    messages: list[Message],
    turn: int,
    step: int,
    request: ModelRequest,
    ledger: _RetryLedger,
) -> AssistantTurn | None:
    policy = options.retry_policy
    attempt_no = 0

    while True:
        attempt_no += 1
        permit = _circuit_acquire(options)
        delay: float | None = None
        error: ProviderError | None = None
        try:
            answer = await _request_assistant_turn(options, messages, turn, step, request)
        except asyncio.CancelledError:
            _circuit_record(options, permit, Outcome.CANCELLED)
            raise
        except Exception as exc:
            error = as_provider_error(exc)
            _circuit_record(options, permit, provider_outcome(error))
            if policy is None or _is_cancelled(options):
                raise
            retries_done = attempt_no - 1
            if not policy.should_retry(error, retries_done):
                raise
            if _circuit_open(options):
                # 这次失败已经把熔断打开（或探测失败）—— 立刻结束，
                # 不要先睡完退避、再在下一轮才发现不能请求。
                raise
            delay = policy.delay_for(retries_done + 1, error)
            if delay is None:
                raise
        else:
            # `None` 表示取消（可见前缀已提交），不能当成功。
            _circuit_record(
                options, permit, Outcome.CANCELLED if answer is None else Outcome.SUCCESS,
            )
            return answer
        finally:
            # 名额先归还，再进入退避：睡眠期间名额不能一直被这条请求占着。
            _circuit_release(options, permit)
        assert error is not None and delay is not None
        ledger.retried = True
        _emit(
            options,
            ProviderRetryScheduled(
                turn=turn,
                step=step,
                attempt=attempt_no,
                delay=delay,
                code=error.code,
                message=str(error),
            ),
        )
        if not await _sleep_cancellable(delay, options.cancel_event):
            raise ProviderError("model request cancelled", ErrorCode.ABORTED) from error
        _emit(options, ProviderRetryStarted(turn=turn, step=step, attempt=attempt_no + 1))


class _RetryLedger:
    """逻辑请求的终局账本。

    只统计**进入过重试**的逻辑请求，每条记一次：`recovered` / `failed` / `cancelled`。
    分母不是物理 attempt —— 一次请求重试三次后成功，按 attempt 算只有 25%，
    按逻辑请求算是 100%。
    """

    def __init__(self, options: RunOptions) -> None:
        self._options = options
        self.retried = False
        self._closed = False

    def finish(self, error: BaseException | None = None, *, cancelled: bool = False) -> None:
        """`cancelled=True` 用于“没抛异常但实际上是取消”的路径（`answer is None`）。"""
        if self._closed or not self.retried:
            return
        self._closed = True
        # 只看依赖键，**不看熔断是否启用**：重试统计是独立口径，
        # 关掉熔断不应该让恢复率凭空消失。
        dependency = self._options.circuit_dependency
        if not dependency:
            return
        if cancelled or (
            error is not None
            and (
                isinstance(error, asyncio.CancelledError)
                or has_code(error, ErrorCode.ABORTED)
                or _is_cancelled(self._options)
            )
        ):
            outcome = "cancelled"
        elif error is None:
            outcome = "recovered"
        else:
            outcome = "failed"
        metrics.RETRY_OUTCOMES.labels(dependency, outcome).inc()
        metrics.log("retry.request_finished", dependency=dependency, outcome=outcome)


def _circuit_acquire(options: RunOptions) -> Permit | None:
    circuit, dependency = options.circuit, options.circuit_dependency
    if circuit is None or not dependency:
        return None
    try:
        return circuit.acquire(dependency)
    except CircuitOpen as open_error:
        # 明确错误码 + 不可重试：任务以清晰原因结束，不会继续退避。
        raise ProviderError(str(open_error), "CIRCUIT_OPEN", retryable=False) from open_error


def _circuit_record(options: RunOptions, permit: Permit | None, outcome: Outcome) -> None:
    if options.circuit is not None:
        options.circuit.record(permit, outcome)


def _circuit_release(options: RunOptions, permit: Permit | None) -> None:
    if options.circuit is not None:
        options.circuit.release(permit)


def _circuit_open(options: RunOptions) -> bool:
    circuit, dependency = options.circuit, options.circuit_dependency
    return circuit is not None and bool(dependency) and circuit.is_open(dependency)


async def _request_assistant_turn(
    options: RunOptions,
    messages: list[Message],
    turn: int,
    step: int,
    request: ModelRequest,
) -> AssistantTurn | None:
    """执行一次流式模型请求，同时等待数据和取消信号。

    被取消时提交可见前缀并返回 None。
    """
    iterator = options.provider.generate_stream(request).__aiter__()
    attempt = _StreamAttempt()
    try:
        while True:
            pending = asyncio.ensure_future(iterator.__anext__())
            completed, value = await _await_or_cancel(options, pending)
            if not completed:
                await attempt.commit_visible_prefix(options, messages, turn, step)
                return None
            if value is _STREAM_END:
                break
            # 提交每块增量前检查取消信号。
            if _is_cancelled(options):
                await attempt.commit_visible_prefix(options, messages, turn, step)
                return None
            attempt.apply(value, options, turn, step)
            # 终止原因是流里的 `finish` chunk（含 aborted / error），不是异常。
            if isinstance(value, StreamFinished):
                return _turn_or_raise(attempt)
    except BaseException as exc:
        # 取消时提交可见前缀，再传播原始取消异常。
        await attempt.commit_visible_prefix(options, messages, turn, step, only_if_cancelled=True)
        if attempt.text_emitted and isinstance(exc, Exception):
            # 正文不能重复输出；仅推理的失败 attempt 可由 retry-started 重置预览。
            error = as_provider_error(exc)
            error.retryable = False
            raise error from exc
        raise
    finally:
        # 所有退出路径都要关流（完成 / 取消 / 出错），而且只关一次：
        # 真 transport 下漏关会一直占着连接，还要等 GC 才发现
        # （表现为 `aclose(): asynchronous generator is already running`）。
        await _close_iterator(iterator)

    # 流耗尽但缺少收尾事件时报告 STREAM_CLOSED。
    raise ProviderError("provider stream ended without a terminal event", ErrorCode.STREAM_CLOSED)


def _turn_or_raise(attempt: _StreamAttempt) -> AssistantTurn:
    """将 error 和 aborted 终止块还原为异常，保留稳定错误码供重试策略判断。"""
    reason = attempt.finish
    if reason.kind not in ("error", "aborted"):
        return attempt.turn()

    failure = reason.failure
    raise ProviderError(
        failure.message if failure is not None else f"model stopped: {reason.kind}",
        failure.code
        if failure is not None
        else (ErrorCode.ABORTED if reason.kind == "aborted" else ErrorCode.PROVIDER),
        status_code=failure.status if failure is not None else None,
        retry_after=(
            failure.retry_after_ms / 1000
            if failure is not None and failure.retry_after_ms is not None
            else None
        ),
    )


async def _await_or_cancel(
    options: RunOptions, pending: asyncio.Future[Any]
) -> tuple[bool, Any]:
    """等一个 provider 操作，同时监听取消。

    返回 `(是否完成, 结果)`；取消时立刻取消该操作并返回 `(False, None)`。
    外层任务被取消（`task.cancel()` / `wait_for` 超时）时也必须收掉 pending，
    否则 provider 的 `__anext__()` / `generate()` 会继续挂在后台。
    """
    try:
        if options.cancel_event is None:
            try:
                return True, await pending
            except StopAsyncIteration:  # 流结束
                return True, _STREAM_END

        waiter = asyncio.ensure_future(options.cancel_event.wait())
        try:
            done, _ = await asyncio.wait({pending, waiter}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()
        if pending in done:
            try:
                return True, pending.result()
            except StopAsyncIteration:
                return True, _STREAM_END
        pending.cancel()
        with contextlib.suppress(BaseException):
            await pending
        return False, None
    except BaseException:
        # 外层取消：把 provider 的子任务也收掉，再向上传播取消。
        pending.cancel()
        with contextlib.suppress(BaseException):
            await pending
        raise


async def _close_iterator(iterator: Any) -> None:
    """放弃一个流时关掉它，释放底层连接。"""
    aclose = getattr(iterator, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except BaseException:  # noqa: BLE001 - 关闭失败不该掩盖取消结果
        return


async def _sleep_cancellable(delay: float, cancel_event: asyncio.Event | None) -> bool:
    """等待退避时间，被取消时立即返回 False。"""
    if cancel_event is None:
        await asyncio.sleep(delay)
        return True
    try:
        await asyncio.wait_for(cancel_event.wait(), timeout=delay)
    except TimeoutError:
        return True
    return False


class _StreamAttempt:
    """维护单次流式请求的组装状态并转发增量事件。

    完成时读取组装结果，取消时提取文本与推理前缀。
    """

    def __init__(self) -> None:
        self._assembler = BlockAssembler()
        self._text_emitted = False

    @property
    def text_emitted(self) -> bool:
        """是否已经输出正文（一旦有，就不再重试这次请求）。"""
        return self._text_emitted

    def apply(self, chunk: StreamChunk, options: RunOptions, turn: int, step: int) -> None:
        self._assembler.push(chunk)
        if isinstance(chunk, TextDelta):
            self._text_emitted = True
            _emit(options, AssistantTextDelta(turn, step, chunk.text))
        elif isinstance(chunk, ReasoningDelta):
            _emit(options, AssistantReasoningDelta(turn, step, chunk.text))

    def turn(self) -> AssistantTurn:
        return self._assembler.turn()

    @property
    def finish(self) -> FinishReason:
        return self._assembler.finish

    async def commit_visible_prefix(
        self,
        options: RunOptions,
        messages: list[Message],
        turn: int,
        step: int,
        *,
        only_if_cancelled: bool = False,
    ) -> None:
        """取消时持久化已输出的文本与推理前缀，并标记 interrupted。"""
        if only_if_cancelled and not _is_cancelled(options):
            return
        blocks = self._assembler.interrupted_blocks()
        text = "".join(
            block.text
            for block in blocks
            if isinstance(block, TextBlock)
        )
        reasoning = "".join(block.text for block in blocks if isinstance(block, ReasoningBlock))
        if text == "" and reasoning == "":
            return
        message = Message(
            role="assistant", content=text, reasoning_content=reasoning, interrupted=True
        )
        messages.append(message)
        await _append(options, MessageEntry(turn, step, message, _now()))
        _emit(options, AssistantTurnEvent(turn, step, message))


async def _execute_tool_calls(
    options: RunOptions,
    messages: list[Message],
    turn: int,
    step: int,
    tool_calls: list[ToolCall],
) -> AgentStatus | None:
    """按模型顺序调度工具调用并提交结果。

    连续 parallel-safe 调用进入有界并发池，exclusive 调用形成屏障。
    完成标记在本 step 的全部结果写回后结束回合。
    """
    next_index = 0
    concluded = False

    while next_index < len(tool_calls):
        mode = _execution_mode(options, tool_calls[next_index])
        group = list(tool_calls[next_index:]) if mode == "parallel" else [tool_calls[next_index]]
        consumed, aborted, group_concluded = await _run_group(
            options, messages, turn, step, group, mode
        )
        next_index += consumed
        concluded = concluded or group_concluded
        if aborted:
            await _write_skipped(options, messages, turn, step, tool_calls[next_index:])
            return "aborted"

    return "completed" if concluded else None


async def _run_group(
    options: RunOptions,
    messages: list[Message],
    turn: int,
    step: int,
    group: list[ToolCall],
    mode: str,
) -> tuple[int, bool, bool]:
    """执行一个独占调用或并行调用池，返回已消费、取消和完成状态。"""
    slots: dict[int, ToolResult] = {}
    in_flight: dict[int, asyncio.Task[None]] = {}
    next_to_start = 0
    started = 0
    committed = 0
    concluded = False
    aborted = _is_cancelled(options)

    async def commit_slot(index: int, result: ToolResult) -> None:
        """完成结果提交后再推进槽位。

        使用独立任务与 shield 保护单次写入，取消时等待至 COMMIT_GRACE_SECONDS。
        超时后继续传播取消，避免重复提交同一结果。
        """
        call = group[index]
        message = _tool_result_message(call, result)
        # 上下文一侧同步完成：不管后面写入是否被打断，这个调用都只有这一条结果。
        messages.append(message)
        pending = asyncio.create_task(_append(options, MessageEntry(turn, step, message, _now())))

        def settle() -> None:
            nonlocal committed, concluded
            committed += 1
            if result.concludes_turn:
                concluded = True
            if options.inbox is not None:
                for context in result.additional_contexts:
                    options.inbox.steer(context)
            _emit(options, ToolFinished(turn, step, call, result))

        try:
            await asyncio.shield(pending)
        except asyncio.CancelledError:
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(asyncio.shield(pending), COMMIT_GRACE_SECONDS)
            if not pending.done():
                pending.add_done_callback(_discard_outcome)
            settle()
            raise
        settle()

    async def commit_ready() -> None:
        """按模型顺序提交连续已完成的槽位。"""
        while committed in slots:
            await commit_slot(committed, slots.pop(committed))

    async def flush_dispatched() -> None:
        """中断收尾：按模型顺序提交**已派发**的全部槽位，缺失的补一条中断结果。

        `commit_ready` 要求连续：靠前的调用被取消后永远不会有 slot，后面已经跑完的
        结果就被堵住丢掉（外层 `task.cancel()` 的典型交错）。
        这里不重跑任何工具 —— 已有结果原样落盘，没有结果的补 synthetic，
        从而保证「每次已派发的调用都有配对的 result」，历史可以直接续接。

        已经推进 `committed` 的槽位已完成提交，收尾不再碰它，避免重复结果。
        """
        while committed < started:
            index = committed
            result = slots.pop(index, None)
            if result is None:
                result = ToolResult(content=ABORTED_IN_FLIGHT, is_error=True)
            await commit_slot(index, result)

    async def run_one(index: int) -> None:
        call = group[index]
        _emit(options, ToolStarted(turn, step, call))
        deferred: list[Message] = []
        initiator = TOOL_INITIATOR.set(options.goal_driver)
        try:
            from .metrics import tool_call

            with tool_call(call.name) as measurement, observe(f"tool.{call.name}", "TOOL", **{
                "tool.name": call.name, "tool_call.id": call.id,
            }) as span:
                result = await options.tools.execute(
                    call, ToolContext(cancelled=_cancel_probe(options), session_id=None,
                                      agent=options.goal_driver, defer_context=deferred.append,
                                      cancel_event=options.cancel_event,
                                      run_control=options.run_control, tool_call_id=call.id),
                )
                span.set_attribute("tool.is_error", result.is_error)
                measurement.outcome = "error" if result.is_error else "ok"
                if result.is_error:
                    record_failure(span, "tool_error")
            result.additional_contexts.extend(deferred)
        finally:
            TOOL_INITIATOR.reset(initiator)
        slots[index] = result

    async def fill() -> None:
        nonlocal next_to_start, started, aborted
        while (
            not aborted
            and next_to_start < len(group)
            and len(in_flight) < options.max_parallel_tool_calls
        ):
            # 派发前重新读取并发模式，遇到独占调用则结束当前并行池。
            if (
                next_to_start > 0
                and mode == "parallel"
                and _execution_mode(options, group[next_to_start]) != "parallel"
            ):
                break
            index = next_to_start
            in_flight[index] = asyncio.create_task(run_one(index))
            started += 1
            next_to_start += 1
            if _is_cancelled(options):
                aborted = True

    try:
        await fill()
        while in_flight:
            await asyncio.wait(in_flight.values(), return_when=asyncio.FIRST_COMPLETED)
            for index in [key for key, task in in_flight.items() if task.done()]:
                del in_flight[index]
            await commit_ready()
            if _is_cancelled(options):
                aborted = True
            if aborted:
                break
            await fill()
    except BaseException:
        # 父运行取消时回收已启动的工具任务，并提交已经完成的结果。
        await _abandon_group(in_flight)
        with contextlib.suppress(asyncio.CancelledError):
            await flush_dispatched()
            await _write_skipped(options, messages, turn, step, group[started:])
        raise

    if aborted:
        # 收齐并提交已启动的调用，为未派发的调用补写跳过结果。
        if in_flight:
            await asyncio.wait(in_flight.values())
        await flush_dispatched()
        await _write_skipped(options, messages, turn, step, group[started:])
        return len(group), True, concluded

    return started, False, concluded


def _discard_outcome(task: asyncio.Task[Any]) -> None:
    """超时后被放弃的收尾任务：取走它的结果/异常，避免 asyncio 报「未取回的错误」。"""
    if not task.cancelled():
        task.exception()


async def _abandon_group(in_flight: dict[int, asyncio.Task[None]]) -> None:
    """取消并等待本组仍在跑的工具任务（不负责提交）。"""
    if not in_flight:
        return
    for task in in_flight.values():
        task.cancel()
    for index in list(in_flight):
        with contextlib.suppress(BaseException):
            await in_flight[index]
    in_flight.clear()


async def _write_skipped(
    options: RunOptions,
    messages: list[Message],
    turn: int,
    step: int,
    calls: list[ToolCall],
) -> None:
    """为尚未派发的调用写入合成的跳过结果。"""
    for call in calls:
        await _record_tool_result(
            options,
            messages,
            turn,
            step,
            call,
            ToolResult(content=ABORTED_BEFORE_DISPATCH, is_error=True),
        )


def _execution_mode(options: RunOptions, call: ToolCall) -> str:
    lookup = getattr(options.tools, "execution_mode", None)
    if lookup is None:
        return "exclusive"
    return lookup(call.name)


def _tool_result_message(call: ToolCall, result: ToolResult) -> Message:
    return Message(role="tool", content=result.content, tool_call_id=call.id, name=call.name)


async def _record_tool_result(
    options: RunOptions,
    messages: list[Message],
    turn: int,
    step: int,
    call: ToolCall,
    result: ToolResult,
) -> None:
    message = _tool_result_message(call, result)
    messages.append(message)
    await _append(options, MessageEntry(turn, step, message, _now()))
    _emit(options, ToolFinished(turn, step, call, result))


# --- 小工具 -------------------------------------------------------------------


def _emit(options: RunOptions, event: AgentEvent) -> None:
    if options.on_event is not None:
        options.on_event(event)


async def _append(options: RunOptions, event: SessionEvent) -> str | None:
    result = None if options.session is None else await options.session.append(event)
    if options.goal_driver is not None:
        options.goal_driver.observe(event)
    return result


async def _commit_message(
    options: RunOptions,
    messages: list[Message],
    turn: int,
    step: int,
    message: Message,
    entry_ids: dict[int, str],
) -> None:
    """把一条消息写进上下文与会话，并记住它的节点 id（压缩时用来记裁剪边界）。"""
    messages.append(message)
    node_id = await _append(options, MessageEntry(turn, step, message, _now()))
    if node_id is not None:
        entry_ids[id(message)] = node_id


async def _end_step(
    options: RunOptions, turn: int, step: int, usage: Usage | None = None,
    context_anchor: ContextUsageAnchor | None = None,
) -> None:
    _emit(options, StepFinished(turn, step))
    await _append(options, StepEndEntry(turn, step, _now(), usage, context_anchor))


def _is_cancelled(options: RunOptions) -> bool:
    return options.cancel_event is not None and options.cancel_event.is_set()


def _cancel_probe(options: RunOptions) -> Callable[[], bool]:
    event = options.cancel_event
    if event is None:
        return lambda: False
    return event.is_set


def _error_text(error: BaseException) -> str:
    return str(error) or error.__class__.__name__


def _now() -> str:
    return datetime.now(UTC).isoformat()


def system_message(content: str) -> Message:
    return Message(role="system", content=content)


def user_message(content: str) -> Message:
    return Message(role="user", content=content)


def assistant_message(content: str) -> Message:
    return Message(role="assistant", content=content)
