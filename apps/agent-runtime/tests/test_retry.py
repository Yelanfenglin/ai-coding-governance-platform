"""验证有限重试、Retry-After 上限、取消、计数重置和可见输出后的重放控制。"""

from __future__ import annotations

import asyncio

from app.contracts import (
    AssistantTextDelta,
    AssistantTurn,
    Message,
    ModelRequest,
    ProviderRetryScheduled,
    ProviderRetryStarted,
    TextDelta,
    ToolCall,
    ToolResult,
)
from app.errors import ErrorCode, ProviderError
from app.loop import RunOptions, run_agent, user_message
from app.providers.mock import stream_from_coroutine
from app.retry import RetryPolicy
from app.tools import FunctionTool, ToolRegistry


def final_turn(text: str) -> AssistantTurn:
    return AssistantTurn(message=Message(role="assistant", content=text), stop_reason="stop")


def rate_limited(*, retry_after: float | None = None) -> ProviderError:
    return ProviderError(
        "slow down", ErrorCode.RATE_LIMIT, status_code=429, retry_after=retry_after
    )


def fast_policy(**kwargs: object) -> RetryPolicy:
    """测试用策略：退避极短且无 jitter，保证确定性。"""
    defaults: dict[str, object] = {
        "max_retries": 5,
        "base_delay": 0.001,
        "max_delay": 0.01,
        "jitter": False,
    }
    defaults.update(kwargs)
    return RetryPolicy(**defaults)  # type: ignore[arg-type]


class FlakyProvider:
    """前几次请求抛指定异常，之后返回最终 turn。"""

    def __init__(self, failures: list[BaseException], final: AssistantTurn) -> None:
        self.failures = list(failures)
        self.final = final
        self.requests: list[ModelRequest] = []

    def generate_stream(self, request):
        return stream_from_coroutine(lambda: self.generate(request))

    async def generate(self, request: ModelRequest) -> AssistantTurn:
        self.requests.append(request)
        if self.failures:
            raise self.failures.pop(0)
        return self.final


async def test_retryable_error_is_retried_and_then_succeeds() -> None:
    provider = FlakyProvider([rate_limited()], final_turn("done"))
    events: list[object] = []

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry(),
            messages=[user_message("go")],
            retry_policy=fast_policy(),
            on_event=events.append,
        )
    )

    assert result.status == "completed"
    assert len(provider.requests) == 2
    assert result.messages[-1].content == "done"

    scheduled = [event for event in events if isinstance(event, ProviderRetryScheduled)]
    started = [event for event in events if isinstance(event, ProviderRetryStarted)]
    assert len(scheduled) == 1
    assert scheduled[0].attempt == 1
    assert scheduled[0].code == ErrorCode.RATE_LIMIT
    assert scheduled[0].delay > 0
    assert [(event.attempt) for event in started] == [2]


async def test_non_retryable_error_is_not_retried() -> None:
    provider = FlakyProvider(
        [ProviderError("unauthorized", ErrorCode.AUTH, status_code=401)], final_turn("never")
    )

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry(),
            messages=[user_message("go")],
            retry_policy=fast_policy(),
        )
    )

    assert result.status == "error"
    assert len(provider.requests) == 1


async def test_retries_are_bounded_by_max_retries() -> None:
    provider = FlakyProvider([rate_limited(), rate_limited()], final_turn("never"))

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry(),
            messages=[user_message("go")],
            retry_policy=fast_policy(max_retries=1),
        )
    )

    assert result.status == "error"
    assert len(provider.requests) == 2, "首次请求 + 1 次重试"


async def test_unclassified_exception_is_classified_then_retried() -> None:
    """异常文本中的 429 归类为 RATE_LIMIT 并重试。"""
    provider = FlakyProvider(
        [RuntimeError("429 rate limit reached")], final_turn("done")
    )

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry(),
            messages=[user_message("go")],
            retry_policy=fast_policy(),
        )
    )

    assert result.status == "completed"
    assert len(provider.requests) == 2


async def test_provider_retry_after_wins_over_backoff() -> None:
    provider = FlakyProvider([rate_limited(retry_after=0.002)], final_turn("done"))
    events: list[object] = []

    await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry(),
            messages=[user_message("go")],
            retry_policy=fast_policy(base_delay=0.5),
            on_event=events.append,
        )
    )

    scheduled = [event for event in events if isinstance(event, ProviderRetryScheduled)]
    assert len(scheduled) == 1
    assert scheduled[0].delay == 0.002


async def test_provider_retry_beyond_max_delay_is_not_retried() -> None:
    """服务端等待时间超过配置上限时结束重试。"""
    provider = FlakyProvider([rate_limited(retry_after=99.0)], final_turn("never"))

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry(),
            messages=[user_message("go")],
            retry_policy=fast_policy(base_delay=0.001, max_delay=0.01),
        )
    )

    assert result.status == "error"
    assert len(provider.requests) == 1


async def test_cancellation_during_failure_stops_retrying() -> None:
    cancel_event = asyncio.Event()

    class CancelOnFailure:
        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        def generate_stream(self, request):
            return stream_from_coroutine(lambda: self.generate(request))

        async def generate(self, request: ModelRequest) -> AssistantTurn:
            self.requests.append(request)
            cancel_event.set()
            raise rate_limited()

    provider = CancelOnFailure()

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry(),
            messages=[user_message("go")],
            cancel_event=cancel_event,
            retry_policy=fast_policy(),
        )
    )

    assert result.status == "aborted"
    assert len(provider.requests) == 1


async def test_cancellation_during_the_backoff_stops_retrying() -> None:
    cancel_event = asyncio.Event()

    class CancelWhileWaiting:
        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        def generate_stream(self, request):
            return stream_from_coroutine(lambda: self.generate(request))

        async def generate(self, request: ModelRequest) -> AssistantTurn:
            self.requests.append(request)
            if len(self.requests) == 1:
                # 退避期间取消：等待应该在取消时立即结束
                async def cancel_soon() -> None:
                    await asyncio.sleep(0)
                    cancel_event.set()

                asyncio.create_task(cancel_soon())
                raise rate_limited()
            return final_turn("done")

    provider = CancelWhileWaiting()

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry(),
            messages=[user_message("go")],
            cancel_event=cancel_event,
            retry_policy=fast_policy(base_delay=0.05, max_delay=0.5, jitter=False),
        )
    )

    assert result.status == "aborted"
    assert len(provider.requests) == 1


async def test_stream_with_visible_output_is_not_retried() -> None:
    """已有可见输出的流式请求不重试：重试会让调用方看到两份内容。"""

    class StreamThenFail:
        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        def generate_stream(self, request: ModelRequest):  # noqa: ANN201
            self.requests.append(request)
            return self._stream()

        async def _stream(self):  # noqa: ANN201
            yield TextDelta(index=0, text="partial")
            raise ProviderError("stream died", ErrorCode.SERVER)

    provider = StreamThenFail()
    events: list[object] = []

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry(),
            messages=[user_message("go")],
            retry_policy=fast_policy(),
            on_event=events.append,
        )
    )

    assert result.status == "error"
    assert len(provider.requests) == 1
    assert any(isinstance(event, AssistantTextDelta) for event in events)
    assert not any(isinstance(event, ProviderRetryScheduled) for event in events)


async def test_tools_are_not_executed_twice_across_retries() -> None:
    executions: list[str] = []

    def echo(args: dict[str, object], context: object) -> ToolResult:
        executions.append(str(args.get("text", "")))
        return ToolResult(content=str(args.get("text", "")))

    registry = ToolRegistry().register(FunctionTool(name="echo", handler=echo))

    class ToolThenFlaky:
        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        def generate_stream(self, request):
            return stream_from_coroutine(lambda: self.generate(request))

        async def generate(self, request: ModelRequest) -> AssistantTurn:
            self.requests.append(request)
            if len(self.requests) == 1:
                return AssistantTurn(
                    message=Message(
                        role="assistant",
                        content="",
                        tool_calls=[
                            ToolCall(id="call-1", name="echo", arguments={"text": "once"})
                        ],
                    ),
                    stop_reason="tool_calls",
                )
            if len(self.requests) == 2:
                raise rate_limited()
            return final_turn("done")

    provider = ToolThenFlaky()

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=registry,
            messages=[user_message("go")],
            retry_policy=fast_policy(),
        )
    )

    assert result.status == "completed"
    assert executions == ["once"], "工具不能被重试重复执行"
    assert [m.content for m in result.messages if m.role == "tool"] == ["once"]
    assert len(provider.requests) == 3


def test_backoff_is_exponential_and_capped() -> None:
    policy = RetryPolicy(base_delay=0.5, max_delay=8.0, jitter=False)

    assert policy.local_delay(1, 0.5) == 0.5
    assert policy.local_delay(2, 0.5) == 1.0
    assert policy.local_delay(3, 0.5) == 2.0
    assert policy.local_delay(10, 0.5) == 8.0, "退避必须夹到 max_delay"


def test_jitter_is_symmetric_and_capped() -> None:
    policy = RetryPolicy(base_delay=1.0, max_delay=8.0, jitter=True, jitter_ratio=0.1)

    assert policy.local_delay(1, 0.0) == 0.9
    assert policy.local_delay(1, 1.0) == 1.1
    assert policy.local_delay(1, 0.5) == 1.0
    assert policy.local_delay(20, 1.0) == 8.0


def test_jitter_can_be_disabled() -> None:
    policy = RetryPolicy(base_delay=1.0, jitter=False)

    assert policy.local_delay(1, 0.0) == 1.0
    assert policy.local_delay(1, 1.0) == 1.0


def test_default_retryable_codes_include_incomplete_streams() -> None:
    policy = RetryPolicy()

    assert policy.retryable_codes == frozenset(
        {"EMPTY_RESPONSE", "RATE_LIMIT", "SERVER", "STREAM_CLOSED", "TIMEOUT", "TRANSPORT"}
    )


def test_should_retry_respects_retry_budget_code_and_override() -> None:
    policy = RetryPolicy(max_retries=2)

    assert policy.should_retry(rate_limited(), 0) is True
    assert policy.should_retry(rate_limited(), 1) is True
    assert policy.should_retry(rate_limited(), 2) is False, "重试次数已用尽"
    assert policy.should_retry(ProviderError("no", ErrorCode.AUTH), 0) is False
    assert policy.should_retry(ProviderError("no", ErrorCode.SERVER, retryable=False), 0) is False


def test_defaults_match_dsh_retry_policy() -> None:
    """默认策略：重试 5 次、初始 0.5 秒、上限 10 秒、抖动比例 0.1。"""
    policy = RetryPolicy()

    assert policy.max_retries == 5
    assert policy.base_delay == 0.5
    assert policy.max_delay == 10.0
    assert policy.jitter_ratio == 0.1


async def test_default_policy_allows_six_attempts() -> None:
    """默认策略下：首次 + 5 次重试 = 6 次请求（退避压到极短以便测试）。"""
    provider = FlakyProvider([rate_limited() for _ in range(6)], final_turn("never"))

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry(),
            messages=[user_message("go")],
            retry_policy=RetryPolicy(base_delay=0.001, max_delay=0.01, jitter=False),
        )
    )

    assert result.status == "error"
    assert len(provider.requests) == 6
