"""熔断：三态状态机、两个边界的记账、半开收尾与逻辑请求终局。

覆盖依赖隔离、半开许可、重试终局与 Web 跨任务恢复。
状态机用可控时钟；HTTP 相关保留真实传输（RAG 500 用的是真回环服务器）。
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from prometheus_client import REGISTRY

from app import circuit
from app.circuit import (
    CircuitBreakers,
    CircuitOpen,
    CircuitPolicy,
    Outcome,
    State,
)
from app.contracts import (
    AssistantTurn,
    Message,
    ModelRequest,
    ProviderRetryScheduled,
    ProviderRetryStarted,
)
from app.errors import ErrorCode, ProviderError
from app.loop import RunOptions, run_agent, user_message
from app.providers.mock import stream_from_coroutine
from app.rag_tools import RagClient, RagClientConfig, RagError
from app.retry import RetryPolicy
from app.tools import ToolRegistry

MODEL_KEY = "model:http://loopback/v1|test-model"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def breaker(failures: int = 3, cooldown: float = 30.0, clock: FakeClock | None = None):
    return circuit.Breaker(
        MODEL_KEY, failures=failures, cooldown=cooldown, clock=clock or FakeClock()
    )


def sample(name: str, labels: dict | None = None) -> float:
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


def service_error() -> ProviderError:
    return ProviderError("upstream 503", ErrorCode.SERVER, status_code=503)


def final_turn(text: str) -> AssistantTurn:
    return AssistantTurn(message=Message(role="assistant", content=text), stop_reason="stop")


def fast_policy(**overrides: object) -> RetryPolicy:
    defaults: dict[str, object] = {
        "max_retries": 5,
        "base_delay": 0.001,
        "max_delay": 0.01,
        "jitter": False,
    }
    defaults.update(overrides)
    return RetryPolicy(**defaults)  # type: ignore[arg-type]


class ScriptedProvider:
    """按脚本抛异常（或返回最终 turn），记录发出去的物理请求数。"""

    def __init__(self, failures: list[BaseException], final: AssistantTurn | None = None) -> None:
        self.failures = list(failures)
        self.final = final or final_turn("done")
        self.requests: list[ModelRequest] = []

    def generate_stream(self, request: ModelRequest):
        return stream_from_coroutine(lambda: self.generate(request))

    async def generate(self, request: ModelRequest) -> AssistantTurn:
        self.requests.append(request)
        if self.failures:
            raise self.failures.pop(0)
        return self.final


class HangingProvider:
    """第一次失败，之后每次请求都挂住（等取消信号把它收掉，不抛异常）。"""

    def __init__(self, failure: BaseException) -> None:
        self.failure = failure
        self.requests: list[ModelRequest] = []

    def generate_stream(self, request: ModelRequest):
        self.requests.append(request)
        if len(self.requests) == 1:
            async def boom():
                raise self.failure
                yield  # pragma: no cover - 只是为了成为 async generator

            return boom()

        async def hang():
            await asyncio.sleep(5)
            yield  # pragma: no cover

        return hang()


def options(
    provider: ScriptedProvider,
    *,
    breakers: CircuitBreakers,
    policy: RetryPolicy | None = None,
    key: str = MODEL_KEY,
    on_event=None,
    cancel_event: asyncio.Event | None = None,
) -> RunOptions:
    return RunOptions(
        provider=provider,
        tools=ToolRegistry(),
        messages=[user_message("go")],
        retry_policy=policy or fast_policy(),
        circuit=breakers,
        circuit_dependency=key,
        on_event=on_event,
        cancel_event=cancel_event,
    )


# --- 状态机 ----------------------------------------------------------------


def test_closed_opens_after_threshold_and_then_rejects() -> None:
    resolved = breaker(failures=3)
    for _ in range(3):
        resolved.record(resolved.acquire(), Outcome.FAILURE)
    assert resolved.state is State.OPEN
    assert resolved.consecutive_failures == 3
    with pytest.raises(CircuitOpen):
        resolved.acquire()


def test_ignored_and_cancelled_neither_count_nor_reset() -> None:
    """429/401/取消既不算失败、也不能把连续失败清零（否则熔断永远开不了）。"""
    resolved = breaker(failures=3)
    resolved.record(resolved.acquire(), Outcome.FAILURE)
    resolved.record(resolved.acquire(), Outcome.IGNORED)
    resolved.record(resolved.acquire(), Outcome.CANCELLED)
    resolved.record(resolved.acquire(), Outcome.FAILURE)
    assert (resolved.state, resolved.consecutive_failures) == (State.CLOSED, 2)
    resolved.record(resolved.acquire(), Outcome.FAILURE)
    assert resolved.state is State.OPEN


def test_success_clears_the_consecutive_count() -> None:
    resolved = breaker(failures=3)
    resolved.record(resolved.acquire(), Outcome.FAILURE)
    resolved.record(resolved.acquire(), Outcome.SUCCESS)
    assert resolved.consecutive_failures == 0


def test_cooldown_grants_exactly_one_probe() -> None:
    clock = FakeClock()
    resolved = breaker(failures=2, clock=clock)
    for _ in range(2):
        resolved.record(resolved.acquire(), Outcome.FAILURE)
    with pytest.raises(CircuitOpen):
        resolved.acquire()

    clock.advance(30)
    probe = resolved.acquire()
    assert probe.probe is True
    with pytest.raises(CircuitOpen):
        resolved.acquire()  # 冷却到期只放一个探测，不是放开全部

    resolved.record(probe, Outcome.SUCCESS)
    resolved.release(probe)
    assert resolved.state is State.CLOSED


def test_probe_failure_reopens_and_restarts_the_cooldown() -> None:
    clock = FakeClock()
    resolved = breaker(failures=2, clock=clock)
    for _ in range(2):
        resolved.record(resolved.acquire(), Outcome.FAILURE)
    clock.advance(30)
    probe = resolved.acquire()
    resolved.record(probe, Outcome.FAILURE)
    resolved.release(probe)
    assert resolved.state is State.OPEN
    with pytest.raises(CircuitOpen):
        resolved.acquire()
    clock.advance(30)
    assert resolved.acquire().probe is True


def test_probe_with_a_non_counting_error_keeps_half_open_and_frees_the_slot() -> None:
    """探测遇到 422/取消这类不计数结果时，不能把探测机会锁到下一次冷却。"""
    clock = FakeClock()
    resolved = breaker(failures=1, clock=clock)
    resolved.record(resolved.acquire(), Outcome.FAILURE)
    clock.advance(30)
    first = resolved.acquire()
    resolved.record(first, Outcome.IGNORED)
    resolved.release(first)
    assert resolved.state is State.HALF_OPEN
    second = resolved.acquire()
    assert second.probe is True and second.generation == first.generation


def test_cancelled_probe_releases_the_slot_without_changing_state() -> None:
    clock = FakeClock()
    resolved = breaker(failures=1, clock=clock)
    resolved.record(resolved.acquire(), Outcome.FAILURE)
    clock.advance(30)
    probe = resolved.acquire()
    resolved.record(probe, Outcome.CANCELLED)
    resolved.release(probe)
    assert resolved.state is State.HALF_OPEN
    assert resolved.acquire().probe is True


def test_stale_release_does_not_free_the_new_probe_slot() -> None:
    """同一轮半开里连续发放两个探测许可时，旧 permit 的释放不能吞掉新名额。"""
    clock = FakeClock()
    resolved = breaker(failures=1, clock=clock)
    resolved.record(resolved.acquire(), Outcome.FAILURE)
    clock.advance(30)
    first = resolved.acquire()
    resolved.record(first, Outcome.IGNORED)
    resolved.release(first)
    second = resolved.acquire()
    resolved.release(first)  # 旧 permit 的重复释放：必须无副作用
    with pytest.raises(CircuitOpen):
        resolved.acquire()
    resolved.release(second)


def test_late_success_from_an_old_request_does_not_close_a_fresh_circuit() -> None:
    """旧请求的晚到成功不能关掉刚打开的熔断（generation 已推进）。"""
    clock = FakeClock()
    resolved = breaker(failures=2, clock=clock)
    early = resolved.acquire()  # generation 0
    resolved.record(resolved.acquire(), Outcome.FAILURE)
    resolved.record(resolved.acquire(), Outcome.FAILURE)
    assert resolved.state is State.OPEN
    resolved.record(early, Outcome.SUCCESS)
    assert resolved.state is State.OPEN


def test_late_failure_from_an_old_probe_does_not_reopen_a_closed_circuit() -> None:
    clock = FakeClock()
    resolved = breaker(failures=1, clock=clock)
    resolved.record(resolved.acquire(), Outcome.FAILURE)
    clock.advance(30)
    probe = resolved.acquire()
    resolved.record(probe, Outcome.SUCCESS)
    resolved.release(probe)
    assert resolved.state is State.CLOSED
    resolved.record(probe, Outcome.FAILURE)  # 同一个探测的重复上报
    assert resolved.state is State.CLOSED


def test_dependencies_do_not_share_state() -> None:
    clock = FakeClock()
    breakers = CircuitBreakers(CircuitPolicy(model_failures=2, rag_failures=2), clock=clock)
    model = breakers.breaker(MODEL_KEY)
    rag = breakers.breaker(circuit.rag_dependency("search", "http://rag"))
    for _ in range(2):
        model.record(model.acquire(), Outcome.FAILURE)
    assert breakers.is_open(MODEL_KEY) is True
    assert rag.state is State.CLOSED


def test_rag_status_and_transport_classification() -> None:
    assert circuit.rag_outcome(200) is Outcome.SUCCESS
    assert circuit.rag_outcome(500) is Outcome.FAILURE
    assert circuit.rag_outcome(503) is Outcome.FAILURE
    assert circuit.rag_outcome(404) is Outcome.IGNORED
    assert circuit.rag_outcome(429) is Outcome.IGNORED
    assert circuit.transport_failure(httpx.ConnectError("refused")) is True
    assert circuit.transport_failure(httpx.ReadTimeout("slow")) is True
    assert circuit.transport_failure(TimeoutError()) is True
    assert circuit.transport_failure(ValueError("local bug")) is False
    assert circuit.transport_failure(RagError("httpx is required")) is False


def test_unclassified_provider_errors_are_not_counted() -> None:
    """`classify_exception` 的兜底码是 PROVIDER：本地程序错误不能喂给熔断。"""
    assert circuit.provider_outcome(service_error()) is Outcome.FAILURE
    assert circuit.provider_outcome(ProviderError("x", ErrorCode.PROVIDER)) is Outcome.IGNORED
    assert circuit.provider_outcome(ProviderError("x", ErrorCode.EMPTY_RESPONSE)) is Outcome.IGNORED
    assert circuit.provider_outcome(ProviderError("x", ErrorCode.RATE_LIMIT)) is Outcome.IGNORED
    assert circuit.provider_outcome(ProviderError("x", ErrorCode.AUTH)) is Outcome.IGNORED
    assert circuit.provider_outcome(ProviderError("x", ErrorCode.ABORTED)) is Outcome.CANCELLED
    # 适配器只带状态码、code 是动态串时也要算服务侧故障
    assert circuit.provider_outcome(
        ProviderError("x", "HTTP_503", status_code=503)
    ) is Outcome.FAILURE
    assert circuit.provider_outcome(
        ProviderError("x", "HTTP_409", status_code=409)
    ) is Outcome.IGNORED


def test_policy_reads_env_and_the_disable_switch(monkeypatch) -> None:
    policy = circuit.CircuitPolicy.from_env({
        "CIRCUIT_BREAKER_FAILURES": "4",
        "CIRCUIT_BREAKER_RAG_FAILURES": "2",
        "CIRCUIT_BREAKER_COOLDOWN_SECONDS": "5",
    })
    assert (policy.model_failures, policy.rag_failures, policy.cooldown) == (4, 2, 5.0)
    assert policy.threshold_for(MODEL_KEY) == 4
    assert policy.threshold_for("rag:search:http://rag") == 2

    assert circuit.from_env({}) is not None
    assert circuit.from_env({"CIRCUIT_BREAKER_DISABLED": "true"}) is None
    assert circuit.from_env({"CIRCUIT_BREAKER_DISABLED": "0"}) is not None


# --- 模型边界（loop 集成）--------------------------------------------------


async def test_second_task_fails_fast_without_any_physical_request() -> None:
    breakers = CircuitBreakers(CircuitPolicy(model_failures=3), clock=FakeClock())
    first = ScriptedProvider([service_error() for _ in range(3)])
    first_result = await run_agent(options(first, breakers=breakers))
    assert first_result.status == "error"
    assert len(first.requests) == 3
    assert breakers.is_open(MODEL_KEY) is True

    second = ScriptedProvider([service_error()])  # 新 provider、同一依赖键
    second_result = await run_agent(options(second, breakers=breakers))
    assert second_result.status == "error"
    assert "circuit open" in (second_result.error or "")
    assert second.requests == []


async def test_existing_failures_cut_a_new_request_before_its_own_budget() -> None:
    """已有 5 次失败时，新请求第一次失败就打开熔断——不给它六次机会。"""
    clock = FakeClock()
    breakers = CircuitBreakers(CircuitPolicy(model_failures=6), clock=clock)
    resolved = breakers.breaker(MODEL_KEY)
    for _ in range(5):
        resolved.record(resolved.acquire(), Outcome.FAILURE)

    provider = ScriptedProvider([service_error() for _ in range(6)])
    result = await run_agent(options(provider, breakers=breakers))
    assert result.status == "error"
    assert len(provider.requests) == 1  # 重试预算还有 5 次，但熔断已经打开


async def test_open_circuit_skips_the_remaining_backoff() -> None:
    """打开之后不该先睡完退避再发现不能请求。"""
    breakers = CircuitBreakers(CircuitPolicy(model_failures=2), clock=FakeClock())
    provider = ScriptedProvider([service_error() for _ in range(6)])
    events: list = []
    started = time.perf_counter()
    result = await run_agent(options(provider, breakers=breakers, on_event=events.append))
    assert result.status == "error"
    retries = [event for event in events if isinstance(event, ProviderRetryScheduled)]
    assert len(provider.requests) == 2
    assert len(retries) == 1  # 第二次失败后直接结束，没有排第三次重试
    assert time.perf_counter() - started < 1.0


@pytest.mark.parametrize("failures_before_success", [1, 2, 3, 4, 5])
async def test_success_inside_the_retry_budget_is_never_truncated(
    failures_before_success: int,
) -> None:
    """全新、独占的熔断状态下，k < min(N, 最大请求次数) 必须恢复。"""
    breakers = CircuitBreakers(CircuitPolicy(model_failures=6), clock=FakeClock())
    provider = ScriptedProvider(
        [service_error() for _ in range(failures_before_success)], final_turn("recovered")
    )
    result = await run_agent(options(provider, breakers=breakers))
    assert result.status == "completed"
    assert len(provider.requests) == failures_before_success + 1


async def test_low_threshold_truncates_a_request_that_would_have_recovered() -> None:
    """阈值低于失败深度时，本可恢复的请求会被截断——这是 N<6 的真实代价。"""
    breakers = CircuitBreakers(CircuitPolicy(model_failures=3), clock=FakeClock())
    provider = ScriptedProvider([service_error() for _ in range(3)], final_turn("would recover"))
    result = await run_agent(options(provider, breakers=breakers))
    assert result.status == "error"
    assert len(provider.requests) == 3  # 第 4 次本会成功，但被拦掉了
    assert breakers.is_open(MODEL_KEY) is True


async def test_retry_outcome_is_counted_once_per_logical_request() -> None:
    key = "model:outcome-recovered|m"
    breakers = CircuitBreakers(CircuitPolicy(model_failures=6), clock=FakeClock())
    provider = ScriptedProvider([service_error()], final_turn("done"))
    before = sample("agent_retry_outcomes_total", {"dependency": key, "outcome": "recovered"})
    await run_agent(options(provider, breakers=breakers, key=key))
    after = sample("agent_retry_outcomes_total", {"dependency": key, "outcome": "recovered"})
    assert after - before == 1


async def test_retry_outcome_failed_covers_a_non_exhausted_terminal() -> None:
    """终局不只“次数耗尽”：熔断拒绝/401/流中断都记 failed。"""
    key = "model:outcome-failed|m"
    breakers = CircuitBreakers(CircuitPolicy(model_failures=2), clock=FakeClock())
    provider = ScriptedProvider([service_error() for _ in range(2)])
    before = sample("agent_retry_outcomes_total", {"dependency": key, "outcome": "failed"})
    result = await run_agent(options(provider, breakers=breakers, key=key))
    assert result.status == "error"
    after = sample("agent_retry_outcomes_total", {"dependency": key, "outcome": "failed"})
    assert after - before == 1


async def test_retry_outcome_cancelled_while_waiting_to_back_off() -> None:
    key = "model:outcome-cancelled|m"
    breakers = CircuitBreakers(CircuitPolicy(model_failures=6), clock=FakeClock())
    cancelled = asyncio.Event()
    provider = ScriptedProvider([service_error()], final_turn("never"))
    events: list = []
    before = sample("agent_retry_outcomes_total", {"dependency": key, "outcome": "cancelled"})

    def receive(event: object) -> None:
        events.append(event)
        if isinstance(event, ProviderRetryScheduled):
            cancelled.set()

    result = await run_agent(options(
        provider, breakers=breakers, key=key, on_event=receive, cancel_event=cancelled,
    ))
    assert result.status in {"error", "aborted"}
    after = sample("agent_retry_outcomes_total", {"dependency": key, "outcome": "cancelled"})
    assert after - before == 1
    assert len(provider.requests) == 1


async def test_cancelled_in_flight_request_does_not_trade_the_circuit() -> None:
    key = "model:outcome-inflight|m"
    breakers = CircuitBreakers(CircuitPolicy(model_failures=1), clock=FakeClock())
    provider = ScriptedProvider([asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        await run_agent(options(provider, breakers=breakers, key=key))
    assert breakers.is_open(key) is False


async def test_cancel_inside_the_second_request_counts_as_cancelled_not_recovered() -> None:
    """回归：取消时 `_request_assistant_turn` 返回 `None` 而**不抛异常**，
    不能被账本当成“无异常 = 恢复”。"""
    key = "model:outcome-cancel-none|m"
    breakers = CircuitBreakers(CircuitPolicy(model_failures=6), clock=FakeClock())
    cancelled = asyncio.Event()
    provider = HangingProvider(service_error())
    before_recovered = sample("agent_retry_outcomes_total",
                              {"dependency": key, "outcome": "recovered"})
    before_cancelled = sample("agent_retry_outcomes_total",
                              {"dependency": key, "outcome": "cancelled"})

    def receive(event: object) -> None:
        if isinstance(event, ProviderRetryStarted):
            cancelled.set()  # 第二次请求已经在飞，取消它

    result = await run_agent(options(
        provider, breakers=breakers, key=key, on_event=receive, cancel_event=cancelled,
    ))
    assert result.status == "aborted"
    assert sample("agent_retry_outcomes_total",
                  {"dependency": key, "outcome": "recovered"}) - before_recovered == 0
    assert sample("agent_retry_outcomes_total",
                  {"dependency": key, "outcome": "cancelled"}) - before_cancelled == 1


async def test_retry_outcome_is_recorded_when_the_circuit_is_disabled() -> None:
    """回归：终局统计不该绑在熔断开关上——关掉熔断也要有恢复率。"""
    key = "model:outcome-no-breaker|m"
    provider = ScriptedProvider([service_error()], final_turn("done"))
    before = sample("agent_retry_outcomes_total", {"dependency": key, "outcome": "recovered"})
    result = await run_agent(RunOptions(
        provider=provider,
        tools=ToolRegistry(),
        messages=[user_message("go")],
        retry_policy=fast_policy(),
        circuit=None,  # 熔断关闭
        circuit_dependency=key,
    ))
    assert result.status == "completed"
    after = sample("agent_retry_outcomes_total", {"dependency": key, "outcome": "recovered"})
    assert after - before == 1


# --- RAG 边界 --------------------------------------------------------------


def rag_client(
    transport, breakers: CircuitBreakers, base_url: str = "http://rag.test"
) -> RagClient:
    return RagClient(
        RagClientConfig(base_url=base_url), transport=transport, breakers=breakers
    )


def chunk_context() -> dict:
    return {
        "chunk_id": "c1",
        "document_id": "d1",
        "product": "p",
        "version": "v1",
        "page_start": 3,
        "page_end": 3,
        "text": "torque 12 N·m",
    }


async def test_http_500_is_counted_as_failure_not_as_success() -> None:
    """回归：`_request` 把 500 当 (status, payload) 正常返回，按异常记账会记成成功。"""
    calls: list[str] = []

    async def transport(url: str, params: dict, headers: dict) -> tuple[int, object]:
        calls.append(url)
        return 500, {"detail": "no embedding model"}

    breakers = CircuitBreakers(CircuitPolicy(rag_failures=3), clock=FakeClock())
    client = rag_client(transport, breakers)
    for _ in range(3):
        with pytest.raises(RagError):
            await client.search("torque")
    assert breakers.is_open(circuit.rag_dependency("search", "http://rag.test")) is True

    with pytest.raises(RagError, match="circuit open"):
        await client.search("torque")
    assert len(calls) == 3  # 被拒绝的那次没有发请求


async def test_search_outage_does_not_block_reading_existing_evidence() -> None:
    """search 依赖 embedding；context 只是读原始片段，不该被一起挡住。"""

    async def transport(url: str, params: dict, headers: dict) -> tuple[int, object]:
        if url.endswith("/search"):
            return 500, {"detail": "embedding down"}
        return 200, chunk_context()

    breakers = CircuitBreakers(CircuitPolicy(rag_failures=2), clock=FakeClock())
    client = rag_client(transport, breakers)
    for _ in range(2):
        with pytest.raises(RagError):
            await client.search("torque")
    assert breakers.is_open(circuit.rag_dependency("search", "http://rag.test")) is True

    payload = await client.context("d1", "c1")
    assert payload["text"] == "torque 12 N·m"
    assert breakers.is_open(circuit.rag_dependency("context", "http://rag.test")) is False


async def test_client_errors_and_empty_hits_do_not_trip_the_circuit() -> None:
    calls = {"count": 0}

    async def transport(url: str, params: dict, headers: dict) -> tuple[int, object]:
        calls["count"] += 1
        if calls["count"] <= 10:
            return 404, {"detail": "unknown chunk"}
        return 200, []

    breakers = CircuitBreakers(CircuitPolicy(rag_failures=3), clock=FakeClock())
    client = rag_client(transport, breakers)
    for _ in range(10):
        with pytest.raises(RagError):
            await client.search("torque")
    assert breakers.is_open(circuit.rag_dependency("search", "http://rag.test")) is False
    assert await client.search("torque") == []  # 空检索是正常结果


async def test_rag_transport_errors_count_but_local_errors_do_not() -> None:
    async def refused(url: str, params: dict, headers: dict) -> tuple[int, object]:
        raise httpx.ConnectError("connection refused")

    breakers = CircuitBreakers(CircuitPolicy(rag_failures=2), clock=FakeClock())
    client = rag_client(refused, breakers)
    for _ in range(2):
        with pytest.raises(httpx.ConnectError):
            await client.search("torque")
    assert breakers.is_open(circuit.rag_dependency("search", "http://rag.test")) is True

    async def missing_httpx(url: str, params: dict, headers: dict) -> tuple[int, object]:
        raise RagError("httpx is required for the default RAG transport")

    other = CircuitBreakers(CircuitPolicy(rag_failures=2), clock=FakeClock())
    local = rag_client(missing_httpx, other, base_url="http://local-bug.test")
    for _ in range(5):
        with pytest.raises(RagError):
            await local.search("torque")
    assert other.is_open(circuit.rag_dependency("search", "http://local-bug.test")) is False


@pytest.fixture
def rag_500_server():
    """真实回环 HTTP：证明 500 走的是真实传输路径，不是注入的假返回值。"""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            payload = json.dumps({"detail": "injected 500"}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01))
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


async def test_real_http_500_trips_the_search_circuit(rag_500_server) -> None:
    breakers = CircuitBreakers(CircuitPolicy(rag_failures=2), clock=FakeClock())
    client = RagClient(RagClientConfig(base_url=rag_500_server, timeout=5), breakers=breakers)
    for _ in range(2):
        with pytest.raises(RagError):
            await client.search("torque")
    assert breakers.is_open(circuit.rag_dependency("search", rag_500_server)) is True
    started = time.perf_counter()
    with pytest.raises(RagError, match="circuit open"):
        await client.search("torque")
    assert time.perf_counter() - started < 1.0  # 被拒绝时不发请求、不等超时


# --- Web 验收：一个进程里三个任务 -------------------------------------------


@pytest.fixture
def switchable_model_server():
    """先持续 503，`healthy` 置真后返回正常 SSE；记录物理请求数。"""
    state = {"healthy": False, "requests": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            state["requests"] += 1
            if state["healthy"]:
                payload = json.dumps({"choices": [{
                    "delta": {"content": "recovered"},
                    "finish_reason": "stop",
                }]})
                body = f"data: {payload}\n\ndata: [DONE]\n\n".encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
            else:
                body = json.dumps({"error": {"message": "injected outage"}}).encode()
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01))
    thread.start()
    try:
        yield SimpleNamespace(url=f"http://127.0.0.1:{server.server_port}/v1", state=state)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


async def test_web_tasks_share_one_circuit_then_recover_after_cooldown(
    tmp_path, monkeypatch, switchable_model_server,
) -> None:
    """一个进程三个任务：故障开闸 → 零请求拒绝 → 冷却后探测恢复。

    阈值/冷却用测试缩短值（2 次失败 / 1 s），生产默认仍是 6 / 30 s。
    """
    pytest.importorskip("fastapi")
    from app import cli as cli_module
    from app.web import API_PREFIX, create_app

    monkeypatch.setenv("CIRCUIT_BREAKER_FAILURES", "2")
    monkeypatch.setenv("CIRCUIT_BREAKER_COOLDOWN_SECONDS", "1")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    server = switchable_model_server
    args = cli_module.build_parser().parse_args([
        "--workspace", str(workspace), "--state-dir", str(tmp_path / "state"),
        "--base-url", server.url, "--model", "web-circuit",
        "--no-memory", "--no-compaction",
    ])
    app = create_app(args)
    key = circuit.model_dependency(server.url, "web-circuit")

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client,
    ):
        conversation = await client.post(
            f"{API_PREFIX}/conversations", json={"title": "circuit", "project_id": "local"},
        )
        conversation_id = conversation.json()["id"]

        async def run_task(prompt: str) -> dict:
            created = await client.post(f"{API_PREFIX}/tasks", json={
                "conversation_id": conversation_id,
                "prompt": prompt,
                "idempotency_key": str(uuid4()),
            })
            assert created.status_code == 201
            identifier = created.json()["id"]
            await asyncio.wait_for(app.state.agent.jobs[identifier].runner, 15)
            return app.state.agent.jobs[identifier].record

        first = await run_task("first")
        assert first["status"] == "failed"
        assert server.state["requests"] == 2  # 阈值 2：第二次失败即开闸
        assert app.state.agent.breakers.states()[key] is State.OPEN

        before = server.state["requests"]
        second = await run_task("second")
        assert second["status"] == "failed"
        assert "circuit open" in (second["response"].get("error") or "")
        assert server.state["requests"] == before  # 零物理请求

        server.state["healthy"] = True
        await asyncio.sleep(1.1)  # 越过冷却；生产是 30 s
        third = await run_task("third")
        assert third["status"] == "completed"
        assert third["response"]["answer"] == "recovered"
        assert server.state["requests"] == before + 1  # 一次探测就够
        assert app.state.agent.breakers.states()[key] is State.CLOSED

        for record in (first, second, third):
            stream = await client.get(f"{API_PREFIX}/tasks/{record['id']}/events")
            events = [json.loads(line[6:]) for line in stream.text.splitlines()
                      if line.startswith("data: ")]
            assert events[-1]["type"] == "TaskFinished"
            assert events[-1]["payload"]["status"] == record["status"]
