"""依赖级熔断：三态状态机 + 半开单探测。

边界只有两个（运行与监控口径见仓库 `docs/observability.md`）：

- 模型请求：`loop._request_with_retry` 在**每次物理请求之前** acquire；
- Agent→RAG 客户端：`RagClient._request` 在 transport 之前 acquire，按 `search` / `context` 分键。

不变量（`tests/test_circuit.py` 逐条对）：

1. 检查点在物理请求之前，首次 attempt 也经过；被拒绝时不发网络请求。
2. 只统计**明确的依赖故障**（5xx / 传输 / 超时 / 流未收尾）。限流、鉴权、参数错误、
   未分类的本地异常只记 `IGNORED`，既不加计数**也不清零**。
3. 半开状态下同时只发一个探测许可；许可带 `generation` 与所有权，
   `release` 只清掉自己那份名额，旧 permit 不会误放新探测的名额。
4. 任何退出路径（成功 / 失败 / 不计数 / 取消）都必须 `release`。
5. 状态转换在锁内，网络调用在锁外。

`acquire`/`record`/`release` 都不 await，所以 asyncio 单线程下不会交错；
锁只是为了在测试或多线程宿主里也成立。
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from . import metrics
from .errors import ErrorCode, has_code


class Outcome(StrEnum):
    """一次物理请求对熔断器的反馈。"""

    SUCCESS = "success"
    FAILURE = "failure"
    #: 与依赖健康无关（限流 / 鉴权 / 参数 / 本地程序错误）：不加计数，也不清零。
    IGNORED = "ignored"
    CANCELLED = "cancelled"


class State(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


STATE_CODES = {State.CLOSED: 0, State.OPEN: 1, State.HALF_OPEN: 2}

#: 计入熔断的模型错误码。
#:
#: `PROVIDER` 故意不在内：`classify_exception` 把它当作兜底码，本地程序错误也会落到这里，
#: 计进去等于让本地 bug 触发熔断。`EMPTY_RESPONSE` 也排除：200 但没有内容可能是模型侧行为，
#: 不足以判定服务不可用。两者仍照常重试，只是不给熔断投喂失败。
DEPENDENCY_FAILURES = frozenset(
    {ErrorCode.SERVER, ErrorCode.TRANSPORT, ErrorCode.TIMEOUT, ErrorCode.STREAM_CLOSED}
)

#: 依赖键前缀：决定用哪一档阈值。
MODEL_PREFIX = "model:"
RAG_PREFIX = "rag:"


def model_dependency(base_url: str, model: str) -> str:
    return f"{MODEL_PREFIX}{base_url.rstrip('/')}|{model}"


def rag_dependency(operation: str, base_url: str) -> str:
    return f"{RAG_PREFIX}{operation}:{base_url.rstrip('/')}"


def provider_outcome(error: BaseException) -> Outcome:
    """把模型请求异常翻译成熔断反馈（只认明确的依赖故障）。"""
    if has_code(error, ErrorCode.ABORTED):
        return Outcome.CANCELLED
    if any(has_code(error, code) for code in DEPENDENCY_FAILURES):
        return Outcome.FAILURE
    # 兜底：适配器只带状态码、code 是动态串（如 `HTTP_503`）时也算服务侧故障。
    status = getattr(error, "status_code", None)
    if isinstance(status, int) and status >= 500:
        return Outcome.FAILURE
    return Outcome.IGNORED


def rag_outcome(status: int) -> Outcome:
    if status < 400:
        return Outcome.SUCCESS
    if status >= 500:
        return Outcome.FAILURE
    return Outcome.IGNORED


def transport_failure(error: BaseException) -> bool:
    """RAG 传输层异常：只认网络/超时，不认本地配置错误。"""
    if isinstance(error, TimeoutError | ConnectionError):
        return True
    httpx = _httpx()
    if httpx is None:
        return False
    if isinstance(error, httpx.UnsupportedProtocol | httpx.InvalidURL):
        return False  # base_url 写错了，是配置问题
    return isinstance(error, httpx.TransportError)


def _httpx():
    try:
        import httpx
    except ImportError:  # httpx 是可选依赖
        return None
    return httpx


class CircuitOpen(RuntimeError):
    """请求被熔断拒绝：没有发出网络调用。"""

    def __init__(self, dependency: str, retry_in: float) -> None:
        super().__init__(
            f"circuit open for {dependency}; next probe in at most {max(retry_in, 0):.1f}s"
        )
        self.dependency = dependency
        self.retry_in = max(retry_in, 0.0)


@dataclass(frozen=True)
class Permit:
    """一次物理请求的通行证。`generation` 区分状态轮次，`probe` 标记半开探测名额。"""

    dependency: str
    generation: int
    probe: bool = False


@dataclass(frozen=True)
class CircuitPolicy:
    """阈值只是初始候选，取值来源见设计文档 §5 / §9。"""

    #: 默认等于 `RetryPolicy.max_retries + 1`：单条请求的重试预算内不会因熔断被截断。
    model_failures: int = 6
    rag_failures: int = 3
    cooldown: float = 30.0

    def threshold_for(self, dependency: str) -> int:
        return self.rag_failures if dependency.startswith(RAG_PREFIX) else self.model_failures

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> CircuitPolicy:
        env = os.environ if environ is None else environ
        return cls(
            model_failures=_positive_int(env.get("CIRCUIT_BREAKER_FAILURES"), 6),
            rag_failures=_positive_int(env.get("CIRCUIT_BREAKER_RAG_FAILURES"), 3),
            cooldown=_positive_float(env.get("CIRCUIT_BREAKER_COOLDOWN_SECONDS"), 30.0),
        )


def _positive_int(raw: str | None, default: int) -> int:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _positive_float(raw: str | None, default: float) -> float:
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def _truthy(raw: str | None) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


class Breaker:
    """单个依赖的三态状态机。"""

    def __init__(
        self,
        dependency: str,
        *,
        failures: int,
        cooldown: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.dependency = dependency
        self.failure_threshold = max(1, int(failures))
        self.cooldown = max(0.0, float(cooldown))
        self._clock = clock
        self._lock = threading.Lock()
        self._state = State.CLOSED
        self._generation = 0
        self._failures = 0
        self._opened_at: float | None = None
        self._probe: Permit | None = None
        self._publish_state()

    # --- 只读视图 ---------------------------------------------------------
    @property
    def state(self) -> State:
        return self._state

    @property
    def consecutive_failures(self) -> int:
        return self._failures

    # --- 契约 -------------------------------------------------------------
    def acquire(self) -> Permit:
        """物理请求前调用。拒绝时抛 `CircuitOpen`，调用方不得发网络请求。"""
        with self._lock:
            now = self._clock()
            if self._state is State.OPEN:
                retry_at = (self._opened_at if self._opened_at is not None else now) + self.cooldown
                if now < retry_at:
                    self._reject(retry_at - now)
                # 冷却到期只是“允许下一次请求试一次”：没有请求就一直开着。
                self._enter(State.HALF_OPEN)
            if self._state is State.HALF_OPEN:
                if self._probe is not None:
                    self._reject(self.cooldown)
                permit = Permit(self.dependency, self._generation, probe=True)
                self._probe = permit
                return permit
            return Permit(self.dependency, self._generation)

    def record(self, permit: Permit | None, outcome: Outcome) -> None:
        """反馈一次物理请求的结果。陈旧 permit 只记日志，不改状态。"""
        if permit is None:
            return
        with self._lock:
            if permit.generation != self._generation:
                self._stale(permit, outcome)
                return
            if permit.probe and self._probe is not permit:
                # 同一轮半开里被替换掉的旧探测（例如已 release 后名额被重新发放）。
                self._stale(permit, outcome)
                return
            if self._state is State.HALF_OPEN and permit.probe:
                if outcome is Outcome.SUCCESS:
                    self._enter(State.CLOSED)
                elif outcome is Outcome.FAILURE:
                    self._enter(State.OPEN)
                # IGNORED / CANCELLED：保持半开，名额由 release 归还，下一个请求可再探测。
                return
            if outcome is Outcome.FAILURE:
                self._failures += 1
                if self._failures >= self.failure_threshold:
                    self._enter(State.OPEN)
            elif outcome is Outcome.SUCCESS:
                self._failures = 0

    def release(self, permit: Permit | None) -> None:
        """归还探测名额。幂等；只清自己的那一份。"""
        if permit is None or not permit.probe:
            return
        with self._lock:
            if self._probe is permit:
                self._probe = None

    # --- 内部 -------------------------------------------------------------
    def _enter(self, state: State) -> None:
        previous, now = self._state, self._clock()
        if previous is state:
            return
        self._state = state
        self._generation += 1
        if state is State.OPEN:
            self._opened_at = now
            self._probe = None
        elif state is State.CLOSED:
            if self._opened_at is not None:
                metrics.CIRCUIT_RECOVERY.labels(self.dependency).observe(now - self._opened_at)
            self._opened_at = None
            self._failures = 0
            self._probe = None
        self._publish_state()
        metrics.CIRCUIT_TRANSITIONS.labels(self.dependency, str(state)).inc()
        if state is State.OPEN:
            metrics.log("circuit.opened", dependency=self.dependency,
                        previous=str(previous), consecutive_failures=self._failures)
        elif state is State.CLOSED:
            metrics.log("circuit.closed", dependency=self.dependency, previous=str(previous))
        else:
            metrics.log("circuit.half_open", dependency=self.dependency, previous=str(previous))

    def _publish_state(self) -> None:
        metrics.CIRCUIT_STATE.labels(self.dependency).set(STATE_CODES[self._state])

    def _reject(self, retry_in: float) -> None:
        metrics.CIRCUIT_REJECTED.labels(self.dependency).inc()
        metrics.log("circuit.rejected", dependency=self.dependency,
                    retry_in=round(retry_in, 3))
        raise CircuitOpen(self.dependency, retry_in)

    def _stale(self, permit: Permit, outcome: Outcome) -> None:
        metrics.log("circuit.stale_outcome", dependency=self.dependency,
                    outcome=str(outcome), generation=permit.generation,
                    current_generation=self._generation)


class CircuitBreakers:
    """按依赖键共享的进程内注册表。"""

    def __init__(self, policy: CircuitPolicy, *,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.policy = policy
        self._clock = clock
        self._lock = threading.Lock()
        self._breakers: dict[str, Breaker] = {}

    def breaker(self, dependency: str) -> Breaker:
        with self._lock:
            breaker = self._breakers.get(dependency)
            if breaker is None:
                breaker = Breaker(
                    dependency,
                    failures=self.policy.threshold_for(dependency),
                    cooldown=self.policy.cooldown,
                    clock=self._clock,
                )
                self._breakers[dependency] = breaker
            return breaker

    def acquire(self, dependency: str) -> Permit:
        return self.breaker(dependency).acquire()

    def record(self, permit: Permit | None, outcome: Outcome) -> None:
        if permit is None:
            return
        self.breaker(permit.dependency).record(permit, outcome)

    def release(self, permit: Permit | None) -> None:
        if permit is None:
            return
        self.breaker(permit.dependency).release(permit)

    def is_open(self, dependency: str) -> bool:
        return self.breaker(dependency).state is State.OPEN

    def states(self) -> dict[str, State]:
        return {key: breaker.state for key, breaker in self._breakers.items()}


def from_env(environ: Mapping[str, str] | None = None) -> CircuitBreakers | None:
    """按环境变量构造注册表；`CIRCUIT_BREAKER_DISABLED` 为真时返回 None（不启用）。"""
    env = os.environ if environ is None else environ
    if _truthy(env.get("CIRCUIT_BREAKER_DISABLED")):
        return None
    return CircuitBreakers(CircuitPolicy.from_env(env))
