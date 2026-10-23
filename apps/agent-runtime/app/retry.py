"""provider 级 retry：只对可重试错误做有限重试。

对齐 DSH `packages/llm/llm-retry/src/index.ts` 与 `packages/llm/llm/src/retry-policy.ts`：

- 延迟 = `min(base_delay * 2 ** (retry - 1), max_delay)`，再乘对称 jitter `[1-r, 1+r]`；
- provider 给的 `Retry-After` 优先，但**超过 max_delay 时放弃重试**（DSH normal 模式的行为）；
- 等待可取消：取消立即停止，不再发起下一次 attempt；
- 重试计数按 step 重置（DSH 的 projection 在 `step/start` 时清空，这里直接用局部变量）。

工具调用不在 retry 范围内：DSH 也不重试工具，避免重复副作用。
"""

from __future__ import annotations

import random as random_module
from collections.abc import Callable
from dataclasses import dataclass, field

from .errors import DEFAULT_RETRYABLE_CODES, ProviderError


@dataclass(frozen=True)
class RetryPolicy:
    """一次 step 内的模型请求重试策略。"""

    #: 首次请求之外最多再试几次（DSH `DEFAULT_MAX_RETRIES`）：5 表示最多 6 次请求。
    max_retries: int = 5
    #: DSH `DEFAULT_INITIAL_DELAY_MS`。
    base_delay: float = 0.5
    #: DSH `DEFAULT_MAX_DELAY_MS`。
    max_delay: float = 10.0
    jitter: bool = True
    #: DSH `DEFAULT_JITTER_RATIO`：对称 jitter 的比例。
    jitter_ratio: float = 0.1
    retryable_codes: frozenset[str] = field(
        default_factory=lambda: frozenset(DEFAULT_RETRYABLE_CODES)
    )
    #: 测试用：注入 [0,1) 采样，让退避可确定。
    random: Callable[[], float] | None = None

    def sample(self) -> float:
        if not self.jitter:
            return 0.5
        if self.random is not None:
            return self.random()
        return random_module.random()

    def local_delay(self, retry: int, sample: float) -> float:
        """DSH `localDelay`：指数退避 + 对称 jitter，最后夹到 max_delay。"""
        exponent = min(retry - 1, 1024)
        exponential = min(self.base_delay * (2**exponent), self.max_delay)
        ratio = self.jitter_ratio if self.jitter else 0.0
        jitter = 1 - ratio + 2 * ratio * sample
        return min(exponential * jitter, self.max_delay)

    def delay_for(self, retry: int, error: ProviderError) -> float | None:
        """第 `retry` 次重试（1-based）要等多久；`None` 表示放弃重试。"""
        retry_after = error.retry_after
        if retry_after is not None:
            if retry_after <= 0 or retry_after > self.max_delay:
                # DSH：provider 要求的等待超过上限时，normal 模式直接放弃。
                return None
            return retry_after
        return self.local_delay(retry, self.sample())

    def should_retry(self, error: ProviderError, retries_done: int) -> bool:
        """`retries_done` 是已经完成的重试次数（首次失败时为 0）。"""
        if retries_done >= self.max_retries:
            return False
        if not error.retryable:
            return False
        return error.code in self.retryable_codes
