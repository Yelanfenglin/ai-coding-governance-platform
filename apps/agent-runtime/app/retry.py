"""对可重试的模型请求执行有限重试。

采用指数退避与对称抖动，优先使用 Retry-After，等待超过上限时终止。
等待可取消，计数在每个 step 重置；工具调用由执行器单独管理。
"""

from __future__ import annotations

import random as random_module
from collections.abc import Callable
from dataclasses import dataclass, field

from .errors import DEFAULT_RETRYABLE_CODES, ProviderError


@dataclass(frozen=True)
class RetryPolicy:
    """一次 step 内的模型请求重试策略。"""

    #: 首次请求之外的重试次数；5 表示最多发出 6 次请求。
    max_retries: int = 5
    #: 初始退避时间，单位为秒。
    base_delay: float = 0.5
    #: 最大退避时间，单位为秒。
    max_delay: float = 10.0
    jitter: bool = True
    #: 对称抖动比例。
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
        """计算指数退避与对称抖动，并限制在 max_delay 内。"""
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
                # 服务端要求的等待时间超过上限时终止重试。
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
