"""provider 错误分类：给 retry 提供可路由的错误码。

对齐 DSH：

- `packages/llm/llm/src/error.ts` —— `HarnessError(message, code, { cause })`：
  稳定的机器可路由 `code`，**路由只看 code，不解析 message**。
- `packages/llm/llm/src/retry-policy.ts` —— 默认可重试集合
  `EMPTY_RESPONSE / RATE_LIMIT / SERVER / TIMEOUT / TRANSPORT`。
- `packages/llm/llm-deepseek/src/adapter.ts::httpErrorCode` —— HTTP 状态码到 code 的映射。
- `packages/llm/llm-pi-ai/src/stream.ts::classifyPiAiError` —— 异常文本到 code 的映射。
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum


class ErrorCode(StrEnum):
    """DSH 的稳定错误码。HTTP_<status> 这类动态码不在枚举内，直接以字符串表示。"""

    ABORTED = "ABORTED"
    AUTH = "AUTH"
    CONTEXT_WINDOW_EXCEEDED = "CONTEXT_WINDOW_EXCEEDED"
    EMPTY_RESPONSE = "EMPTY_RESPONSE"
    INVALID_CREDENTIAL = "INVALID_CREDENTIAL"
    INVALID_REQUEST = "INVALID_REQUEST"
    PROVIDER = "PROVIDER"
    QUOTA = "QUOTA"
    RATE_LIMIT = "RATE_LIMIT"
    SERVER = "SERVER"
    STREAM_CLOSED = "STREAM_CLOSED"
    TIMEOUT = "TIMEOUT"
    TRANSPORT = "TRANSPORT"


#: DSH retry-policy.ts 的 DEFAULT_RETRYABLE_CODES。
DEFAULT_RETRYABLE_CODES = frozenset(
    {
        ErrorCode.EMPTY_RESPONSE,
        ErrorCode.RATE_LIMIT,
        ErrorCode.SERVER,
        ErrorCode.TIMEOUT,
        ErrorCode.TRANSPORT,
    }
)


class ProviderError(Exception):
    """一次模型请求失败。`retryable` 由 code 推导，除非显式覆盖。"""

    def __init__(
        self,
        message: str,
        code: str | ErrorCode = ErrorCode.PROVIDER,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.status_code = status_code
        self.retry_after = retry_after  # 秒；来自 Retry-After
        self.retryable = (
            self.code in DEFAULT_RETRYABLE_CODES if retryable is None else retryable
        )


class TransportError(ProviderError):
    """HTTP 层失败（非 2xx）。code 由状态码推导。"""


_QUOTA_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"insufficient[\s_-]+(?:quota|balance|credits?)",
        r"(?:quota|usage[\s_-]+limit)[\s_-]+(?:exceeded|exhausted|reached)",
        r"exceed(?:ed|s)?[\s_-]+(?:(?:your|the)[\s_-]+)?(?:current[\s_-]+)?quota",
        r"(?:balance|credits?)[\s_-]+(?:exhausted|depleted)",
        r"out[\s_-]+of[\s_-]+(?:credits?|budget)",
    )
)

_CONTEXT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?:^|[^a-z0-9])context[\s_-](?:length|window)[\s_-]"
        r"(?:exceed(?:ed|s)?|overflow(?:ed)?|limit[\s_-]exceeded)(?:$|[^a-z0-9])",
        r"(?:maximum|max)(?:\s+(?:allowed|supported))?\s+context\s+(?:length|window)",
        r"\b(?:request|prompt|input|messages?)\s+(?:is\s+|are\s+)?too\s+(?:large|long)\s+for\s+"
        r"(?:(?:this|the)\s+)?(?:model(?:'s)?\s+)?context(?:\s+window)?\b",
        r"\b(?:input|prompt|request)\s+(?:is\s+)?too\s+(?:long|large)\s+for\s+(?:this|the)\s+model\b",
    )
)


def is_quota_exceeded(detail: str) -> bool:
    """识别「配额/余额耗尽」而不是瞬时限流（DSH `isQuotaExceededError`）。"""
    return any(pattern.search(detail) for pattern in _QUOTA_PATTERNS)


def is_context_window_exceeded(detail: str) -> bool:
    """识别「超出上下文窗口」（DSH `isContextWindowExceededError`）。"""
    return any(pattern.search(detail) for pattern in _CONTEXT_PATTERNS)


def http_error_code(status: int, detail: str = "") -> str:
    """HTTP 状态码 → 错误码（DSH `httpErrorCode`）。"""
    if status in (401, 403):
        return ErrorCode.AUTH
    if status == 413:
        return ErrorCode.INVALID_REQUEST
    if is_quota_exceeded(detail):
        return ErrorCode.QUOTA
    if status == 429:
        return ErrorCode.RATE_LIMIT
    if status == 400:
        return (
            ErrorCode.CONTEXT_WINDOW_EXCEEDED
            if is_context_window_exceeded(detail)
            else ErrorCode.INVALID_REQUEST
        )
    if status == 408:
        return ErrorCode.TIMEOUT
    if status >= 500:
        return ErrorCode.SERVER
    return f"HTTP_{status}"


def classify_exception(error: BaseException) -> str:
    """异常 → 错误码（DSH `classifyPiAiError` 的顺序与判定）。"""
    if isinstance(error, ProviderError):
        return error.code
    if isinstance(error, asyncio.CancelledError):
        return ErrorCode.ABORTED
    message = str(error)
    if re.search(r"\b(?:401|403)\b", message):
        return ErrorCode.AUTH
    if is_quota_exceeded(message):
        return ErrorCode.QUOTA
    if re.search(r"\b429\b|rate.?limit", message, re.IGNORECASE):
        return ErrorCode.RATE_LIMIT
    if re.search(
        r"\b413\b|failed to buffer the request body:\s*length limit exceeded"
        r"|payload too large|request body too large",
        message,
        re.IGNORECASE,
    ):
        return ErrorCode.INVALID_REQUEST
    if re.search(r"\b400\b|invalid.?request", message, re.IGNORECASE):
        return ErrorCode.INVALID_REQUEST
    if re.search(r"\b5\d\d\b", message):
        return ErrorCode.SERVER
    if re.search(r"\btime(?:d)?\s*out\b|timeout", message, re.IGNORECASE):
        return ErrorCode.TIMEOUT
    if re.search(r"stream ended (?:before|without)\b", message, re.IGNORECASE):
        return ErrorCode.TRANSPORT
    if re.search(
        r"\b(?:network|connection|socket|fetch)\b|\bECONN[A-Z]+\b"
        r"|\b(?:other side closed|premature close|terminated)\b",
        message,
        re.IGNORECASE,
    ):
        return ErrorCode.TRANSPORT
    if isinstance(error, TimeoutError | OSError):
        return ErrorCode.TRANSPORT
    return ErrorCode.PROVIDER


def as_provider_error(error: BaseException) -> ProviderError:
    """把任意异常包成 ProviderError；已经是 ProviderError 的原样返回。"""
    if isinstance(error, ProviderError):
        return error
    return ProviderError(str(error) or type(error).__name__, classify_exception(error))


def has_code(error: BaseException, code: str | ErrorCode) -> bool:
    """按**值**判断错误码。

    `ProviderError.code` 是 `str`，所以 `error.code is ErrorCode.ABORTED` 永远是 False
    —— `is` 比的是对象身份，不是枚举值。所有「按 code 分流」的判断都必须走这里。
    """
    actual = getattr(error, "code", None)
    return isinstance(actual, str) and actual == str(code)


def retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    """解析 `Retry-After`（秒或 HTTP 日期），沿用 DSH adapter 的处理；单位换算成秒。"""
    for key, value in headers.items():
        if key.lower() != "retry-after":
            continue
        value = value.strip()
        if value.isdigit():
            seconds = float(value)
            if seconds > 0:
                return seconds
            continue
        try:
            moment = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            continue
        if moment is None:
            continue
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        remaining = (moment - datetime.now(UTC)).total_seconds()
        if remaining > 0:
            return remaining
    return None
