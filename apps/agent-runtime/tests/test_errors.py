"""验证 HTTP 状态与异常分类，以及可重试错误集合。"""

from __future__ import annotations

import asyncio

import pytest

from app.errors import (
    DEFAULT_RETRYABLE_CODES,
    ErrorCode,
    ProviderError,
    as_provider_error,
    classify_exception,
    http_error_code,
    is_context_window_exceeded,
    is_quota_exceeded,
    retry_after_seconds,
)


@pytest.mark.parametrize(
    ("status", "detail", "expected"),
    [
        (401, "", ErrorCode.AUTH),
        (403, "", ErrorCode.AUTH),
        (413, "", ErrorCode.INVALID_REQUEST),
        (429, "", ErrorCode.RATE_LIMIT),
        (400, "", ErrorCode.INVALID_REQUEST),
        (400, "maximum context length exceeded", ErrorCode.CONTEXT_WINDOW_EXCEEDED),
        (408, "", ErrorCode.TIMEOUT),
        (500, "", ErrorCode.SERVER),
        (503, "", ErrorCode.SERVER),
        (418, "", "HTTP_418"),
        (402, "insufficient balance", ErrorCode.QUOTA),
    ],
)
def test_http_error_code_matches_dsh_mapping(status: int, detail: str, expected: str) -> None:
    assert http_error_code(status, detail) == expected


def test_quota_and_context_wordings_are_recognized() -> None:
    assert is_quota_exceeded("your current quota exceeded")
    assert is_quota_exceeded("insufficient credits")
    assert not is_quota_exceeded("rate limit reached")
    assert is_context_window_exceeded("context_length_exceeded")
    assert is_context_window_exceeded("this request is too large for the model context window")
    assert not is_context_window_exceeded("invalid request")


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("401 unauthorized", ErrorCode.AUTH),
        ("403 forbidden", ErrorCode.AUTH),
        ("rate limit reached", ErrorCode.RATE_LIMIT),
        ("429 too many requests", ErrorCode.RATE_LIMIT),
        ("request timed out", ErrorCode.TIMEOUT),
        ("upstream 503 unavailable", ErrorCode.SERVER),
        ("connection reset by peer", ErrorCode.TRANSPORT),
        ("fetch failed", ErrorCode.TRANSPORT),
        ("stream ended before message_stop", ErrorCode.TRANSPORT),
        ("ECONNRESET", ErrorCode.TRANSPORT),
        ("request body too large", ErrorCode.INVALID_REQUEST),
        ("400 invalid request", ErrorCode.INVALID_REQUEST),
        ("insufficient quota", ErrorCode.QUOTA),
        ("something weird", ErrorCode.PROVIDER),
    ],
)
def test_exception_classification(message: str, expected: ErrorCode) -> None:
    assert classify_exception(RuntimeError(message)) == expected


def test_classification_uses_builtin_types_and_cancellation() -> None:
    assert classify_exception(TimeoutError("boom")) == ErrorCode.TRANSPORT
    assert classify_exception(asyncio.CancelledError()) == ErrorCode.ABORTED
    assert classify_exception(ProviderError("nope", ErrorCode.AUTH)) == ErrorCode.AUTH


def test_default_retryable_codes_include_incomplete_streams() -> None:
    assert frozenset(
        {"EMPTY_RESPONSE", "RATE_LIMIT", "SERVER", "STREAM_CLOSED", "TIMEOUT", "TRANSPORT"}
    ) == DEFAULT_RETRYABLE_CODES


@pytest.mark.parametrize(
    ("code", "retryable"),
    [
        (ErrorCode.RATE_LIMIT, True),
        (ErrorCode.SERVER, True),
        (ErrorCode.TIMEOUT, True),
        (ErrorCode.TRANSPORT, True),
        (ErrorCode.EMPTY_RESPONSE, True),
        (ErrorCode.AUTH, False),
        (ErrorCode.INVALID_REQUEST, False),
        (ErrorCode.QUOTA, False),
        (ErrorCode.CONTEXT_WINDOW_EXCEEDED, False),
        (ErrorCode.STREAM_CLOSED, True),
        (ErrorCode.ABORTED, False),
    ],
)
def test_retryable_is_derived_from_the_code(code: ErrorCode, retryable: bool) -> None:
    assert ProviderError("boom", code).retryable is retryable


def test_retryable_can_be_overridden() -> None:
    assert ProviderError("boom", ErrorCode.AUTH, retryable=True).retryable is True
    assert ProviderError("boom", ErrorCode.SERVER, retryable=False).retryable is False


def test_status_code_and_retry_after_are_carried() -> None:
    error = ProviderError("slow down", ErrorCode.RATE_LIMIT, status_code=429, retry_after=2.5)

    assert error.code == "RATE_LIMIT"
    assert error.status_code == 429
    assert error.retry_after == 2.5


def test_retry_after_header_parsing() -> None:
    assert retry_after_seconds({"Retry-After": "3"}) == 3.0
    assert retry_after_seconds({"retry-after": "0"}) is None
    assert retry_after_seconds({"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}) is None
    assert retry_after_seconds({}) is None


def test_as_provider_error_wraps_unknown_exceptions() -> None:
    wrapped = as_provider_error(RuntimeError("connection reset by peer"))

    assert isinstance(wrapped, ProviderError)
    assert wrapped.code == ErrorCode.TRANSPORT
    assert wrapped.retryable is True

    original = ProviderError("nope", ErrorCode.AUTH)
    assert as_provider_error(original) is original
