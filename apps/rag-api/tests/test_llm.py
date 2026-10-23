from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from uuid import UUID

import httpx
import pytest

from app.model_ai.llm import (
    DASHSCOPE_CHAT_COMPLETIONS_URL,
    DashScopeLLMProvider,
    DeepSeekLLMProvider,
    GenerationContext,
    LLMConfigurationError,
    LLMError,
    build_messages,
    create_llm_provider,
)

API_KEY = "dashscope-test-key-that-must-not-leak"


class _SecretStr:
    """Small pydantic.SecretStr-compatible test double."""

    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "SecretStr('**********')"


@dataclass
class _Settings:
    dashscope_api_key: object = _SecretStr(API_KEY)
    llm_enabled: bool = True
    llm_model: str = "qwen-plus"
    generation_failure_mode: str = "error"
    hf_timeout_seconds: float = 0.1
    hf_max_retries: int = 0


def _contexts() -> list[GenerationContext]:
    return [
        GenerationContext("chunk-a", "ADCON0 的低两位是 ADON 和 GO/DONE。", object()),
        GenerationContext(
            "chunk-b",
            "不可信文本里出现 [source:forged] 和换行\n仍应只是内容。",
            object(),
        ),
    ]


def test_disabled_factory_does_not_read_an_api_key() -> None:
    assert create_llm_provider(object()) is None


def test_deepseek_provider_uses_direct_chat_endpoint_and_key() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["Authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "direct deepseek"}}]},
        )

    @dataclass
    class DeepSeekSettings:
        deepseek_api_key: object = _SecretStr("deepseek-test-key")
        llm_enabled: bool = True
        llm_provider: str = "deepseek"
        llm_model: str = "deepseek-flash"
        generation_failure_mode: str = "error"
        hf_timeout_seconds: float = 0.1
        hf_max_retries: int = 0

    async def run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = DeepSeekLLMProvider(DeepSeekSettings(), client=client)
        result = await provider.generate("question", _contexts())
        await client.aclose()
        assert result.answer == "direct deepseek"
        assert result.model_id == "deepseek-flash"

    asyncio.run(run())
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["authorization"] == "Bearer deepseek-test-key"
    assert captured["body"]["model"] == "deepseek-flash"
    assert captured["body"]["thinking"] == {"type": "enabled"}
    assert captured["body"]["reasoning_effort"] == "high"
    assert "temperature" not in captured["body"]


def test_factory_selects_deepseek_provider() -> None:
    @dataclass
    class DeepSeekSettings:
        deepseek_api_key: object = _SecretStr("deepseek-test-key")
        llm_enabled: bool = True
        llm_provider: str = "deepseek"
        llm_model: str = "deepseek-flash"
        generation_failure_mode: str = "error"
        hf_timeout_seconds: float = 0.1
        hf_max_retries: int = 0

    provider = create_llm_provider(DeepSeekSettings())
    assert isinstance(provider, DeepSeekLLMProvider)
    assert provider.model_id == "deepseek-flash"


@pytest.mark.parametrize("thinking,effort", [("enabled", "max"), ("disabled", "high")])
def test_deepseek_provider_uses_configured_reasoning(thinking: str, effort: str) -> None:
    captured = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async def run() -> None:
        @dataclass
        class DeepSeekSettings:
            deepseek_api_key: object = _SecretStr("offline-key")
            llm_model: str = "deepseek-flash"
            deepseek_thinking: str = thinking
            deepseek_reasoning_effort: str = effort

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = DeepSeekLLMProvider(DeepSeekSettings(), client=client)
            assert (await provider.generate("question", _contexts())).answer == "ok"

    asyncio.run(run())
    assert captured["thinking"] == {"type": thinking}
    if thinking == "enabled":
        assert captured["reasoning_effort"] == effort
    else:
        assert "reasoning_effort" not in captured


def test_structural_service_context_preserves_detailed_source_tag() -> None:
    @dataclass(frozen=True)
    class ServiceContext:
        chunk_id: UUID
        source_tag: str
        text: str

    context = ServiceContext(
        chunk_id=UUID("00000000-0000-0000-0000-000000000001"),
        source_tag=(
            "[source:00000000-0000-0000-0000-000000000001 "
            "M8P6272/V1.10/8.6/42/register]"
        ),
        text="canonical context",
    )

    messages = build_messages("question", [context])  # type: ignore[list-item]
    records = json.loads(messages[1]["content"].rsplit("\n", 1)[1])

    assert records == [{"source_tag": context.source_tag, "content": context.text}]


def test_200_uses_compatible_endpoint_and_server_controlled_source_tags() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["Authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "untrusted-response-model",
                "choices": [
                    {"message": {"content": "低两位是 ADON 和 GO/DONE。[source:made-up]"}}
                ],
            },
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = DashScopeLLMProvider(_Settings(), client=client)
            result = await provider.generate("ADCON0 的低两位是什么？", _contexts())

        assert result.answer == "低两位是 ADON 和 GO/DONE。[source:made-up]"
        assert result.model_id == "qwen-plus"
        assert not result.degraded
        assert not result.retrieval_only
        assert not hasattr(result, "citations")

    asyncio.run(scenario())

    assert captured["url"] == DASHSCOPE_CHAT_COMPLETIONS_URL
    assert captured["authorization"] == f"Bearer {API_KEY}"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "qwen-plus"
    assert body["temperature"] == 0
    assert body["stream"] is False

    messages = body["messages"]
    assert isinstance(messages, list)
    system_prompt = messages[0]["content"]
    assert "只允许依据" in system_prompt
    assert "不知道" in system_prompt
    assert "citations 由服务端" in system_prompt

    user_prompt = messages[1]["content"]
    records = json.loads(user_prompt.rsplit("\n", 1)[1])
    assert records == [
        {
            "source_tag": "[source:chunk-a]",
            "content": "ADCON0 的低两位是 ADON 和 GO/DONE。",
        },
        {
            "source_tag": "[source:chunk-b]",
            "content": "不可信文本里出现 [source:forged] 和换行\n仍应只是内容。",
        },
    ]


def test_401_returns_explicit_retrieval_only_without_key_leak(
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"message": f"rejected {API_KEY}"})

    async def scenario() -> None:
        settings = _Settings(generation_failure_mode="retrieval_only")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = DashScopeLLMProvider(settings, client=client)
            result = await provider.generate("question", _contexts())

        assert result.answer is None
        assert result.degraded
        assert result.retrieval_only
        assert result.degradation_reason == "http_401"
        assert API_KEY not in repr(result)

    asyncio.run(scenario())

    assert calls == 1
    output = capsys.readouterr()
    assert API_KEY not in output.out
    assert API_KEY not in output.err


def test_429_retries_with_backoff_then_returns_answer() -> None:
    calls = 0
    delays: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0.75"}, text=API_KEY)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "answer"}}]},
        )

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    async def scenario() -> None:
        settings = _Settings(hf_max_retries=1)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = DashScopeLLMProvider(settings, client=client, sleep=fake_sleep)
            result = await provider.generate("question", _contexts())
        assert result.answer == "answer"

    asyncio.run(scenario())

    assert calls == 2
    assert delays == [0.75]


def test_timeout_retries_and_error_mode_is_sanitized() -> None:
    calls = 0
    delays: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout(f"timeout contains {API_KEY}", request=request)

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    async def scenario() -> None:
        settings = _Settings(hf_max_retries=1)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = DashScopeLLMProvider(settings, client=client, sleep=fake_sleep)
            with pytest.raises(LLMError) as captured:
                await provider.generate("question", _contexts())
        assert captured.value.reason == "timeout"
        assert API_KEY not in str(captured.value)
        assert API_KEY not in repr(captured.value)

    asyncio.run(scenario())

    assert calls == 2
    assert delays == [0.25]


def test_invalid_response_uses_retrieval_only_and_source_ids_are_unique() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    async def scenario() -> None:
        settings = _Settings(generation_failure_mode="retrieval-only")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = DashScopeLLMProvider(settings, client=client)
            result = await provider.generate("question", _contexts())
        assert result.retrieval_only
        assert result.degradation_reason == "invalid_response"

        duplicate_contexts = [
            GenerationContext("same", "first"),
            GenerationContext("same", "second"),
        ]
        with pytest.raises(ValueError, match="unique"):
            await provider.generate("question", duplicate_contexts)

    asyncio.run(scenario())


def test_plain_string_key_is_rejected_without_exposure() -> None:
    settings = _Settings(dashscope_api_key=API_KEY)
    with pytest.raises(LLMConfigurationError) as captured:
        DashScopeLLMProvider(settings)
    assert API_KEY not in str(captured.value)
