"""provider：响应解析、finish_reason 映射、坏响应报错，全程不访问真实网络。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import pytest

from app.assembler import BlockAssembler
from app.contracts import (
    AssistantTurn,
    Message,
    ModelRequest,
    StreamChunk,
    StreamFinished,
    ToolCall,
    ToolResult,
    Usage,
)
from app.errors import ErrorCode, ProviderError
from app.providers import (
    MockProvider,
    MockProviderExhausted,
    OpenAiCompatibleProvider,
    StreamTransport,
    TransportError,
    TransportRequest,
)
from app.providers.openai_compatible import (
    chunks_from_sse,
    to_wire_message,
    to_wire_tool,
)
from app.tools import FunctionTool, ToolRegistry


def echo_tool() -> FunctionTool:
    return FunctionTool(
        name="echo",
        description="Echo text back",
        handler=lambda args, context: ToolResult(content=str(args.get("text", ""))),
    )


def request(messages: list[Message], registry: ToolRegistry) -> ModelRequest:
    return ModelRequest(messages=messages, tools=registry.list())



def sse_text(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def sse_transport(seen: list[TransportRequest], *payloads: dict) -> StreamTransport:
    """回放固定 SSE 的 transport；`[DONE]` 由这里统一补上。"""

    def transport(request: TransportRequest) -> AsyncIterator[str]:
        seen.append(request)

        async def source() -> AsyncIterator[str]:
            for payload in payloads:
                yield sse_text(payload)
            yield "data: [DONE]\n\n"

        return source()

    return transport


async def collect(
    provider: OpenAiCompatibleProvider, req: ModelRequest
) -> list[StreamChunk]:
    return [event async for event in provider.generate_stream(req)]


async def test_summary_output_cap_is_sent_to_the_real_wire_body() -> None:
    seen: list[TransportRequest] = []
    provider = OpenAiCompatibleProvider(
        base_url="https://example.test/v1", model="model",
        stream_transport=sse_transport(
            seen, {"choices": [{"delta": {"content": "summary"}, "finish_reason": "stop"}]},
        ),
    )
    await collect(provider, ModelRequest(
        [Message("user", "summarize")], [], purpose="compaction", max_output_tokens=8192,
    ))
    assert json.loads(seen[0].body)["max_tokens"] == 8192


async def test_provider_builds_request_and_uses_injected_transport() -> None:
    seen: list[TransportRequest] = []
    provider = OpenAiCompatibleProvider(
        base_url="https://example.test/v1/",
        model="deepseek-v4-flash",
        api_key="test-key",
        stream_transport=sse_transport(
            seen, {"choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}]}
        ),
    )
    registry = ToolRegistry()

    events = await collect(provider, request([Message(role="user", content="go")], registry))

    assert isinstance(events[-1], StreamFinished)
    assert assemble(events).turn().message.content == "done"
    assert seen[0].url == "https://example.test/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer test-key"
    body = json.loads(seen[0].body)
    assert body["model"] == "deepseek-v4-flash"
    assert body["messages"] == [{"role": "user", "content": "go"}]
    # 只有流式请求：wire 上一定带 stream + usage trailer。
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert "tools" not in body


async def test_provider_sends_tools_and_tool_messages_in_wire_format() -> None:
    seen: list[TransportRequest] = []
    provider = OpenAiCompatibleProvider(
        base_url="https://example.test/v1",
        model="m",
        stream_transport=sse_transport(
            seen, {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
        ),
    )
    registry = ToolRegistry().register(echo_tool())
    messages = [
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "hi"})],
        ),
        Message(role="tool", content="hi", tool_call_id="call-1", name="echo"),
    ]

    await collect(provider, request(messages, registry))

    body = json.loads(seen[0].body)
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo text back",
                "parameters": {"type": "object"},
            },
        }
    ]
    assert body["messages"][0]["tool_calls"][0]["function"]["arguments"] == '{"text": "hi"}'
    assert body["messages"][1] == {
        "role": "tool",
        "content": "hi",
        "tool_call_id": "call-1",
        "name": "echo",
    }


async def test_http_error_status_raises_transport_error() -> None:
    def transport(request: TransportRequest) -> AsyncIterator[str]:
        async def source() -> AsyncIterator[str]:
            raise TransportError("chat completion failed: 429 rate limited", ErrorCode.RATE_LIMIT)
            yield ""  # pragma: no cover - 仅为把函数变成 async generator

        return source()

    provider = OpenAiCompatibleProvider(
        base_url="https://example.test/v1", model="m", stream_transport=transport
    )

    with pytest.raises(TransportError):
        await collect(provider, request([Message(role="user", content="go")], ToolRegistry()))


async def test_mock_provider_records_requests_and_exhausts() -> None:
    provider = MockProvider(
        [AssistantTurn(message=Message(role="assistant", content="done"), stop_reason="stop")]
    )

    events = await collect(
        provider, request([Message(role="user", content="go")], ToolRegistry())
    )
    assert assemble(events).turn().message.content == "done"
    assert len(provider.requests) == 1

    with pytest.raises(MockProviderExhausted):
        await collect(provider, request([Message(role="user", content="go")], ToolRegistry()))


def test_wire_helpers_round_trip_a_tool_message() -> None:
    wire = to_wire_message(Message(role="tool", content="hi", tool_call_id="c1", name="echo"))

    assert wire == {"role": "tool", "content": "hi", "tool_call_id": "c1", "name": "echo"}
    assert to_wire_tool(ToolRegistry().register(echo_tool()).list()[0])["type"] == "function"


# --- 流式（Phase A）-----------------------------------------------------------


async def sse_chunks(lines: list[str]) -> list[StreamChunk]:
    async def source():  # noqa: ANN202
        for line in lines:
            yield line

    return [chunk async for chunk in chunks_from_sse(source())]


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def assemble(chunks: list[StreamChunk]):  # noqa: ANN201
    assembler = BlockAssembler()
    for chunk in chunks:
        assembler.push(chunk)
    return assembler


async def test_chunks_from_sse_opens_closes_blocks_and_defers_termination() -> None:
    chunks = await sse_chunks(
        [
            sse({"choices": [{"delta": {"content": "hello "}, "finish_reason": None}]}),
            sse({"choices": [{"delta": {"content": "world"}}]}),
            sse(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-1",
                                        "function": {"name": "echo", "arguments": '{"text":"hi"}'},
                                    }
                                ]
                            }
                        }
                    ]
                }
            ),
            sse(
                {
                    "choices": [{"finish_reason": "tool_calls"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                }
            ),
            "data: [DONE]\n\n",
        ]
    )

    # block-end / usage / finish 全部延迟到 [DONE]，与 DSH translate.ts 一致。
    assert [type(chunk).__name__ for chunk in chunks] == [
        "BlockStart",
        "TextDelta",
        "TextDelta",
        "BlockStart",
        "ToolCallDelta",
        "BlockEnd",
        "BlockEnd",
        "UsageChunk",
        "StreamFinished",
    ]

    turn = assemble(chunks).turn()
    assert turn.message.content == "hello world"
    assert turn.message.tool_calls == [
        ToolCall(id="call-1", name="echo", arguments={"text": "hi"})
    ]
    assert turn.stop_reason == "tool_calls"
    assert turn.usage == Usage(input_tokens=3, output_tokens=2, total_tokens=5)


async def test_sse_reasoning_opens_its_own_block_before_text() -> None:
    chunks = await sse_chunks(
        [
            sse({"choices": [{"delta": {"reasoning_content": "think"}}]}),
            sse({"choices": [{"delta": {"content": "answer"}}]}),
            sse({"choices": [{"finish_reason": "stop"}]}),
            "data: [DONE]\n\n",
        ]
    )

    assert [type(chunk).__name__ for chunk in chunks] == [
        "BlockStart",
        "ReasoningDelta",
        "BlockStart",
        "TextDelta",
        "BlockEnd",
        "BlockEnd",
        "StreamFinished",
    ]
    # reasoning 先开块，所以拿到 index 0；text 是 index 1。
    assert chunks[0].block_type == "reasoning"
    assert chunks[2].block_type == "text"


async def test_missing_finish_reason_defaults_to_stop() -> None:
    """DSH `translate.ts`：`pendingFinish ?? {kind: 'stop'}` —— 缺 finish_reason 不是错误。"""
    chunks = await sse_chunks(
        [sse({"choices": [{"delta": {"content": "dangling"}}]}), "data: [DONE]\n\n"]
    )

    assert assemble(chunks).finish.kind == "stop"
    assert assemble(chunks).turn().message.content == "dangling"


async def test_sse_without_done_is_stream_closed() -> None:
    """DSH `sse.ts`：没有 `[DONE]` 结尾才算 STREAM_CLOSED。"""
    with pytest.raises(ProviderError) as caught:
        await sse_chunks([sse({"choices": [{"delta": {"content": "dangling"}}]})])

    assert caught.value.code == ErrorCode.STREAM_CLOSED
    assert caught.value.retryable is True


@pytest.mark.parametrize("delta", [{}, {"content": ""}, {"reasoning_content": "think"}])
async def test_empty_completion_becomes_an_error_finish(delta) -> None:
    """明确 stop 却一个 block 都没有 → terminal error finish（EMPTY_RESPONSE）。"""
    chunks = await sse_chunks(
        [sse({"choices": [{"delta": delta, "finish_reason": "stop"}]}), "data: [DONE]\n\n"]
    )

    finish = assemble(chunks).finish
    assert finish.kind == "error"
    assert finish.failure is not None
    assert finish.failure.code == ErrorCode.EMPTY_RESPONSE.value


async def test_tool_arguments_that_are_not_json_are_kept_raw() -> None:
    """DSH `parseArguments`：坏 JSON 保留原文，不让整个响应失败。"""
    chunks = await sse_chunks(
        [
            sse(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-1",
                                        "function": {"name": "echo", "arguments": "not-json"},
                                    }
                                ]
                            }
                        }
                    ]
                }
            ),
            sse({"choices": [{"finish_reason": "tool_calls"}]}),
            "data: [DONE]\n\n",
        ]
    )

    turn = assemble(chunks).turn()
    assert turn.message.tool_calls == [
        ToolCall(id="call-1", name="echo", arguments="not-json")
    ]


async def test_tool_with_non_json_arguments_reports_an_argument_error() -> None:
    """坏参数留给工具层报错，而不是让整个 run 崩掉。"""
    registry = ToolRegistry().register(echo_tool())

    result = await registry.execute(ToolCall(id="c", name="echo", arguments="not-json"))

    assert result.is_error
    assert "not a JSON object" in result.content


async def test_sse_cancellation_stops_the_stream() -> None:
    cancel_event = asyncio.Event()
    cancel_event.set()

    async def source():  # noqa: ANN202
        yield sse({"choices": [{"delta": {"content": "never"}}]})

    with pytest.raises(ProviderError) as caught:
        async for _ in chunks_from_sse(source(), cancelled=cancel_event.is_set):
            pass

    assert caught.value.code == ErrorCode.ABORTED


async def test_stream_transport_http_errors_are_classified() -> None:
    async def failing(request: TransportRequest) -> AsyncIterator[str]:
        raise TransportError(
            "chat completion failed: 429 slow down",
            ErrorCode.RATE_LIMIT,
            status_code=429,
            retry_after=2.0,
        )
        yield ""  # 仅为让它是 async generator

    provider = OpenAiCompatibleProvider(
        base_url="https://example.test/v1", model="m", stream_transport=failing
    )

    with pytest.raises(TransportError) as caught:
        async for _ in provider.generate_stream(
            request([Message(role="user", content="go")], ToolRegistry())
        ):
            pass

    assert caught.value.code == ErrorCode.RATE_LIMIT
    assert caught.value.retryable is True
    assert caught.value.retry_after == 2.0


async def test_status_errors_carry_the_dsh_code() -> None:
    def transport(http_request: TransportRequest) -> AsyncIterator[str]:
        async def source() -> AsyncIterator[str]:
            raise TransportError(
                "chat completion failed: 401 unauthorized",
                ErrorCode.AUTH,
                status_code=401,
                retry_after=0.0,
            )
            yield ""  # pragma: no cover - 仅为把函数变成 async generator

        return source()

    provider = OpenAiCompatibleProvider(
        base_url="https://example.test/v1", model="m", stream_transport=transport
    )

    with pytest.raises(TransportError) as caught:
        await collect(provider, request([Message(role="user", content="go")], ToolRegistry()))

    assert caught.value.code == ErrorCode.AUTH
    assert caught.value.retryable is False
    assert caught.value.status_code == 401
