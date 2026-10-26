"""OpenAI-compatible chat completions provider（可选）。

默认 transport 用 httpx，但 httpx 是 optional 依赖且延迟导入；
测试通过注入 transport 完成，不访问真实网络。

HTTP 状态码与异常一律转成 `app.errors.ProviderError`，code 沿用 DSH 的错误码
（`app/errors.py` 里 `http_error_code` / `classify_exception`）。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from opentelemetry import trace

from .. import metrics, observability
from ..contracts import (
    BlockEnd,
    BlockStart,
    BlockType,
    ContentBlock,
    FinishReason,
    LlmFailure,
    Message,
    ModelRequest,
    ReasoningBlock,
    ReasoningDelta,
    StreamChunk,
    StreamFinished,
    TextBlock,
    TextDelta,
    ToolCallBlock,
    ToolCallDelta,
    ToolDefinition,
    Usage,
    UsageChunk,
)
from ..errors import (
    ErrorCode,
    ProviderError,
    has_code,
    http_error_code,
    retry_after_seconds,
)
from ..observability import failure, record_usage


class TransportError(ProviderError):
    """HTTP 层失败（非 2xx）。code 由状态码推导。"""


class MalformedCompletion(ProviderError):
    """响应结构不符合预期：provider 语义错误，不可重试。"""

    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.PROVIDER)


@dataclass(slots=True)
class TransportRequest:
    url: str
    headers: dict[str, str]
    body: str
    cancelled: Callable[[], bool] | None = None


#: 唯一的 transport 形状：逐块产出 SSE 原始文本（通常按行）。
#:
#: **没有非流式 transport。** DSH `LlmAdapter.stream()` 是 "the only required
#: method"，adapter 只发流式请求；本地不再保留「拉完整 body 再解析」的第二条传输路径。
StreamTransport = Callable[[TransportRequest], AsyncIterator[str]]


class OpenAiCompatibleProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        stream_transport: StreamTransport | None = None,
        *,
        thinking: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.extra_headers = headers or {}
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        # transport 延迟解析：httpx 是 optional 依赖，只有真正发起请求时才需要它。
        self._stream_transport = stream_transport

    async def generate_stream(self, request: ModelRequest) -> AsyncIterator[StreamChunk]:
        # A stream is resumed/closed by different __anext__ tasks. Never keep an
        # attached context across yield (OpenHands RootSpan/use_span pattern).
        attributes = {
            "openinference.span.kind": "LLM", "llm.model_name": self.model,
            "gen_ai.request.model": self.model,
            "llm.invocation_parameters": json.dumps({
                "thinking": self.thinking, "reasoning_effort": self.reasoning_effort,
            }),
            "agent.request.purpose": request.purpose or "response",
        }
        span = observability.tracer.start_span("llm.request", attributes=attributes)
        measurement = metrics.ModelMeasurement(self.model, request.purpose or "response")
        started = time.perf_counter()
        first_chunk = True
        first_text = True
        finished = False
        translated = None
        try:
            chunks = self._resolve_stream_transport()(TransportRequest(
                url=self._url(), headers=self._headers(),
                body=json.dumps(self._body(request), ensure_ascii=False),
                cancelled=request.cancelled,
            ))
            translated = chunks_from_sse(chunks, cancelled=request.cancelled)
            while True:
                with trace.use_span(span, end_on_exit=False, record_exception=False,
                                    set_status_on_exception=False):
                    try:
                        chunk = await anext(translated)
                    except StopAsyncIteration:
                        break
                if first_chunk:
                    metrics.FIRST_EVENT.labels(self.model).observe(time.perf_counter() - started)
                    span.set_attribute("llm.first_chunk_ms", (time.perf_counter() - started) * 1000)
                    first_chunk = False
                if first_text and isinstance(chunk, TextDelta):
                    metrics.FIRST_TEXT.labels(self.model).observe(time.perf_counter() - started)
                    span.set_attribute("llm.first_text_ms", (time.perf_counter() - started) * 1000)
                    first_text = False
                if isinstance(chunk, UsageChunk):
                    record_usage(span, chunk.usage)
                    measurement.usage = chunk.usage
                elif isinstance(chunk, StreamFinished):
                    finished = True
                    measurement.outcome = "ok"
                    span.set_attribute("gen_ai.response.finish_reasons", [chunk.reason.kind])
                    if chunk.reason.kind == "aborted":
                        measurement.outcome = "cancelled"
                        span.set_attribute("agent.outcome", "cancelled")
                    elif chunk.reason.failure is not None:
                        measurement.outcome = "error"
                        measurement.code = chunk.reason.failure.code or chunk.reason.kind
                        failure(span, chunk.reason.failure.code or chunk.reason.kind)
                yield chunk
        except asyncio.CancelledError:
            measurement.outcome = "cancelled"
            raise
        except Exception as exc:
            if has_code(exc, ErrorCode.ABORTED):
                measurement.outcome = "cancelled"
                span.set_attribute("agent.outcome", "cancelled")
            else:
                measurement.outcome = "error"
                measurement.code = str(getattr(exc, "code", type(exc).__name__))
                failure(span, str(getattr(exc, "code", type(exc).__name__)))
            raise
        finally:
            if translated is not None:
                await _aclose_stream(translated)
            if not finished:
                span.set_attribute("agent.outcome", "interrupted")
            with trace.use_span(span, end_on_exit=False, record_exception=False,
                                set_status_on_exception=False):
                measurement.finish()
            span.end()

    def _resolve_stream_transport(self) -> StreamTransport:
        if self._stream_transport is None:
            self._stream_transport = httpx_stream_transport()
        return self._stream_transport

    def _url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _body(self, request: ModelRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [to_wire_message(message) for message in request.messages],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if request.tools:
            body["tools"] = [to_wire_tool(tool) for tool in request.tools]
        if request.max_output_tokens is not None:
            body["max_tokens"] = request.max_output_tokens
        if self.thinking is not None:
            body["thinking"] = {"type": self.thinking}
        if self.reasoning_effort is not None and self.thinking != "disabled":
            body["reasoning_effort"] = self.reasoning_effort
        return body

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers


def httpx_stream_transport(timeout: float = 60.0) -> StreamTransport:
    try:
        import httpx  # 延迟导入：同上
    except ImportError as exc:  # pragma: no cover - 只在未安装 httpx 时触发
        raise RuntimeError(
            "httpx is required for the default transport; install the 'http' extra"
        ) from exc

    async def transport(request: TransportRequest) -> AsyncIterator[str]:
        # Loopback services must not be sent through Windows/system HTTP proxies.
        trust_env = urlsplit(request.url).hostname not in {"127.0.0.1", "localhost", "::1"}
        try:
            async with (
                httpx.AsyncClient(timeout=timeout, trust_env=trust_env) as client,
                client.stream(
                    "POST", request.url, headers=request.headers, content=request.body
                ) as response,
            ):
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise TransportError(
                        f"chat completion failed: {response.status_code} {body}",
                        http_error_code(response.status_code, body),
                        status_code=response.status_code,
                        retry_after=retry_after_seconds(dict(response.headers)),
                    )

                # aiter_lines() removes line endings; restore them for the SSE parser.
                async for line in response.aiter_lines():
                    yield f"{line}\n"
        except httpx.TimeoutException as exc:
            # httpx timeouts need not contain a message or inherit builtin TimeoutError.
            raise TransportError(str(exc) or type(exc).__name__, ErrorCode.TIMEOUT) from exc
        except (httpx.NetworkError, httpx.RemoteProtocolError) as exc:
            raise TransportError(str(exc) or type(exc).__name__, ErrorCode.TRANSPORT) from exc

    return transport


# --- SSE → StreamChunk --------------------------------------------------------


@dataclass(slots=True)
class _OpenBlock:
    index: int
    kind: BlockType
    text: str = ""
    call_id: str | None = None
    name: str | None = None


def _accept_identity(current: str | None, incoming: object) -> str | None:
    """DSH `acceptIdentity`：非空字符串才覆盖，否则保留已有值。"""
    return incoming if isinstance(incoming, str) and incoming else current


def _close_block(block: _OpenBlock) -> ContentBlock:
    """DSH `closeBlock`：按 kind 产出最终 block。"""
    if block.kind == "text":
        return TextBlock(text=block.text)
    if block.kind == "reasoning":
        return ReasoningBlock(text=block.text)
    return ToolCallBlock(
        id=block.call_id or "", name=block.name or "", arguments=block.text
    )


class _SseTranslator:
    """把 SSE payload 翻成 `StreamChunk`（DSH `translate.ts`）。

    - reasoning 先于 content，同一 delta 内的顺序是固定的；
    - 空字符串**不**开块；
    - text / reasoning 各自全局唯一一个 block，tool-call 按上游 `index` 复用；
    - 所有 `block-end` 延迟到 `[DONE]`。
    """

    def __init__(self) -> None:
        self._next_index = 0
        self.text_block: _OpenBlock | None = None
        self.reasoning_block: _OpenBlock | None = None
        self.tool_blocks: dict[int, _OpenBlock] = {}
        self.order: list[_OpenBlock] = []
        self.usage: Usage | None = None
        self.finish: FinishReason | None = None

    def _open(self, kind: BlockType) -> _OpenBlock:
        block = _OpenBlock(index=self._next_index, kind=kind)
        self._next_index += 1
        self.order.append(block)
        return block

    def consume(self, payload: dict[str, Any]) -> list[StreamChunk]:
        """完整处理一个 SSE payload。同一块里可能同时有 delta 与 finish_reason。"""
        chunks: list[StreamChunk] = []
        span = trace.get_current_span()
        if isinstance(payload.get("model"), str):
            span.set_attribute("gen_ai.response.model", payload["model"])
        if isinstance(payload.get("id"), str):
            span.set_attribute("gen_ai.response.id", payload["id"])

        # usage 可能挂在 finish 块上，也可能是末尾独立的 usage-only 块，保留最新。
        usage = payload.get("usage")
        if isinstance(usage, dict):
            parsed_usage = _parse_usage(usage)
            if parsed_usage is not None:
                self.usage = parsed_usage

        choices = payload.get("choices")
        if not isinstance(choices, list):
            return chunks
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                chunks.extend(self._consume_delta(delta))
            reason = choice.get("finish_reason")
            if isinstance(reason, str):
                self.finish = _map_finish_reason(reason)
        return chunks

    def _consume_delta(self, delta: dict[str, Any]) -> list[StreamChunk]:
        chunks: list[StreamChunk] = []

        reasoning = delta.get("reasoning_content")
        if reasoning is None:
            reasoning = delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning != "":
            if self.reasoning_block is None:
                self.reasoning_block = self._open("reasoning")
                chunks.append(
                    BlockStart(index=self.reasoning_block.index, block_type="reasoning")
                )
            self.reasoning_block.text += reasoning
            chunks.append(ReasoningDelta(index=self.reasoning_block.index, text=reasoning))

        content = delta.get("content")
        if isinstance(content, str) and content != "":
            if self.text_block is None:
                self.text_block = self._open("text")
                chunks.append(BlockStart(index=self.text_block.index, block_type="text"))
            self.text_block.text += content
            chunks.append(TextDelta(index=self.text_block.index, text=content))

        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list):
            chunks.extend(self._consume_tool_calls(tool_calls))
        return chunks

    def _consume_tool_calls(self, tool_calls: list[Any]) -> list[StreamChunk]:
        chunks: list[StreamChunk] = []
        for item in tool_calls:
            if not isinstance(item, dict):
                continue
            raw_index = item.get("index")
            index = raw_index if isinstance(raw_index, int) else len(self.tool_blocks)
            block = self.tool_blocks.get(index)
            if block is None:
                block = self._open("tool-call")
                self.tool_blocks[index] = block
                chunks.append(BlockStart(index=block.index, block_type="tool-call"))
            block.call_id = _accept_identity(block.call_id, item.get("id"))
            fragment = ""
            function = item.get("function")
            if isinstance(function, dict):
                block.name = _accept_identity(block.name, function.get("name"))
                raw_arguments = function.get("arguments")
                if isinstance(raw_arguments, str):
                    fragment = raw_arguments
            block.text += fragment
            chunks.append(
                ToolCallDelta(
                    index=block.index,
                    call_id=block.call_id or "",
                    name=block.name,
                    arguments_delta=fragment,
                )
            )
        return chunks

    def close(self) -> list[StreamChunk]:
        """流结束时补齐 block-end / usage / finish。"""
        chunks: list[StreamChunk] = [
            BlockEnd(index=block.index, block=_close_block(block)) for block in self.order
        ]
        if self.usage is not None:
            chunks.append(UsageChunk(usage=self.usage))
        chunks.append(StreamFinished(reason=self._terminal_reason()))
        return chunks

    def _terminal_reason(self) -> FinishReason:
        reason = self.finish or FinishReason(kind="stop")
        # Reasoning alone is not an answer; an empty content with tool calls is valid.
        if reason.kind == "stop" and self.text_block is None and not self.tool_blocks:
            return FinishReason(
                kind="error",
                failure=LlmFailure(
                    message="model returned a completed response with no content",
                    code=ErrorCode.EMPTY_RESPONSE.value,
                ),
            )
        return reason


def _is_complete_sse_line(text: str) -> bool:
    """残留内容是否已经是一个完整、可解析的 SSE 行（用于行式 transport 的兼容判断）。"""
    stripped = text.strip()
    if not stripped.startswith("data:"):
        return False
    payload = stripped[len("data:") :].strip()
    if payload == "[DONE]":
        return True
    try:
        json.loads(payload)
    except ValueError:
        return False
    return True


async def _sse_lines(chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    """把 transport 产出的文本块切成行。

    契约：**transport 产出的是文本块**，块边界不隐含行边界 —— 一个合法块也可能
    刚好切在 JSON 对象中间，因此只以 `\\n` 作为行分隔符，未终止的尾行留在 buffer 里。

    兼容：有些 transport 每次 yield 一整行但去掉了行尾换行符（httpx
    `aiter_lines()` 的历史用法）。这时靠「下一个块以 `data:` 开头」+「残留本身已经是一个
    完整可解析的 SSE 行」两个条件才判定上一块是完整行 —— 只满足前者不足以切分，
    否则 `metadata: PWM` 这类正文里含 `data:` 的内容会被误切成残缺 JSON。
    默认 transport 已把行补回 `\\n`，不走这条兼容分支。
    """
    buffer = ""
    pending_partial = False

    async for chunk in chunks:
        if pending_partial and chunk.startswith("data:") and _is_complete_sse_line(buffer):
            # 上一块是行式 transport 的完整行：先把它作为一行交出去。
            yield buffer
            buffer = ""
        buffer += chunk
        pending_partial = False

        if "\n" in buffer:
            parts = buffer.split("\n")
            buffer = parts.pop()
            for line in parts:
                yield line
        if not chunk.endswith("\n") and buffer:
            pending_partial = True

    if buffer:
        yield buffer


async def chunks_from_sse(
    chunks: AsyncIterator[str],
    cancelled: Callable[[], bool] | None = None,
) -> AsyncIterator[StreamChunk]:
    """把 SSE 文本翻译成 `StreamChunk`（DSH `translate.ts::translate`）。

    `block-end`、`usage`、`finish` 都延迟到 `[DONE]`：finish_reason 之后可能还有独立
    的 usage-only trailer 块，提前收尾会丢掉它。流结束却没有 `[DONE]` 时抛
    `STREAM_CLOSED`（DSH `sse.ts` 的 "SSE stream ended without [DONE]"）。

    没有 `finish_reason` **不是**错误：DSH 是 `pendingFinish ?? {kind: 'stop'}`。

    退出时（正常结束、`[DONE]` 后 break、取消、异常）都要把**上游 transport** 关掉：
    只关自己时，真实 HTTP 响应会一直挂到 GC 才释放，表现为
    `aclose(): asynchronous generator is already running`。
    """
    translator = _SseTranslator()
    saw_done = False
    lines = _sse_lines(chunks)
    try:
        async for line in lines:
            if cancelled is not None and cancelled():
                raise ProviderError("model request cancelled", ErrorCode.ABORTED)
            line = line.strip()
            if line == "" or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                saw_done = True
                break
            try:
                payload = json.loads(data)
            except ValueError as exc:
                raise MalformedCompletion(
                    "malformed completion: SSE payload is not JSON"
                ) from exc
            if not isinstance(payload, dict):
                raise MalformedCompletion("malformed completion: SSE payload is not an object")
            for chunk in translator.consume(payload):
                yield chunk

        if not saw_done:
            raise ProviderError("SSE stream ended without [DONE]", ErrorCode.STREAM_CLOSED)

        for chunk in translator.close():
            yield chunk
    finally:
        # 先关行包装、再关 transport：两层都不能漏。
        await _aclose_stream(lines)
        await _aclose_stream(chunks)


async def _aclose_stream(stream: Any) -> None:
    """尽力关闭一个异步流（可能已被关过、也可能不是 generator）。"""
    close = getattr(stream, "aclose", None)
    if close is None:
        return
    with contextlib.suppress(BaseException):
        await close()


def _map_finish_reason(reason: str) -> FinishReason:
    """DSH `mapFinishReason`：未知原因归入 error，不能当成正常 stop。"""
    if reason == "stop":
        return FinishReason(kind="stop")
    if reason == "tool_calls":
        return FinishReason(kind="tool-calls")
    if reason == "length":
        return FinishReason(kind="max-tokens")
    return FinishReason(
        kind="error",
        failure=LlmFailure(message=f"model stopped: {reason}", code=reason.upper()),
    )


def _parse_usage(raw: Any) -> Usage | None:
    if not isinstance(raw, dict):
        return None
    input_tokens = int(raw.get("prompt_tokens") or 0)
    output_tokens = int(raw.get("completion_tokens") or 0)
    total = raw.get("total_tokens")
    prompt_details = raw.get("prompt_tokens_details") or {}
    completion_details = raw.get("completion_tokens_details") or {}
    cached = raw.get("prompt_cache_hit_tokens", prompt_details.get("cached_tokens"))
    missed = raw.get("prompt_cache_miss_tokens")
    reasoning = completion_details.get("reasoning_tokens")
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=int(total) if total is not None else input_tokens + output_tokens,
        cached_input_tokens=int(cached) if cached is not None else None,
        uncached_input_tokens=(int(missed) if missed is not None else
                               input_tokens - int(cached) if cached is not None else None),
        reasoning_tokens=int(reasoning) if reasoning is not None else None,
    )


def to_wire_message(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    # DSH llm-deepseek/serialize.ts::serializeAssistant: replay reasoning verbatim,
    # including assistant turns without tool calls; never merge it into content.
    if message.role == "assistant" and message.reasoning_content:
        payload["reasoning_content"] = message.reasoning_content
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in message.tool_calls
        ]
    if message.role == "tool":
        if message.tool_call_id is not None:
            payload["tool_call_id"] = message.tool_call_id
        if message.name is not None:
            payload["name"] = message.name
    return payload


def to_wire_tool(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }
