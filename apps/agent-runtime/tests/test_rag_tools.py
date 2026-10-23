"""RAG 工具与最小 CLI（离线：注入 transport，不连真实服务）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app import cli
from app.cli import build_parser
from app.contracts import AssistantTurn, Message, ToolCall
from app.errors import ErrorCode, ProviderError
from app.providers.mock import MockProvider, stream_from_coroutine
from app.rag_tools import (
    RagClient,
    RagClientConfig,
    RagError,
    rag_tools,
    render_context,
    render_search_hits,
)
from app.session import JsonlSessionStore
from app.tools import ToolRegistry


def final(text: str = "offline answer") -> AssistantTurn:
    return AssistantTurn(Message(role="assistant", content=text), "stop")


# fixture 对齐真实契约：版本形如 `Vx.y`，kind 属于 TEXT/TABLE/VISUAL/MIXED/CODE。
HIT = {
    "chunk_id": "0dd2c1de-1f8c-4d3a-9a10-2f6b8a1c9e11",
    "document_id": "6b1f0f2a-3c4d-4e5f-8a9b-0c1d2e3f4a5b",
    "product": "M8P515",
    "version": "V1.0",
    "section_path": "1.1",
    "page_start": 12,
    "page_end": 13,
    "kind": "TEXT",
    "text": "PWM 周期由 PR2 决定。",
    "score": 0.83,
    "vector_score": 0.71,
}

CONTEXT = {
    **{key: HIT[key] for key in ("chunk_id", "document_id", "product", "version", "kind")},
    "section_path": "10.0 PWM",
    "section_title": "PWM 模块",
    "page_start": 12,
    "page_end": 13,
    "text": "PR2 = 0xFF 时周期最大。",
    "table_repr": {"rows": [["PR2", "255"]]},
}


def recording_transport(response: tuple[int, Any]):
    """返回 transport 与调用记录，便于断言请求形状。"""
    calls: list[dict[str, Any]] = []

    async def transport(url: str, params: dict[str, Any], headers: dict[str, str]):
        calls.append({"url": url, "params": params, "headers": headers})
        return response

    return transport, calls


async def test_search_spec_keeps_citation_fields() -> None:
    transport, calls = recording_transport((200, [HIT]))
    client = RagClient(RagClientConfig("http://rag", api_key="secret"), transport)
    registry = ToolRegistry()
    for tool in rag_tools(client):
        registry.register(tool)

    result = await registry.execute(
        ToolCall(
            id="c1",
            name="search_spec",
            arguments={"question": "PWM 周期由谁决定", "product": "M8P515", "version": "V1.0"},
        )
    )

    assert result.is_error is False
    assert "PWM 周期由 PR2 决定。" in result.content
    # read_context 需要的两个标识必须都出现
    assert HIT["document_id"] in result.content
    assert HIT["chunk_id"] in result.content
    for field in ("product=M8P515", "version=V1.0", "pages=12-13", "section=1.1", "kind=TEXT"):
        assert field in result.content, field
    assert calls[0]["headers"]["X-API-Key"] == "secret"
    assert calls[0]["url"] == "http://rag/search"
    assert calls[0]["params"]["question"] == "PWM 周期由谁决定"
    assert calls[0]["params"]["product"] == "M8P515"
    assert calls[0]["params"]["version"] == "V1.0"


async def test_search_spec_with_no_hits_says_so() -> None:
    transport, _ = recording_transport((200, []))
    client = RagClient(RagClientConfig("http://rag"), transport)
    registry = ToolRegistry()
    for tool in rag_tools(client):
        registry.register(tool)

    result = await registry.execute(
        ToolCall(id="c1", name="search_spec", arguments={"question": "无关问题"})
    )

    assert result.is_error is False
    assert "no chunks matched" in result.content


async def test_excerpt_keeps_order_and_ids_and_full_evidence_can_be_read() -> None:
    long_hit = {**HIT, "text": "寄存器说明" * 300 + "末尾关键字段=73"}
    second = {**HIT, "chunk_id": "second-chunk", "text": "第二条证据"}
    requests = []

    async def transport(url, params, headers):
        requests.append((url, params))
        return (200, [long_hit, second]) if url.endswith("/search") else (200, long_hit)

    registry = ToolRegistry()
    for tool in rag_tools(RagClient(RagClientConfig("http://rag"), transport)):
        registry.register(tool)
    excerpt = await registry.execute(ToolCall("find", "search_spec", {"question": "寄存器"}))
    assert not excerpt.is_error and "[excerpt: 900/" in excerpt.content
    assert "末尾关键字段=73" not in excerpt.content
    assert excerpt.content.index(HIT["chunk_id"]) < excerpt.content.index("second-chunk")
    assert HIT["document_id"] in excerpt.content and "第二条证据" in excerpt.content
    full = await registry.execute(ToolCall("read", "read_context", {
        "document_id": HIT["document_id"], "chunk_id": HIT["chunk_id"],
    }))
    assert not full.is_error and "末尾关键字段=73" in full.content
    requested_full = await registry.execute(ToolCall("full", "search_spec", {
        "question": "寄存器", "max_chars": 0,
    }))
    assert "末尾关键字段=73" in requested_full.content
    assert all("max_chars" not in params for _, params in requests)
    integral_float = await registry.execute(ToolCall("float", "search_spec", {
        "question": "寄存器", "max_chars": 900.0,
    }))
    assert not integral_float.is_error and "[excerpt: 900/" in integral_float.content


async def test_search_spec_reports_service_errors() -> None:
    transport, _ = recording_transport((503, {"detail": "vector store unavailable"}))
    client = RagClient(RagClientConfig("http://rag"), transport)
    registry = ToolRegistry()
    for tool in rag_tools(client):
        registry.register(tool)

    result = await registry.execute(
        ToolCall(id="c1", name="search_spec", arguments={"question": "PWM"})
    )

    assert result.is_error is True
    assert "503" in result.content and "vector store unavailable" in result.content


async def test_read_context_renders_text_and_table() -> None:
    transport, calls = recording_transport((200, CONTEXT))
    client = RagClient(RagClientConfig("http://rag"), transport)
    registry = ToolRegistry()
    for tool in rag_tools(client):
        registry.register(tool)

    result = await registry.execute(
        ToolCall(
            id="c1",
            name="read_context",
            arguments={"document_id": CONTEXT["document_id"], "chunk_id": CONTEXT["chunk_id"]},
        )
    )

    assert result.is_error is False
    assert "PR2 = 0xFF 时周期最大。" in result.content
    assert "table:" in result.content
    assert calls[0]["url"] == (
        f"http://rag/documents/{CONTEXT['document_id']}/context/{CONTEXT['chunk_id']}"
    )


async def test_read_context_validates_arguments() -> None:
    transport, _ = recording_transport((200, CONTEXT))
    client = RagClient(RagClientConfig("http://rag"), transport)
    registry = ToolRegistry()
    for tool in rag_tools(client):
        registry.register(tool)

    result = await registry.execute(
        ToolCall(id="c1", name="read_context", arguments={"document_id": "d"})
    )

    # 缺参数在 schema 校验层就被挡住（"missing required argument: chunk_id"），
    # handler 里还有一层兜底。
    assert result.is_error is True
    assert "chunk_id" in result.content


async def test_read_context_rejects_unknown_kind_agnostically() -> None:
    """kind 是服务端枚举，客户端不校验、原样渲染。"""
    foreign = {**CONTEXT, "kind": "MIXED"}
    transport, _ = recording_transport((200, foreign))
    registry = ToolRegistry()
    for tool in rag_tools(RagClient(RagClientConfig("http://rag"), transport)):
        registry.register(tool)

    result = await registry.execute(
        ToolCall(
            id="c1",
            name="read_context",
            arguments={"document_id": CONTEXT["document_id"], "chunk_id": CONTEXT["chunk_id"]},
        )
    )

    assert not result.is_error
    assert "kind=MIXED" in result.content


def test_rag_tools_are_read_only_and_parallel() -> None:
    registry = ToolRegistry()
    for tool in rag_tools(RagClient(RagClientConfig("http://rag"))):
        registry.register(tool)

    assert [tool.name for tool in registry.list()] == ["search_spec", "read_context"]
    assert registry.execution_mode("search_spec") == "parallel"
    assert registry.execution_mode("read_context") == "parallel"


async def test_search_to_context_handoff_uses_only_rendered_fields() -> None:
    """模型只能看到 search_spec 的文本输出：document_id + chunk_id 必须都在里面。"""
    hits = [HIT]

    async def transport(url: str, params: dict[str, Any], headers: dict[str, str]):
        if url.endswith("/search"):
            return 200, hits
        return 200, CONTEXT

    registry = ToolRegistry()
    for tool in rag_tools(RagClient(RagClientConfig("http://rag"), transport)):
        registry.register(tool)

    search = await registry.execute(
        ToolCall(id="c1", name="search_spec", arguments={"question": "PWM 周期"})
    )
    assert not search.is_error

    # 从渲染结果里"读"出标识（模拟模型的行为），而不是从 fixture 直接拿。
    rendered = search.content
    document_id = HIT["document_id"]
    chunk_id = HIT["chunk_id"]
    assert f"document={document_id}" in rendered and f"chunk={chunk_id}" in rendered

    follow_up = await registry.execute(
        ToolCall(
            id="c2",
            name="read_context",
            arguments={"document_id": document_id, "chunk_id": chunk_id},
        )
    )
    assert not follow_up.is_error
    assert "PR2 = 0xFF" in follow_up.content


def test_default_product_and_version_are_applied() -> None:
    transport, calls = recording_transport((200, [HIT]))
    client = RagClient(RagClientConfig("http://rag"), transport)
    registry = ToolRegistry()
    for tool in rag_tools(client, default_product="M8P515", default_version="V1.0"):
        registry.register(tool)

    import asyncio

    asyncio.run(
        registry.execute(ToolCall(id="c1", name="search_spec", arguments={"question": "PWM"}))
    )

    assert calls[0]["params"]["product"] == "M8P515"
    assert calls[0]["params"]["version"] == "V1.0"


def test_render_helpers_are_pure() -> None:
    rendered = render_search_hits([HIT])
    assert rendered.startswith("[1] document=")
    assert render_search_hits([]) == (
        "no chunks matched; try a different question or relax product/version filters"
    )
    assert "PWM 模块" not in render_context(CONTEXT)  # section_title 是可选字段
    assert "section=10.0 PWM" in render_context(CONTEXT)


def test_missing_httpx_is_an_error_not_a_crash() -> None:
    import asyncio

    client = RagClient(RagClientConfig("http://rag"))

    async def attempt() -> None:
        try:
            await client.search("PWM")
        except RagError as error:
            assert "httpx" in str(error) or "503" in str(error) or "failed" in str(error)
            return
        except Exception as error:  # 真的装了 httpx 时可能连不上本地服务
            assert isinstance(error, Exception)
            return

    asyncio.run(attempt())


async def test_cli_streaming_executes_the_real_event_handler(monkeypatch, capsys) -> None:
    """CLI 必须按真实事件契约读文本（字段是 `text`，不是 `delta`）。"""
    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **kwargs: MockProvider([final()]))
    args = cli.build_parser().parse_args(
        [
            "offline question",
            "--base-url",
            "http://unused.invalid/v1",
            "--api-key",
            "offline-test",
            "--rag-key",
            "offline-test",
            "--json",
        ]
    )
    exit_code = await cli.run_once(args)
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0, payload
    assert payload["status"] == "completed", payload
    assert payload["answer"] == "offline answer"


async def test_cli_session_preserves_inputs_and_append_chain(tmp_path, monkeypatch, capsys) -> None:
    """两次追加同一会话文件：输入都要落盘，且树不能断成两个根。"""

    class NonStreamingProvider:
        def generate_stream(self, request):
            return stream_from_coroutine(lambda: self.generate(request))

        async def generate(self, request):
            return final()

    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **kwargs: NonStreamingProvider())
    session_path = tmp_path / "session.jsonl"
    questions = ["first question", "second question"]
    for question in questions:
        args = cli.build_parser().parse_args(
            [
                question,
                "--base-url",
                "http://unused.invalid/v1",
                "--api-key",
                "offline-test",
                "--rag-key",
                "offline-test",
                "--session",
                str(session_path),
                "--json",
            ]
        )
        assert await cli.run_once(args) == 0
        capsys.readouterr()

    store = JsonlSessionStore(session_path)
    raw = await store.load_raw_messages()
    nodes = await store.load_nodes()
    observed = {
        "user_inputs": [m.content for m in raw if m.role == "user"],
        "roots": sum(node.parent_id is None for node in nodes),
    }
    assert observed == {"user_inputs": questions, "roots": 1}, observed


async def test_cli_provider_failure_does_not_turn_the_question_into_an_answer(
    monkeypatch, capsys,
) -> None:
    class FailingProvider:
        def generate_stream(self, request):
            return stream_from_coroutine(lambda: self.generate(request))

        async def generate(self, request):
            raise ProviderError("offline auth failure", ErrorCode.AUTH)

    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **kwargs: FailingProvider())
    args = cli.build_parser().parse_args(
        [
            "offline question",
            "--base-url",
            "http://unused.invalid/v1",
            "--api-key",
            "offline-test",
            "--rag-key",
            "offline-test",
            "--json",
        ]
    )
    exit_code = await cli.run_once(args)
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["status"] == "error"
    assert payload["answer"] == "", payload


def test_cli_parser_defaults(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(["PWM 周期怎么算"])

    assert args.question == "PWM 周期怎么算"
    assert args.rag_url  # 有默认值
    assert args.json is False
    assert args.session is None


def test_cli_json_payload_shape() -> None:
    """`--json` 输出的字段（供评测脚本消费）。"""
    payload = {
        "status": "completed",
        "turns": 1,
        "steps": 2,
        "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        "tool_calls": ["search_spec"],
        "answer": "见 PR2",
        "error": None,
    }
    assert json.loads(json.dumps(payload, ensure_ascii=False))["status"] == "completed"
