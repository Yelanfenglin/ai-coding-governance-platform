"""RAG 工具：把 `apps/rag-api` 的检索能力接成 Agent 工具。

- `search_spec`  -> `GET /search`
- `read_context` -> `GET /documents/{doc_id}/context/{chunk_id}`

引用必须保留：型号（product）/ 版本（version）/ chunk_id / 页码 / section_path，
否则回答无法追溯。两个工具都是只读，标 `parallel`（可与其它只读工具并发）。

httpx 是可选依赖：没装时默认 transport 会给出明确报错；测试与离线运行可以注入
`RagTransport`。
"""

from __future__ import annotations

from collections.abc import Awaitable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlencode, urlsplit

from .contracts import ToolResult
from .tools import FunctionTool

#: `GET /search` 的返回字段（`apps/rag-api/app/schemas.py::SearchHit`）。
SearchHit = dict[str, Any]
#: `GET /documents/{doc_id}/context/{chunk_id}` 的返回字段（`ChunkContext`）。
ChunkContext = dict[str, Any]


class RagError(RuntimeError):
    """RAG 调用失败（网络 / 404 / 422 / 5xx / 鉴权）。"""


@dataclass(frozen=True)
class RagClientConfig:
    base_url: str
    api_key: str | None = None
    timeout: float = 20.0

    @property
    def search_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/search"

    def context_url(self, document_id: str, chunk_id: str) -> str:
        return f"{self.base_url.rstrip('/')}/documents/{document_id}/context/{chunk_id}"


class RagTransport(Protocol):
    """一次 HTTP 调用：`(url, params) -> (status_code, payload)`。"""

    def __call__(
        self, url: str, params: dict[str, Any], headers: dict[str, str]
    ) -> Awaitable[tuple[int, Any]]: ...


class RagClient:
    """rag-api 的最小客户端（只覆盖 Agent 需要的两个端点）。"""

    def __init__(
        self, config: RagClientConfig, transport: RagTransport | None = None
    ) -> None:
        self.config = config
        self._transport = transport

    async def search(
        self,
        question: str,
        *,
        product: str | None = None,
        version: str | None = None,
        kind: str | None = None,
        top_k: int = 8,
    ) -> list[SearchHit]:
        params: dict[str, Any] = {"question": question, "top_k": top_k}
        if product:
            params["product"] = product
        if version:
            params["version"] = version
        if kind:
            params["kind"] = kind
        status, payload = await self._request(self.config.search_url, params)
        if status >= 400:
            raise RagError(f"search failed: {status} {_detail(payload)}")
        if not isinstance(payload, list):
            raise RagError("search failed: unexpected payload shape")
        return [hit for hit in payload if isinstance(hit, dict)]

    async def context(self, document_id: str, chunk_id: str) -> ChunkContext:
        status, payload = await self._request(
            self.config.context_url(document_id, chunk_id), {}
        )
        if status >= 400:
            raise RagError(f"read_context failed: {status} {_detail(payload)}")
        if not isinstance(payload, dict):
            raise RagError("read_context failed: unexpected payload shape")
        return payload

    async def _request(self, url: str, params: dict[str, Any]) -> tuple[int, Any]:
        headers = {"Accept": "application/json"}
        if self.config.api_key:
            headers["X-API-Key"] = self.config.api_key
        if self._transport is not None:
            return await self._transport(url, params, headers)
        return await self._httpx_request(url, params, headers)

    async def _httpx_request(
        self, url: str, params: dict[str, Any], headers: dict[str, str]
    ) -> tuple[int, Any]:
        try:
            import httpx  # 延迟导入：httpx 是可选依赖
        except ImportError as exc:
            raise RagError("httpx is required for the default RAG transport") from exc

        import json as _json

        query = urlencode({key: value for key, value in params.items()})
        trust_env = urlsplit(url).hostname not in {"127.0.0.1", "localhost", "::1"}
        async with httpx.AsyncClient(timeout=self.config.timeout, trust_env=trust_env) as client:
            response = await client.get(f"{url}?{query}", headers=headers)
        try:
            payload = response.json()
        except ValueError:
            payload = response.text
        if isinstance(payload, str):
            with suppress(ValueError):
                payload = _json.loads(payload)
        return response.status_code, payload


# --- 渲染：引用字段一律保留 ---------------------------------------------------


def render_search_hit(hit: SearchHit, index: int = 1, *, max_chars: int = 0) -> str:
    """单条命中（供测试与复用）。"""
    lines = [f"[{index}] {_citation_line(hit)}"]
    text = str(hit.get("text", "")).strip()
    if text:
        if max_chars and len(text) > max_chars:
            lines.append(text[:max_chars])
            lines.append(
                f"[excerpt: {max_chars}/{len(text)} chars; use read_context with the above "
                "document_id/chunk_id to read the complete evidence before using its details]"
            )
        else:
            lines.append(text)
    return "\n".join(lines)


def render_search_hits(hits: list[SearchHit], *, max_chars: int = 0) -> str:
    """把检索结果渲染成带引用的文本；没有命中时明确说没有。"""
    if not hits:
        return "no chunks matched; try a different question or relax product/version filters"

    return "\n\n".join(
        render_search_hit(hit, index, max_chars=max_chars) for index, hit in enumerate(hits, 1)
    )


def render_context(context: ChunkContext) -> str:
    """渲染一段上下文原文（含表格时附上表格表示）。"""
    lines = [_citation_line(context)]
    text = str(context.get("text", "")).strip()
    if text:
        lines.append(text)
    table = context.get("table_repr")
    if table:
        lines.append(f"table: {table}")
    image = context.get("image_ref")
    if image:
        lines.append(f"image: {image}")
    return "\n".join(lines)


def _citation_line(hit: dict[str, Any]) -> str:
    """引用行：document_id / chunk_id / 型号 / 版本 / 页码 / 章节路径 —— 缺一不可追溯。

    `document_id` 必须出现：`read_context` 需要 `document_id` + `chunk_id` 两个参数，
    模型只能从这段文本里拿到它们。
    """
    parts = [
        f"document={hit.get('document_id', '?')}",
        f"chunk={hit.get('chunk_id', '?')}",
        f"product={hit.get('product', '?')}",
        f"version={hit.get('version', '?')}",
        f"pages={hit.get('page_start', '?')}-{hit.get('page_end', '?')}",
    ]
    section = hit.get("section_path")
    if section:
        parts.append(f"section={section}")
    kind = hit.get("kind")
    if kind:
        parts.append(f"kind={kind}")
    score = hit.get("score")
    if score is not None:
        parts.append(f"score={float(score):.3f}")
    return " ".join(parts)


def _detail(payload: Any) -> str:
    if isinstance(payload, dict):
        return str(payload.get("detail", payload))
    return str(payload)


# --- 工具 -------------------------------------------------------------------


def search_spec_tool(
    client: RagClient,
    *,
    default_product: str | None = None,
    default_version: str | None = None,
) -> FunctionTool:
    async def execute(args: dict[str, Any], context: Any) -> ToolResult:
        question = str(args.get("question", "")).strip()
        if not question:
            return ToolResult(content="search_spec requires 'question'", is_error=True)
        try:
            hits = await client.search(
                question,
                product=str(args.get("product") or default_product or "") or None,
                version=str(args.get("version") or default_version or "") or None,
                kind=str(args.get("kind") or "") or None,
                top_k=int(args.get("top_k", 8)),  # type: ignore[arg-type]
            )
        except RagError as error:
            return ToolResult(content=f"search_spec failed: {error}", is_error=True)
        return ToolResult(content=render_search_hits(
            hits, max_chars=int(args.get("max_chars", 900)),
        ))

    return FunctionTool(
        name="search_spec",
        description=(
            "Locate specification evidence for a concrete missing fact or register. Returns "
            "cited excerpts (default 900 chars per hit); use read_context for selected full chunks "
            "instead of repeating broad searches. Citation identifiers and hit order are preserved."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "product": {"type": "string"},
                "version": {"type": "string"},
                "kind": {"type": "string"},
                "top_k": {"type": "integer"},
                "max_chars": {"type": "integer", "minimum": 0, "maximum": 8000,
                              "description": "Excerpt chars per hit; 0 returns full text"},
            },
            "required": ["question"],
        },
        handler=execute,
        execution_mode="parallel",  # 只读
    )


def read_context_tool(client: RagClient) -> FunctionTool:
    async def execute(args: dict[str, Any], context: Any) -> ToolResult:
        document_id = str(args.get("document_id", "")).strip()
        chunk_id = str(args.get("chunk_id", "")).strip()
        if not document_id or not chunk_id:
            return ToolResult(
                content="read_context requires 'document_id' and 'chunk_id'", is_error=True
            )
        try:
            payload = await client.context(document_id, chunk_id)
        except RagError as error:
            return ToolResult(content=f"read_context failed: {error}", is_error=True)
        return ToolResult(content=render_context(payload))

    return FunctionTool(
        name="read_context",
        description=(
            "Read a RAG specification chunk using document_id and chunk_id UUIDs returned "
            "by search_spec. This cannot read memory entries or session names; "
            "current durable facts are already in the memory snapshot."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "document_id": {"type": "string"},
                "chunk_id": {"type": "string"},
            },
            "required": ["document_id", "chunk_id"],
        },
        handler=execute,
        execution_mode="parallel",  # 只读
    )


def rag_tools(
    client: RagClient,
    *,
    default_product: str | None = None,
    default_version: str | None = None,
) -> list[FunctionTool]:
    """一套 RAG 工具：`search_spec` + `read_context`。"""
    return [
        search_spec_tool(
            client, default_product=default_product, default_version=default_version
        ),
        read_context_tool(client),
    ]
