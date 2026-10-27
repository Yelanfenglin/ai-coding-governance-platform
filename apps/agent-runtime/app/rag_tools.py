"""RAG 工具：把 `apps/rag-api` 的检索能力接成 Agent 工具。

- `search_spec`  -> `GET /search`
- `read_context` -> `GET /documents/{doc_id}/context/{chunk_id}`

引用必须保留：型号（product）/ 版本（version）/ chunk_id / 页码 / section_path，
否则回答无法追溯。两个工具都是只读，标 `parallel`（可与其它只读工具并发）。

httpx 是可选依赖：没装时默认 transport 会给出明确报错；测试与离线运行可以注入
`RagTransport`。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode, urlsplit

from opentelemetry.propagate import inject

from .circuit import (
    CircuitBreakers,
    CircuitOpen,
    Outcome,
    Permit,
    rag_dependency,
    rag_outcome,
    transport_failure,
)
from .contracts import ToolResult
from .observability import failure, observe
from .spec_evidence import SpecEvidence, SpecRelationGraph
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
        self,
        config: RagClientConfig,
        transport: RagTransport | None = None,
        breakers: CircuitBreakers | None = None,
    ) -> None:
        self.config = config
        self._transport = transport
        self._breakers = breakers

    async def search(
        self,
        question: str,
        *,
        product: str | None = None,
        version: str | None = None,
        kind: str | None = None,
        top_k: int = 8,
        rerank: bool = False,
        candidate_k: int | None = None,
    ) -> list[SearchHit]:
        params: dict[str, Any] = {"question": question, "top_k": top_k}
        if product:
            params["product"] = product
        if version:
            params["version"] = version
        if kind:
            params["kind"] = kind
        if rerank:
            params["rerank"] = True
        if candidate_k is not None:
            params["candidate_k"] = candidate_k
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
        operation = "search" if url == self.config.search_url else "context"
        # 两个操作分键：search 依赖 embedding/向量检索，context 只读原始片段。
        # embedding 挂掉不该挡住“读取已有证据”，所以它们不共用一套状态。
        dependency = rag_dependency(operation, self.config.base_url)
        permit = self._acquire(dependency)
        try:
            with observe(
                f"rag.client.{operation}", **{"http.request.method": "GET"}
            ) as span:
                headers = {"Accept": "application/json"}
                inject(headers)
                if self.config.api_key:
                    headers["X-API-Key"] = self.config.api_key
                try:
                    status, payload = (
                        await self._transport(url, params, headers)
                        if self._transport is not None
                        else await self._httpx_request(url, params, headers)
                    )
                except asyncio.CancelledError:
                    self._record(permit, Outcome.CANCELLED)
                    raise
                except Exception as exc:
                    # 只把明确的网络/超时算作依赖故障；httpx 缺装这类本地问题不算。
                    self._record(
                        permit, Outcome.FAILURE if transport_failure(exc) else Outcome.IGNORED,
                    )
                    raise
                # 500 在这里是正常返回的 (status, payload)，异常在外层才构造：
                # 所以必须按状态码记账，否则 500 会被记成成功。
                self._record(permit, rag_outcome(status))
                span.set_attribute("http.response.status_code", status)
                if status >= 400:
                    failure(span, f"http_{status}")
                return status, payload
        finally:
            if self._breakers is not None and permit is not None:
                self._breakers.release(permit)

    def _acquire(self, dependency: str) -> Permit | None:
        if self._breakers is None:
            return None
        try:
            return self._breakers.acquire(dependency)
        except CircuitOpen as open_error:
            # 工具返回错误结果，模型可以继续用本地能力并说明“检索不可用”。
            raise RagError(str(open_error)) from open_error

    def _record(self, permit: Permit | None, outcome: Outcome) -> None:
        if self._breakers is not None:
            self._breakers.record(permit, outcome)

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


def render_search_hits(
    hits: list[SearchHit], *, max_chars: int = 0, rerank_requested: bool = False,
) -> str:
    """把检索结果渲染成带引用的文本；没有命中时明确说没有。"""
    if not hits:
        return "no chunks matched; try a different question or relax product/version filters"

    rendered = "\n\n".join(
        render_search_hit(hit, index, max_chars=max_chars) for index, hit in enumerate(hits, 1)
    )
    if rerank_requested and any(hit.get("rerank_score") is None for hit in hits):
        return (
            "[reranker unavailable; retrieval-order fallback, not reranked evidence]\n" + rendered
        )
    return rendered


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
    if hit.get("rerank_score") is not None:
        parts.append(f"rerank_score={float(hit['rerank_score']):.3f}")
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
    evidence: SpecEvidence | None = None,
) -> FunctionTool:
    async def execute(args: dict[str, Any], context: Any) -> ToolResult:
        question = str(args.get("question", "")).strip()
        if not question:
            return ToolResult(content="search_spec requires 'question'", is_error=True)
        top_k = int(args.get("top_k", 8))
        rerank = bool(args.get("rerank", False))
        candidate_k = int(args["candidate_k"]) if "candidate_k" in args else None
        if candidate_k is not None and (not rerank or candidate_k < top_k):
            return ToolResult(
                content="candidate_k requires rerank=true and must be at least top_k",
                is_error=True,
            )
        product = str(args.get("product") or default_product or "") or None
        version = str(args.get("version") or "") or None
        if version is None and product == default_product:
            version = default_version
        try:
            hits = await client.search(
                question,
                product=product,
                version=version,
                kind=str(args.get("kind") or "") or None,
                top_k=top_k, rerank=rerank, candidate_k=candidate_k,
            )
        except RagError as error:
            return ToolResult(content=f"search_spec failed: {error}", is_error=True)
        max_chars = int(args.get("max_chars", 0))
        if evidence is not None:
            for hit in hits:
                text = str(hit.get("text", "")).strip()
                evidence.record(hit, text[:max_chars] if max_chars else text)
        return ToolResult(content=render_search_hits(
            hits, max_chars=max_chars, rerank_requested=rerank,
        ))

    return FunctionTool(
        name="search_spec",
        description=(
            "Locate specification evidence for a concrete missing fact or register. Returns "
            "complete cited chunk text by default. Set max_chars explicitly only when excerpts "
            "are wanted; read_context expands selected original contexts without searching again. "
            "Citation IDs and hit order are preserved. "
            "For competing results, optionally set rerank=true to rank a wider candidate pool "
            "before returning top_k excerpts (default pool: max(top_k,30), within server limits). "
            "This adds "
            "a reranker request and requires server support; do not enable for every lookup. "
            f"Default scope: product={default_product!r}, version={default_version!r}. "
            "Overriding product does not inherit the old product's version; specify both "
            "product and version for a version-specific cross-product comparison."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "product": {"type": "string"},
                "version": {"type": "string"},
                "kind": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
                "rerank": {"type": "boolean"},
                "candidate_k": {"type": "integer", "minimum": 1, "maximum": 50,
                                "description": "Recall pool for rerank; must be >= top_k"},
                "max_chars": {"type": "integer", "minimum": 0, "maximum": 8000,
                              "description": "Excerpt chars per hit; default 0 returns full text"},
            },
            "required": ["question"],
        },
        handler=execute,
        execution_mode="parallel",  # 只读
    )


def read_context_tool(client: RagClient, *, evidence: SpecEvidence | None = None) -> FunctionTool:
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
        rendered = render_context(payload)
        if evidence is not None:
            evidence.record(payload, rendered.partition("\n")[2])
        return ToolResult(content=rendered)

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
    relations: SpecRelationGraph | None = None,
    workspace: Path | None = None,
) -> list[FunctionTool]:
    """RAG tools; the optional graph/checklist observes this agent's own evidence."""
    evidence = SpecEvidence(relations) if relations is not None else None
    tools = [
        search_spec_tool(
            client, default_product=default_product, default_version=default_version,
            evidence=evidence,
        ),
        read_context_tool(client, evidence=evidence),
    ]
    if evidence is not None:
        async def read_source(document_id: str, chunk_id: str):
            payload = await client.context(document_id, chunk_id)
            return payload, render_context(payload).partition("\n")[2]

        source_reader = None
        if workspace is not None:
            from .register_facts import read_register_facts

            async def source_reader(path, function, graph):
                return await asyncio.to_thread(
                    read_register_facts, workspace, path, function, graph.fields,
                )

        tools.append(evidence.tool(read_source, source_reader))
    return tools
