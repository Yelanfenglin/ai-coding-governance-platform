"""LLM structured query planner (SelfQuery-style), evaluated before production use.

The planner turns an ambiguous/compound question into a small JSON decision:

    {intent, product, version, strategy, needs_clarification, rewritten_query}

It is intentionally **not** on the hot path.  The deterministic ``build_plan`` in
``app.retrieval`` stays the default; a caller decides when to spend one LLM call
(missing scope, multiple sub-questions, comparison, or low first-pass confidence).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx

from ..query_filters import FilterCondition, parse_filter_conditions

Intent = Literal["lookup", "comparison", "multi_evidence", "clarify", "out_of_scope"]
Strategy = Literal["dense", "metadata_filter", "multi_document", "clarify"]
INTENTS: tuple[str, ...] = ("lookup", "comparison", "multi_evidence", "clarify", "out_of_scope")
STRATEGIES: tuple[str, ...] = ("dense", "metadata_filter", "multi_document", "clarify")

PLANNER_SYSTEM_PROMPT = """你是 MCU 规格书 RAG 的查询规划器。只输出一个 JSON 对象，不要输出解释、
Markdown 或代码围栏。字段固定为：
{"intent": "lookup|comparison|multi_evidence|clarify|out_of_scope",
 "product": 字符串或 null, "version": 字符串或 null,
 "strategy": "dense|metadata_filter|multi_document|clarify",
 "needs_clarification": true/false,
 "rewritten_query": 字符串,
 "queries": ["可独立检索的改写1", "改写2"],
 "filters": [{"field": "product|version|register|section_id|package", "operator": "eq|in", "value": 字符串或字符串数组}],
 "confidence": 0 到 1 的小数,
 "reasons": 字符串数组}
规则：
- 问题明确问单一型号的单个事实 → lookup + dense；
- 问题要求比较两个型号/两个来源，或多个独立条件 → comparison/multi_evidence；
- 问题没有型号、或型号/版本无法判断，且不能从问题中确定 → clarify + needs_clarification=true；
- 问题超出规格书范围 → out_of_scope + clarify；
- rewritten_query 必须是可独立检索的完整问题，保留型号、版本、寄存器/位/引脚/数值原样；
- queries 给 1 到 3 条可独立检索的查询，第一条尽量保留原问题，其余只做必要的同义改写/子问题拆分；
- filters 只在问题里明确出现时才填，值必须是原文中的型号/版本/寄存器/章节；不明确就不填；
- 不确定时降低 confidence，不要编造型号或版本。"""  # noqa: E501


class PlannerError(RuntimeError):
    """Planner call or output failed validation."""


@dataclass(frozen=True, slots=True)
class QueryPlan:
    intent: Intent
    product: str | None
    version: str | None
    strategy: Strategy
    needs_clarification: bool
    rewritten_query: str
    confidence: float
    queries: tuple[str, ...] = ()
    filters: tuple[FilterCondition, ...] = ()
    reasons: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def valid(self) -> bool:
        return (
            self.intent in INTENTS
            and self.strategy in STRATEGIES
            and isinstance(self.rewritten_query, str)
            and 0.0 <= self.confidence <= 1.0
        )


class Planner(Protocol):
    async def plan(
        self, question: str, product: str | None = None, version: str | None = None
    ) -> QueryPlan: ...


def _clean_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise PlannerError("planner output contains no JSON object")
    return text[start : end + 1]


def parse_plan(text: str) -> QueryPlan:
    try:
        payload = json.loads(_clean_json(text))
    except (ValueError, TypeError) as exc:
        raise PlannerError("planner output is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise PlannerError("planner output is not a JSON object")

    intent = str(payload.get("intent") or "").strip().lower()
    strategy = str(payload.get("strategy") or "").strip().lower()
    product = payload.get("product")
    version = payload.get("version")
    confidence = payload.get("confidence", 0.0)
    reasons = payload.get("reasons") or []
    raw_queries = payload.get("queries") or []
    if isinstance(raw_queries, list):
        queries = tuple(
            str(item).strip()
            for item in raw_queries
            if isinstance(item, str) and str(item).strip()
        )
    elif isinstance(raw_queries, str) and raw_queries.strip():
        queries = (raw_queries.strip(),)
    else:
        queries = ()
    filters = parse_filter_conditions(payload.get("filters"))

    if intent not in INTENTS:
        raise PlannerError(f"invalid intent: {intent!r}")
    if strategy not in STRATEGIES:
        raise PlannerError(f"invalid strategy: {strategy!r}")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError) as exc:
        raise PlannerError("invalid confidence") from exc
    if not 0.0 <= confidence_value <= 1.0:
        raise PlannerError("confidence out of range")

    return QueryPlan(
        intent=intent,  # type: ignore[arg-type]
        product=str(product).strip() if product else None,
        version=str(version).strip() if version else None,
        strategy=strategy,  # type: ignore[arg-type]
        needs_clarification=bool(payload.get("needs_clarification", False)),
        rewritten_query=str(payload.get("rewritten_query") or "").strip(),
        confidence=confidence_value,
        queries=queries,
        filters=filters,
        reasons=tuple(str(item) for item in reasons) if isinstance(reasons, list) else (),
        raw=payload,
    )


def fallback_plan(
    question: str, product: str | None = None, version: str | None = None, error: str = ""
) -> QueryPlan:
    """Conservative plan used when the LLM call/output fails."""

    return QueryPlan(
        intent="clarify" if not product or not version else "lookup",
        product=product,
        version=version,
        strategy="clarify" if not product or not version else "dense",
        needs_clarification=not product or not version,
        rewritten_query=question,
        confidence=0.0,
        reasons=("fallback", error) if error else ("fallback",),
    )


_MODEL_LIKE = re.compile(r"\b[A-Z]{1,2}\d[A-Z0-9]{2,}\b")


def should_use_planner(
    question: str, product: str | None = None, version: str | None = None
) -> tuple[bool, tuple[str, ...]]:
    """Structural gate: call the LLM planner only when the cheap plan is insufficient.

    No query keyword list is used here.  Signals are structural: missing scope,
    multiple question marks, multiple model-like identifiers, or a long question.
    """

    reasons: list[str] = []
    if not product or not version:
        reasons.append("missing_scope")
    if question.count("?") + question.count("？") > 1:
        reasons.append("multiple_questions")
    models = {match.upper() for match in _MODEL_LIKE.findall(question)}
    if len(models) >= 2:
        reasons.append("multiple_models")
    if len(question) > 160:
        reasons.append("long_question")
    return bool(reasons), tuple(reasons)


class DashScopePlanner:
    """One-shot JSON planner over an OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "qwen-plus",
        base_url: str = "https://dashscope.aliyuncs.com",
        path: str = "/compatible-mode/v1/chat/completions",
        timeout_seconds: float = 30.0,
        max_retries: int = 1,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._url = base_url.rstrip("/") + path
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._client = client
        self.last_usage: dict[str, Any] = {}

    async def plan(
        self, question: str, product: str | None = None, version: str | None = None
    ) -> QueryPlan:
        content = await self._complete(question, product, version)
        return parse_plan(content)

    async def _complete(self, question: str, product: str | None, version: str | None) -> str:
        user = json.dumps(
            {
                "question": question,
                "explicit_product": product,
                "explicit_version": version,
            },
            ensure_ascii=False,
        )
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        client = self._client or httpx.AsyncClient(timeout=self._timeout, trust_env=False)
        owns_client = self._client is None
        try:
            last_error: Exception | None = None
            for attempt in range(self._max_retries + 1):
                try:
                    response = await client.post(
                        self._url,
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        json=body,
                    )
                    if response.status_code >= 500 or response.status_code == 429:
                        raise PlannerError(f"planner upstream status {response.status_code}")
                    response.raise_for_status()
                    payload = response.json()
                    self.last_usage = payload.get("usage") or {}
                    content = payload["choices"][0]["message"]["content"]
                    if not isinstance(content, str):
                        raise PlannerError("planner returned non-text content")
                    return content
                except Exception as exc:  # retry once on transport/5xx
                    last_error = exc
                    if attempt >= self._max_retries:
                        break
            raise PlannerError(f"planner call failed: {type(last_error).__name__}")
        finally:
            if owns_client:
                await client.aclose()


class DeepSeekPlanner(DashScopePlanner):
    """Direct DeepSeek planner for local development (chat-completions path)."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        timeout_seconds: float = 30.0,
        max_retries: int = 1,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            api_key,
            model=model,
            base_url=base_url,
            path="/chat/completions",
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            client=client,
        )
