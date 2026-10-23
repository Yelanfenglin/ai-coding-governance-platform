from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.planning import (
    DashScopePlanner,
    PlannerError,
    fallback_plan,
    parse_plan,
    should_use_planner,
)

VALID = {
    "intent": "comparison",
    "product": "M8P515",
    "version": "V2.3",
    "strategy": "multi_document",
    "needs_clarification": False,
    "rewritten_query": "M8P515 和 M8P516 的 PWM 分辨率有什么区别？",
    "confidence": 0.82,
    "reasons": ["two models"],
}


def test_parse_plan_accepts_plain_and_fenced_json() -> None:
    plan = parse_plan(json.dumps(VALID, ensure_ascii=False))
    assert plan.intent == "comparison"
    assert plan.strategy == "multi_document"
    assert plan.product == "M8P515"
    assert plan.needs_clarification is False
    assert plan.valid

    fenced = "```json\n" + json.dumps(VALID) + "\n```"
    assert parse_plan(fenced).intent == "comparison"


def test_parse_plan_accepts_multi_query_variants() -> None:
    payload = {
        **VALID,
        "queries": [
            "M8P515 和 M8P516 的 PWM 分辨率有什么区别？",
            "M8P515 PWM 分辨率是多少？",
            "M8P516 PWM 分辨率是多少？",
        ],
    }
    plan = parse_plan(json.dumps(payload, ensure_ascii=False))
    assert plan.queries == (
        "M8P515 和 M8P516 的 PWM 分辨率有什么区别？",
        "M8P515 PWM 分辨率是多少？",
        "M8P516 PWM 分辨率是多少？",
    )


def test_parse_plan_accepts_typed_filters() -> None:
    payload = {
        **VALID,
        "filters": [
            {"field": "register", "operator": "in", "value": ["PUA", "PUB"]},
            {"field": "section_id", "operator": "eq", "value": "6.3"},
        ],
    }
    plan = parse_plan(json.dumps(payload, ensure_ascii=False))
    assert [(item.field, item.operator, item.values) for item in plan.filters] == [
        ("register", "in", ("PUA", "PUB")),
        ("section_id", "eq", ("6.3",)),
    ]


def test_parse_plan_rejects_invalid_payloads() -> None:
    with pytest.raises(PlannerError):
        parse_plan("not json")
    with pytest.raises(PlannerError):
        parse_plan(json.dumps({**VALID, "intent": "guess"}))
    with pytest.raises(PlannerError):
        parse_plan(json.dumps({**VALID, "confidence": 1.5}))


def test_fallback_plan_requires_clarification_without_scope() -> None:
    plan = fallback_plan("怎么配置？", error="timeout")
    assert plan.needs_clarification is True
    assert plan.strategy == "clarify"
    assert plan.confidence == 0.0
    assert "fallback" in plan.reasons
    assert "timeout" in plan.reasons

    scoped = fallback_plan("ADCON0 的低两位", "M8P6272", "V1.10")
    assert scoped.needs_clarification is False
    assert scoped.strategy == "dense"


def test_dashscope_planner_parses_response_and_records_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "qwen-plus"
        assert body["response_format"] == {"type": "json_object"}
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(VALID)}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160},
            },
        )

    async def run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        planner = DashScopePlanner("test-key", client=client)
        plan = await planner.plan("M8P515 和 M8P516 有什么区别？")
        assert plan.intent == "comparison"
        assert planner.last_usage["total_tokens"] == 160
        await client.aclose()

    asyncio.run(run())


def test_dashscope_planner_retries_then_raises() -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, json={"error": "upstream"})

    async def run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        planner = DashScopePlanner("test-key", client=client, max_retries=1)
        with pytest.raises(PlannerError):
            await planner.plan("q")
        await client.aclose()

    asyncio.run(run())
    assert calls["n"] == 2


def test_gate_uses_structural_signals_only() -> None:
    assert should_use_planner("ADCON0 的低两位是什么？", "M8P6272", "V1.10") == (
        False,
        (),
    )
    use, reasons = should_use_planner("ADCON0 的低两位是什么？")
    assert use is True
    assert reasons == ("missing_scope",)

    use, reasons = should_use_planner("M8P515 和 M8P516 有什么区别？", "M8P515", "V2.3")
    assert use is True
    assert "multiple_models" in reasons

    use, reasons = should_use_planner("第一个问题？第二个问题？", "M8P515", "V2.3")
    assert use is True
    assert "multiple_questions" in reasons

    use, reasons = should_use_planner("怎么配置？", "M8P515", "V2.3")
    assert use is False
    assert reasons == ()
