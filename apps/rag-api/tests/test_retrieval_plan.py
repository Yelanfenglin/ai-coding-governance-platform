from __future__ import annotations

from types import SimpleNamespace

from app.retrieval import (
    RetrievalPlan,
    apply_rerank_results,
    build_plan,
    normalize_question,
    rule_rerank,
)


def _hit(chunk_id: str, section: str, score: float = 0.5, kind: str = "TEXT") -> SimpleNamespace:
    return SimpleNamespace(
        chunk_id=chunk_id,
        section_path=section,
        text=f"text-{chunk_id}",
        score=score,
        vector_score=score,
        rerank_score=None,
        kind=kind,
    )


def test_normalize_question_strips_and_normalizes_width() -> None:
    assert normalize_question("  ＡＤＣＯＮ０  的低两位？ ") == "ADCON0 的低两位?"


def test_build_plan_is_deterministic_and_keeps_explicit_scope() -> None:
    plan = build_plan("  PWM 波形图  ", "M8P516", "V1.10")
    assert isinstance(plan, RetrievalPlan)
    assert plan.product == "M8P516"
    assert plan.version == "V1.10"
    assert plan.strategy == "dense"
    assert plan.reasons == ("explicit_product", "explicit_version")
    assert plan.normalized_question == "PWM 波形图"
    assert plan.needs_clarification is False

    missing = build_plan("PWM 波形图")
    assert missing.needs_clarification is True


def test_rule_rerank_dedupes_and_preserves_cosine_order() -> None:
    hits = [
        _hit("a", "1 概述"),
        _hit("b", "2 定时器"),
        _hit("a", "1 概述"),
        _hit("c", "3 中断"),
    ]
    plan = build_plan("ADC 转换结束后读哪个标志")
    selected = rule_rerank(hits, plan, 5)
    assert [hit.chunk_id for hit in selected] == ["a", "b", "c"]


def test_rule_rerank_caps_prose_sections_at_two() -> None:
    hits = [
        _hit("r1", "6.2 IOB"),
        _hit("r2", "6.2 IOB"),
        _hit("r3", "6.2 IOB"),
        _hit("r4", "6.3 IOD"),
    ]
    plan = build_plan("PUD6 控制位能不能置 1")
    selected = rule_rerank(hits, plan, 5)
    assert [hit.chunk_id for hit in selected] == ["r1", "r2", "r4"]


def test_rule_rerank_allows_three_visual_candidates_per_section() -> None:
    hits = [
        _hit("f1", "7.7 PWMx 波形图", kind="VISUAL"),
        _hit("f2", "7.7 PWMx 波形图", kind="VISUAL"),
        _hit("f3", "7.7 PWMx 波形图", kind="VISUAL"),
        _hit("f4", "7.7 PWMx 波形图", kind="VISUAL"),
    ]
    plan = build_plan("PWM 波形图里两路的高电平宽度")
    selected = rule_rerank(hits, plan, 5)
    assert [hit.chunk_id for hit in selected] == ["f1", "f2", "f3"]


def test_rule_rerank_honours_limit() -> None:
    hits = [_hit(f"h{i}", f"section-{i}") for i in range(10)]
    plan = build_plan("ADC 转换结束后读哪个标志")
    assert len(rule_rerank(hits, plan, 3)) == 3


def test_apply_rerank_results_orders_by_score() -> None:
    hits = [_hit("a", "1", 0.2), _hit("b", "2", 0.9)]
    results = [
        SimpleNamespace(id="b", rerank_score=0.95),
        SimpleNamespace(id="a", rerank_score=0.4),
    ]
    reranked = apply_rerank_results(hits, results, 2)
    assert [hit.chunk_id for hit in reranked] == ["b", "a"]
