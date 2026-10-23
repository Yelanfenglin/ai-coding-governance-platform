from __future__ import annotations

from uuid import uuid4

import pytest

from app.query_filters import FilterCondition, QueryFilterError, parse_filter_conditions
from app.rag_service import RagQueryService
from app.schemas import KnowledgeChunkKind, SearchHit


def _hit(
    *, product="M8P5170", version="V1.6", section="6 端口 > 6.3 PUA", text="PUA 上拉控制"
) -> SearchHit:
    return SearchHit(
        chunk_id=uuid4(),
        document_id=uuid4(),
        product=product,
        version=version,
        section_path=section,
        page_start=10,
        page_end=10,
        kind=KnowledgeChunkKind.MIXED,
        text=text,
        score=1.0,
        vector_score=1.0,
    )


def test_parse_filter_conditions_accepts_eq_and_in() -> None:
    filters = parse_filter_conditions(
        [
            {"field": "product", "operator": "eq", "value": "M8P5170"},
            {"field": "register", "operator": "in", "value": ["PUA", "PUB"]},
        ]
    )
    assert filters == (
        FilterCondition("product", "eq", "M8P5170"),
        FilterCondition("register", "in", ("PUA", "PUB")),
    )


def test_parse_filter_conditions_rejects_unknown_field() -> None:
    with pytest.raises(QueryFilterError):
        parse_filter_conditions([{"field": "unknown", "operator": "eq", "value": "x"}])


def test_parse_filter_conditions_rejects_empty_in() -> None:
    with pytest.raises(QueryFilterError):
        parse_filter_conditions([{"field": "register", "operator": "in", "value": []}])


def test_hard_scope_does_not_fall_back_to_other_products() -> None:
    wrong = _hit(product="M8P6272", version="V1.10")
    kept, reasons = RagQueryService._apply_planned_filters(
        [wrong],
        (
            FilterCondition("product", "eq", "M8P5170"),
            FilterCondition("version", "eq", "V1.6"),
            FilterCondition("register", "eq", "PUA"),
        ),
    )
    assert kept == []
    assert reasons == ("query_filter_hard_scope_no_match",)


def test_optional_filter_relaxes_only_after_hard_scope() -> None:
    hit = _hit()
    kept, reasons = RagQueryService._apply_planned_filters(
        [hit],
        (
            FilterCondition("product", "eq", "M8P5170"),
            FilterCondition("register", "eq", "NOT_PRESENT"),
        ),
    )
    assert kept == [hit]
    assert reasons == ("query_filter_relaxed_optional",)


def test_hit_matches_product_register_and_section() -> None:
    hit = _hit()
    assert RagQueryService._hit_matches_filters(
        hit,
        (
            FilterCondition("product", "eq", "M8P5170"),
            FilterCondition("register", "in", ("PUA", "PUB")),
            FilterCondition("section_id", "eq", "6.3"),
        ),
    )
    assert not RagQueryService._hit_matches_filters(
        hit, (FilterCondition("product", "eq", "M8P6272"),)
    )


def test_hit_matches_package_from_text() -> None:
    hit = _hit(section="19 封装", text="SOP16 外形图")
    assert RagQueryService._hit_matches_filters(
        hit, (FilterCondition("package", "eq", "SOP16"),)
    )
    assert not RagQueryService._hit_matches_filters(
        hit, (FilterCondition("package", "eq", "SSOP24"),)
    )


def test_eq_is_not_substring_for_package_or_register() -> None:
    hit = _hit(section="19 封装 SSOP28", text="SSOP28 外形图")
    assert not RagQueryService._hit_matches_filters(
        hit, (FilterCondition("package", "eq", "SOP28"),)
    )
    register_hit = _hit(section="6.5 IOACR", text="IOACR 方向控制")
    assert not RagQueryService._hit_matches_filters(
        register_hit, (FilterCondition("register", "eq", "IOA"),)
    )


def test_section_id_is_not_arbitrary_substring() -> None:
    hit = _hit(section="16 串口 > 16.3 串口寄存器", text="UART")
    assert not RagQueryService._hit_matches_filters(
        hit, (FilterCondition("section_id", "eq", "6.3"),)
    )
    exact = _hit(section="6 端口 > 6.3 PUA", text="PUA")
    assert RagQueryService._hit_matches_filters(
        exact, (FilterCondition("section_id", "eq", "6.3"),)
    )


def test_version_filter_uses_canonical_version() -> None:
    hit = _hit(version="V1.6")
    assert RagQueryService._hit_matches_filters(
        hit, (FilterCondition("version", "eq", "V1.06"),)
    )
