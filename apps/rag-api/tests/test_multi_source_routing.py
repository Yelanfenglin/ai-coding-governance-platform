from __future__ import annotations

from uuid import uuid4

from app.rag_service import RagQueryService
from app.schemas import KnowledgeChunkKind, SearchHit


def _hit(chunk_id, score: float, product: str = "M8P5170") -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id=uuid4(),
        product=product,
        version="V1.6",
        section_path="5 中断",
        page_start=10,
        page_end=10,
        kind=KnowledgeChunkKind.TEXT,
        text="evidence",
        score=score,
        vector_score=score,
    )


def test_products_in_question_detects_multiple_products_in_order() -> None:
    products = ["M8P5170", "M8P6272", "M9F6601"]
    found = RagQueryService._products_in_question(
        "对照 M8P6272 V1.10 和 M8P5170 V1.06，控制位在哪里？",
        products,
    )
    assert found == ["M8P6272", "M8P5170"]


def test_products_in_question_does_not_match_product_prefix() -> None:
    assert RagQueryService._products_in_question("M8P5150 的寄存器", ["M8P515"]) == []


def test_products_in_question_ignores_unmentioned_products() -> None:
    assert RagQueryService._products_in_question("M8P5170 的寄存器", ["M8P5170", "M8P6272"]) == [
        "M8P5170"
    ]


def test_structural_query_variants_splits_multi_question_prompt() -> None:
    variants = RagQueryService._structural_query_variants(
        "M8P515 的 PWM 分辨率是多少？M8P6272 的 PWM 分辨率是多少？"
    )
    assert variants == [
        "M8P515 的 PWM 分辨率是多少",
        "M8P6272 的 PWM 分辨率是多少",
    ]


def test_structural_query_variants_keeps_single_question() -> None:
    assert RagQueryService._structural_query_variants("ADCON0 的低两位是什么？") == []


def test_merge_multi_source_hits_deduplicates_and_rank_fuses() -> None:
    shared, first_only, filler, second_only = uuid4(), uuid4(), uuid4(), uuid4()
    batches = [
        [_hit(shared, 1.0), _hit(first_only, 0.9)],
        [_hit(shared, 1.0), _hit(filler, 0.9), _hit(second_only, 0.8)],
    ]
    merged = RagQueryService._merge_multi_source_hits(batches, limit=4, k0=60.0)
    assert merged[0].chunk_id == shared
    assert {hit.chunk_id for hit in merged} == {shared, first_only, filler, second_only}
    assert merged[0].score > merged[1].score
    assert merged[-1].score < merged[1].score


def test_merge_multi_source_hits_respects_limit() -> None:
    batches = [[_hit(uuid4(), 1.0), _hit(uuid4(), 0.9)]]
    assert len(RagQueryService._merge_multi_source_hits(batches, limit=1)) == 1
