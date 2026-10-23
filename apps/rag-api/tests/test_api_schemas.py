from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas import JobAccepted, QueryRequest, SearchHit, SearchParams


def test_query_request_normalizes_boundary_values() -> None:
    request = QueryRequest(
        question="  ADCON0 的低两位是什么？  ",
        product="  M8P6272  ",
        version="1.10",
    )

    assert request.question == "ADCON0 的低两位是什么？"
    assert request.product == "M8P6272"
    assert request.version == "V1.10"
    assert request.top_k == 8
    assert QueryRequest(question="q", version="v01.010").version == "V1.10"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("question", "   "),
        ("question", "x" * 4_001),
        ("version", "1.10.1"),
        ("product", "  "),
        ("top_k", 0),
        ("top_k", 51),
    ],
)
def test_query_request_rejects_invalid_boundaries(field: str, value: object) -> None:
    values: dict[str, object] = {"question": "valid question"}
    values[field] = value
    with pytest.raises(ValidationError):
        QueryRequest.model_validate(values)


def test_search_params_parse_kind_and_do_not_create_expression_fields() -> None:
    params = SearchParams(
        question="  register  ",
        product="M8P6272' || current == true",
        version="V1.10",
        kind="TABLE",
    )

    assert params.question == "register"
    assert params.kind.value == "TABLE"
    assert "expression" not in params.model_dump()
    assert "filter" not in params.model_dump()


def test_job_id_must_equal_document_id() -> None:
    document_id = uuid4()
    accepted = JobAccepted(
        job_id=document_id,
        document_id=document_id,
        status="pending",
    )
    assert accepted.job_id == accepted.document_id

    with pytest.raises(ValidationError):
        JobAccepted(job_id=uuid4(), document_id=document_id, status="pending")


def test_search_score_provenance_is_explicit() -> None:
    hit = SearchHit(
        chunk_id=uuid4(),
        document_id=uuid4(),
        product="M8P6272",
        version="V1.10",
        section_path="2 > 2.6",
        page_start=15,
        page_end=15,
        kind="TABLE",
        text="ADCON0",
        score=0.9,
        vector_score=0.8,
        rerank_score=0.9,
    )

    payload = hit.model_dump()
    assert payload["vector_score"] == 0.8
    assert payload["rerank_score"] == 0.9
    assert "confidence" not in payload
