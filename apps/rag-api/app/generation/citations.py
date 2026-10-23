"""Citation adapter: SearchHit + context text -> API Citation."""

from __future__ import annotations

from ..schemas import Citation, SearchHit


def build_citation(hit: SearchHit, context_text: str) -> Citation:
    """Build the public citation object exactly as the service previously did."""

    return Citation(
        product=hit.product,
        version=hit.version,
        section_path=hit.section_path,
        page=hit.page_start,
        kind=hit.kind,
        text=context_text,
        score=hit.rerank_score if hit.rerank_score is not None else hit.vector_score,
        chunk_id=hit.chunk_id,
        vector_score=hit.vector_score,
        rerank_score=hit.rerank_score,
    )
