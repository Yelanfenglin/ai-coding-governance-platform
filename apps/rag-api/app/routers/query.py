"""Search, RAG query, and canonical context routes."""

from __future__ import annotations

from typing import Annotated, NoReturn
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from ..dependencies import get_rag_service, require_api_key
from ..rag_service import (
    GenerationUnavailableError,
    RagNotFoundError,
    RagQueryService,
    RagUnavailableError,
    RagValidationError,
)
from ..schemas import ChunkContext, QueryRequest, QueryResponse, SearchHit, SearchParams

router = APIRouter(tags=["query"], dependencies=[Depends(require_api_key)])


def _raise_http(exc: Exception) -> NoReturn:
    if isinstance(exc, RagNotFoundError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from None
    if isinstance(exc, RagValidationError):
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from None
    if isinstance(exc, (GenerationUnavailableError, RagUnavailableError)):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from None
    raise exc


def search_params(
    question: Annotated[str, Query(min_length=1, max_length=4_000)],
    product: Annotated[str | None, Query()] = None,
    version: Annotated[str | None, Query()] = None,
    kind: Annotated[str | None, Query()] = None,
    top_k: Annotated[int, Query(ge=1, le=50)] = 8,
    rerank: Annotated[bool, Query()] = False,
    candidate_k: Annotated[int | None, Query(ge=1, le=50)] = None,
) -> SearchParams:
    """Build the model explicitly so validator failures remain HTTP 422."""

    try:
        return SearchParams(
            question=question,
            product=product,
            version=version,
            kind=kind,
            top_k=top_k,
            rerank=rerank,
            candidate_k=candidate_k,
        )
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


@router.get("/search", response_model=list[SearchHit])
async def search(
    params: Annotated[SearchParams, Depends(search_params)],
    service: Annotated[RagQueryService, Depends(get_rag_service)],
) -> list[SearchHit]:
    try:
        return await service.search(params)
    except (RagNotFoundError, RagValidationError, RagUnavailableError) as exc:
        _raise_http(exc)


@router.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    service: Annotated[RagQueryService, Depends(get_rag_service)],
) -> QueryResponse:
    try:
        return await service.query(body)
    except (
        GenerationUnavailableError,
        RagNotFoundError,
        RagValidationError,
        RagUnavailableError,
    ) as exc:
        _raise_http(exc)


@router.get(
    "/documents/{doc_id}/context/{chunk_id}",
    response_model=ChunkContext,
)
async def context(
    doc_id: UUID,
    chunk_id: UUID,
    service: Annotated[RagQueryService, Depends(get_rag_service)],
) -> ChunkContext:
    try:
        return await service.context(doc_id, chunk_id)
    except (RagNotFoundError, RagValidationError, RagUnavailableError) as exc:
        _raise_http(exc)


__all__ = ["context", "query", "router", "search", "search_params"]
