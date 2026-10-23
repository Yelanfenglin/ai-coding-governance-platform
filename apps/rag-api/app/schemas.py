"""Pydantic models for the public RAG HTTP boundary.

The database and provider layers deliberately do not leak their models through
this module.  In particular, all user supplied scope values are normalized
before a repository or vector-store adapter sees them.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

QUESTION_MAX_LENGTH = 4_000
# Settings may lower this ceiling at runtime.  Keeping a hard protocol ceiling
# prevents an unbounded request from reaching a service whose settings were not
# yet resolved (for example, OpenAPI validation and unit tests).
TOP_K_PROTOCOL_MAX = 50

_VERSION_RE = re.compile(r"^[Vv]?(?P<major>\d+)\.(?P<minor>\d+)$")


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PARSING = "parsing"
    PARSED = "parsed"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class KnowledgeChunkKind(StrEnum):
    TEXT = "TEXT"
    TABLE = "TABLE"
    VISUAL = "VISUAL"
    MIXED = "MIXED"
    CODE = "CODE"


Language = Literal["zh", "en", "zh-CN"]


def normalize_version(value: str) -> str:
    """Return the canonical ``Vx.y`` representation used by PostgreSQL."""

    if not isinstance(value, str):
        raise ValueError("version must be a string")
    match = _VERSION_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError("version must use the Vx.y format")
    major = int(match.group("major"))
    minor = int(match.group("minor"))
    return f"V{major}.{minor}"


def normalize_product(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("product must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError("product must not be blank")
    return normalized


def normalize_question(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("question must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError("question must not be blank")
    if len(normalized) > QUESTION_MAX_LENGTH:
        raise ValueError(f"question must contain at most {QUESTION_MAX_LENGTH} characters")
    return normalized


class _ResponseModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True, extra="ignore")


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JobAccepted(_ResponseModel):
    job_id: UUID
    document_id: UUID
    status: DocumentStatus

    @model_validator(mode="after")
    def job_is_document(self) -> JobAccepted:
        if self.job_id != self.document_id:
            raise ValueError("job_id must equal document_id in API v1")
        return self


class DocumentStatusResponse(_ResponseModel):
    document_id: UUID = Field(validation_alias=AliasChoices("document_id", "id"))
    status: DocumentStatus
    progress: Annotated[int | None, Field(ge=0, le=100)] = None
    error: str | None = Field(
        default=None,
        validation_alias=AliasChoices("error", "parse_error"),
    )


class DocumentResponse(DocumentStatusResponse):
    product: str
    version: str
    language: Language
    title: str
    filename: str
    current: bool
    file_sha256: str | None = None
    size_bytes: Annotated[int | None, Field(ge=0)] = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    superseded_at: datetime | None = None

    @field_validator("product")
    @classmethod
    def canonical_product(cls, value: str) -> str:
        return normalize_product(value)

    @field_validator("version")
    @classmethod
    def canonical_version(cls, value: str) -> str:
        return normalize_version(value)


class DocumentListResponse(_ResponseModel):
    items: list[DocumentResponse]
    total: Annotated[int, Field(ge=0)]


class _QuestionScopeModel(_RequestModel):
    question: str
    product: str | None = None
    version: str | None = None
    top_k: Annotated[int, Field(ge=1, le=TOP_K_PROTOCOL_MAX)] = 8

    @field_validator("question", mode="before")
    @classmethod
    def clean_question(cls, value: object) -> str:
        return normalize_question(value)  # type: ignore[arg-type]

    @field_validator("product", mode="before")
    @classmethod
    def clean_product(cls, value: object) -> str | None:
        if value is None:
            return None
        return normalize_product(value)  # type: ignore[arg-type]

    @field_validator("version", mode="before")
    @classmethod
    def clean_version(cls, value: object) -> str | None:
        if value is None:
            return None
        return normalize_version(value)  # type: ignore[arg-type]


class QueryRequest(_QuestionScopeModel):
    pass


class SearchParams(_QuestionScopeModel):
    kind: KnowledgeChunkKind | None = None


class SearchHit(_ResponseModel):
    chunk_id: UUID
    document_id: UUID
    product: str
    version: str
    section_path: str
    page_start: Annotated[int, Field(ge=1)]
    page_end: Annotated[int, Field(ge=1)]
    kind: KnowledgeChunkKind
    text: str
    # ``score`` remains the frozen API field.  The two explicit fields make its
    # provenance unambiguous without inventing a misleading "confidence".
    score: float
    vector_score: float
    rerank_score: float | None = None

    @model_validator(mode="after")
    def valid_page_range(self) -> SearchHit:
        if self.page_end < self.page_start:
            raise ValueError("page_end must be greater than or equal to page_start")
        return self


class Citation(_ResponseModel):
    product: str
    version: str
    section_path: str
    page: Annotated[int, Field(ge=1)]
    kind: KnowledgeChunkKind
    text: str
    score: float
    chunk_id: UUID
    vector_score: float
    rerank_score: float | None = None


class EvidenceFinding(_ResponseModel):
    kind: str
    detail: str
    product: str = ""
    register: str = ""
    bit: int | None = None


class AnswerFinding(_ResponseModel):
    kind: str
    value: str = ""
    detail: str = ""


class QueryChecks(_ResponseModel):
    evidence: list[EvidenceFinding] = Field(default_factory=list)
    answer: list[AnswerFinding] = Field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return bool(self.evidence or self.answer)


class QueryResponse(_ResponseModel):
    answer: str
    citations: list[Citation]
    model: str
    latency_ms: Annotated[int, Field(ge=0)]
    degraded: bool = False
    degraded_reason: str | None = None
    checks: QueryChecks | None = None


class ChunkContext(_ResponseModel):
    chunk_id: UUID = Field(validation_alias=AliasChoices("chunk_id", "id"))
    document_id: UUID
    product: str
    version: str
    section_path: str
    section_title: str | None = None
    page_start: Annotated[int, Field(ge=1)]
    page_end: Annotated[int, Field(ge=1)]
    kind: KnowledgeChunkKind
    text: str
    table_repr: dict[str, object] | list[object] | None = None
    image_ref: str | None = None
    chunk_metadata: dict[str, object] | None = None


class DependencyHealth(_ResponseModel):
    ok: bool
    configured: bool = True
    detail: str | None = None
    dimension: int | None = None
    collection: str | None = None
    loaded: bool | None = None


class HealthResponse(_ResponseModel):
    ready: bool
    postgres: DependencyHealth
    milvus: DependencyHealth
    providers: dict[str, DependencyHealth]


__all__ = [
    "ChunkContext",
    "Citation",
    "DependencyHealth",
    "DocumentListResponse",
    "DocumentResponse",
    "DocumentStatus",
    "DocumentStatusResponse",
    "HealthResponse",
    "JobAccepted",
    "KnowledgeChunkKind",
    "Language",
    "QueryRequest",
    "QueryResponse",
    "SearchHit",
    "SearchParams",
    "TOP_K_PROTOCOL_MAX",
    "normalize_product",
    "normalize_question",
    "normalize_version",
]
