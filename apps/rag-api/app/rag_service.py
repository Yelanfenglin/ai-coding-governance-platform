"""Retrieval and generation orchestration for the RAG API."""

from __future__ import annotations

import asyncio
import inspect
import math
import re
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from opentelemetry import trace

from .answer_checks import check_answer
from .checks.runtime import to_query_checks
from .evidence_checks import (
    Citation as EvidenceCitation,
)
from .evidence_checks import (
    detect_duplicate_registers,
    detect_findings,
)
from .filter_translator import translate_filters
from .generation.citations import build_citation
from .generation.context import ContextBuilder
from .observability import metrics
from .observability.otel import observe, observed
from .observability.trace import QueryTrace, TraceSink
from .planning import should_use_planner
from .query_filters import FilterCondition
from .retrieval import (
    RetrievalPlan,
    build_plan,
)
from .retrieval.bm25 import BM25Index
from .retrieval.pipeline import RetrievalPipeline
from .scalar_metadata import package_mentioned_in
from .schemas import (
    ChunkContext,
    Citation,
    KnowledgeChunkKind,
    QueryChecks,
    QueryRequest,
    QueryResponse,
    SearchHit,
    SearchParams,
    normalize_version,
)
from .selection.neighbor import NeighborExpander
from .selection.selector import SelectionPipeline


class RagServiceError(RuntimeError):
    """Base class for safe, router-facing service errors."""


class RagNotFoundError(RagServiceError):
    pass


class RagValidationError(RagServiceError):
    pass


class RagUnavailableError(RagServiceError):
    pass


class GenerationUnavailableError(RagUnavailableError):
    pass


@dataclass(frozen=True, slots=True)
class SearchScope:
    product: str | None
    version: str | None


@dataclass(frozen=True, slots=True)
class VectorSearchScope:
    """Structured Milvus scope; user strings never become expressions."""

    document_ids: Sequence[UUID]
    kinds: Sequence[KnowledgeChunkKind] = ()


@dataclass(frozen=True, slots=True)
class RerankInput:
    id: UUID
    chunk: object
    text: str
    rerank_score: float

    @property
    def candidate_id(self) -> UUID:
        return self.id

    @property
    def chunk_id(self) -> UUID:
        return self.id

    @property
    def vector_score(self) -> float:
        return self.rerank_score


@dataclass(frozen=True, slots=True)
class GenerationContext:
    chunk_id: UUID
    source_tag: str
    text: str

    @property
    def content(self) -> str:
        return f"{self.source_tag}\n{self.text}"


def _get(value: object, *names: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _setting(settings: object | None, name: str, default: Any) -> Any:
    if settings is None:
        return default
    if isinstance(settings, Mapping):
        return settings.get(name, default)
    return getattr(settings, name, default)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _as_uuid(value: object) -> UUID:
    if isinstance(value, UUID):
        return value
    return UUID(str(value))


def _as_kind(value: object) -> KnowledgeChunkKind:
    if isinstance(value, KnowledgeChunkKind):
        return value
    raw = getattr(value, "value", value)
    return KnowledgeChunkKind(str(raw).upper())


def _section_path(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        return " > ".join(str(part) for part in value)
    return str(value)


class RagQueryService:
    """Coordinate PG scoping, vector recall, rerank, and optional generation."""

    def __init__(
        self,
        session_factory: Any = None,
        store: Any = None,
        embedding_provider: Any = None,
        reranker: Any = None,
        llm_provider: Any = None,
        planner: Any = None,
        settings: object | None = None,
        *,
        repository_factory: Any = None,
        repository: Any = None,
    ) -> None:
        self._session_factory = session_factory
        self._store = store
        self._embedding_provider = embedding_provider
        self._reranker = reranker
        self._llm_provider = llm_provider
        self._planner = planner
        self._settings = settings
        self._repository_factory = repository_factory
        self._repository = repository
        self._bm25_cache: dict[tuple[str, ...], tuple[float, BM25Index, dict[str, object]]] = {}
        self._bm25_degraded = False
        trace_enabled = bool(_setting(settings, "query_trace_enabled", False))
        trace_path = _setting(settings, "query_trace_path", None)
        self._trace_sink = TraceSink(trace_path if trace_enabled else None)

    @property
    def max_top_k(self) -> int:
        return int(_setting(self._settings, "query_top_k_max", 50))

    @asynccontextmanager
    async def _repo_scope(self) -> AsyncIterator[Any]:
        if self._repository is not None:
            yield self._repository
            return
        if self._session_factory is None:
            raise RagUnavailableError("repository is unavailable")

        session_context = self._session_factory()
        session_context = await _maybe_await(session_context)
        async with session_context as session:
            factory = self._repository_factory
            if factory is None:
                try:
                    from .db.repo import RagRepository
                except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - integration
                    raise RagUnavailableError("repository is unavailable") from exc
                factory = RagRepository
            yield factory(session)

    def _store_accepts_filters(self) -> bool:
        """True when the store's ``search`` takes a ``filters`` keyword.

        Duck-typed stores (and older adapters) do not, so the capability is
        detected once instead of passed blindly and retried on TypeError.
        """

        cached = getattr(self, "_store_filters_supported", None)
        if cached is not None:
            return bool(cached)
        try:
            parameters = inspect.signature(self._store.search).parameters
        except (TypeError, ValueError):
            supported = False
        else:
            supported = "filters" in parameters or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        self._store_filters_supported = supported
        return supported

    def _validate_top_k(self, top_k: int) -> None:
        if not 1 <= top_k <= self.max_top_k:
            raise RagValidationError(f"top_k must be between 1 and {self.max_top_k}")

    async def search(self, request: SearchParams) -> list[SearchHit]:
        with observe("rag.search", "RETRIEVER", **{
            "rag.top_k": request.top_k, "rag.product": request.product or "",
            "rag.version": request.version or "",
        }) as span:
            self._validate_top_k(request.top_k)
            hits = await self._retrieve(request)
            metrics.HITS.observe(len(hits))
            if getattr(self, "_bm25_degraded", False):
                metrics.DEGRADED.labels("bm25").inc()
            span.set_attributes({"rag.hit_count": len(hits),
                                 "rag.bm25_degraded": bool(getattr(self, "_bm25_degraded", False))})
            for index, hit in enumerate(hits):
                span.set_attribute(f"retrieval.documents.{index}.document.id", str(hit.chunk_id))
                span.set_attribute(f"rag.documents.{index}.document_id", str(hit.document_id))
            return hits

    @property
    def _retrieval_pipeline(self) -> RetrievalPipeline:
        """Lazily built so a `__new__`-constructed service still works."""

        pipeline = self.__dict__.get("_retrieval_pipeline_instance")
        if pipeline is None:
            pipeline = RetrievalPipeline(self)
            self.__dict__["_retrieval_pipeline_instance"] = pipeline
        return pipeline

    async def _retrieve(
        self,
        request: SearchParams | QueryRequest,
        filters: Sequence[FilterCondition] = (),
    ) -> list[SearchHit]:
        return await self._retrieval_pipeline.retrieve(request, filters)


    async def _retrieve_multi_source(
        self,
        request: SearchParams | QueryRequest,
        products: Sequence[str],
        filters: Sequence[FilterCondition] = (),
    ) -> list[SearchHit]:
        return await self._retrieval_pipeline.retrieve_multi_source(request, products, filters)


    async def _retrieve_queries(
        self,
        request: SearchParams | QueryRequest,
        planned_queries: Sequence[str],
        filters: Sequence[FilterCondition] = (),
    ) -> list[SearchHit]:
        return await self._retrieval_pipeline.retrieve_queries(request, planned_queries, filters)


    @staticmethod
    def _identifier_in_text(value: str, text: str) -> bool:
        pattern = rf"(?<![A-Z0-9]){re.escape(value)}(?![A-Z0-9-])"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None

    @staticmethod
    def _section_matches(value: str, section: str) -> bool:
        pattern = rf"(?<![\d.]){re.escape(value)}(?![\d])"
        return re.search(pattern, section, flags=re.IGNORECASE) is not None

    @staticmethod
    def _apply_planned_filters(
        hits: Sequence[SearchHit], filters: Sequence[FilterCondition]
    ) -> tuple[list[SearchHit], tuple[str, ...]]:
        """Keep explicit product/version hard scope; relax optional filters only."""

        current = list(hits)
        hard_scope = tuple(
            condition for condition in filters if condition.field in {"product", "version"}
        )
        optional = tuple(
            condition for condition in filters if condition.field not in {"product", "version"}
        )
        if hard_scope:
            current = [
                hit for hit in current if RagQueryService._hit_matches_filters(hit, hard_scope)
            ]
            if not current:
                return [], ("query_filter_hard_scope_no_match",)
        reasons: list[str] = []
        if optional:
            filtered = [
                hit for hit in current if RagQueryService._hit_matches_filters(hit, optional)
            ]
            if filtered:
                current = filtered
            else:
                reasons.append("query_filter_relaxed_optional")
        return current, tuple(reasons)

    @staticmethod
    def _hit_matches_filters(
        hit: SearchHit, filters: Sequence[FilterCondition]
    ) -> bool:
        """Exact, field-aware post-retrieval filter.

        This is the safety filter until the same semantics are pushed into the
        new collection's scalar filter expression.
        """

        section = hit.section_path
        text = hit.text
        for condition in filters:
            values = {value.casefold() for value in condition.values}
            if condition.field == "product":
                matched = hit.product.casefold() in values
            elif condition.field == "version":
                matched = normalize_version(hit.version) in {
                    normalize_version(value) for value in condition.values
                }
            elif condition.field == "register":
                matched = any(
                    RagQueryService._identifier_in_text(value, text)
                    or RagQueryService._identifier_in_text(value, section)
                    for value in condition.values
                )
            elif condition.field == "package":
                # Membership on the canonical package vocabulary, so "SOP28"
                # never matches an "SSOP28" hit.  SearchHit does not carry the
                # stored scalar field yet; text evidence is the same evidence the
                # backfill extracts from, so both paths agree.
                matched = any(
                    package_mentioned_in(text, value)
                    or package_mentioned_in(section, value)
                    for value in condition.values
                )
            elif condition.field == "section_id":
                matched = any(
                    RagQueryService._section_matches(value, section)
                    for value in condition.values
                )
            else:  # pragma: no cover - parser rejects unknown fields
                matched = True
            if not matched:
                return False
        return True

    @staticmethod
    def _structural_query_variants(question: str) -> list[str]:
        """Split a multi-question prompt into independently searchable variants."""

        parts = [part.strip() for part in re.split(r"[?？;；]+", question)]
        parts = [part for part in parts if len(part) >= 6]
        return parts[:2] if len(parts) >= 2 else []

    @staticmethod
    def _merge_multi_source_hits(
        batches: Sequence[Sequence[SearchHit]],
        limit: int,
        *,
        k0: float = 60.0,
    ) -> list[SearchHit]:
        """Reciprocal-rank merge of per-product candidate lists."""

        scores: dict[str, float] = defaultdict(float)
        first: dict[str, SearchHit] = {}
        for hits in batches:
            for rank, hit in enumerate(hits, 1):
                key = str(hit.chunk_id)
                scores[key] += 1.0 / (k0 + rank)
                first.setdefault(key, hit)
        ordered = sorted(scores, key=lambda key: (-scores[key], key))
        return [
            first[key].model_copy(update={"score": scores[key]})
            for key in ordered[: max(1, int(limit))]
        ]

    async def _matched_products(self, question: str) -> list[str]:
        try:
            async with self._repo_scope() as repo:
                documents = await repo.resolve_search_documents(
                    SearchScope(product=None, version=None)
                )
        except Exception:
            return []
        products = sorted(
            {
                str(_get(document, "product", default="")).strip()
                for document in documents
                if str(_get(document, "product", default="")).strip()
            },
            key=len,
            reverse=True,
        )
        return self._products_in_question(question, products)

    @staticmethod
    def _products_in_question(question: str, products: Sequence[str]) -> list[str]:
        upper = question.upper()
        matches: list[tuple[int, str]] = []
        for product in products:
            name = product.upper()
            pattern = rf"(?<![A-Z0-9]){re.escape(name)}(?![A-Z0-9])"
            match = re.search(pattern, upper)
            if match is not None:
                matches.append((match.start(), product))
        return [product for _, product in sorted(matches)]

    async def _retrieve_one(
        self,
        request: SearchParams | QueryRequest,
        filters: Sequence[FilterCondition] = (),
    ) -> list[SearchHit]:
        return await self._retrieval_pipeline.retrieve_one(request, filters)


    async def _fuse_bm25(
        self,
        request: SearchParams | QueryRequest,
        dense: list[SearchHit],
        document_ids: Sequence[UUID],
        documents_by_id: Mapping[UUID, object],
    ) -> list[SearchHit]:
        return await self._retrieval_pipeline.fuse_bm25(
            request, dense, document_ids, documents_by_id
        )

    @staticmethod
    def _vector_scope(document_ids: Sequence[UUID], kinds: Sequence[KnowledgeChunkKind]) -> object:
        """Use the store's nominal DTO while retaining a lightweight test fallback."""

        try:
            from .vectordb.milvus_store import KnowledgeChunkKind as StoreChunkKind
            from .vectordb.milvus_store import VectorSearchScope as StoreSearchScope
        except (ImportError, ModuleNotFoundError):  # pragma: no cover - isolated API tests
            return VectorSearchScope(document_ids=tuple(document_ids), kinds=tuple(kinds))
        store_kinds = tuple(StoreChunkKind(kind.value) for kind in kinds)
        return StoreSearchScope(document_ids=tuple(document_ids), kinds=store_kinds)

    @staticmethod
    def _canonical_mapping(canonical: object) -> dict[str, object]:
        if isinstance(canonical, Mapping):
            return {str(key): value for key, value in canonical.items()}
        if isinstance(canonical, Sequence):
            result: dict[str, object] = {}
            for chunk in canonical:
                embedding_id = _get(chunk, "embedding_id")
                if embedding_id is not None:
                    result[str(embedding_id)] = chunk
            return result
        return {}

    def _assemble_hit(
        self,
        chunk: object,
        vector_hit: object,
        documents_by_id: Mapping[UUID, object],
    ) -> SearchHit | None:
        try:
            chunk_id = _as_uuid(_get(chunk, "id", "chunk_id"))
            document_id = _as_uuid(_get(chunk, "document_id"))
            document = documents_by_id.get(document_id)
            if document is None:
                return None
            raw_score = _get(vector_hit, "vector_score", "score", "distance")
            vector_score = float(raw_score)
            if not math.isfinite(vector_score):
                raise ValueError("non-finite vector score")
            page_start = int(_get(chunk, "page_start"))
            page_end = int(_get(chunk, "page_end"))
            # Reuse the indexed evidence, including figure descriptions and inline code.
            # The display text can contain only an image placeholder.
            text = _get(chunk, "embedding_text")
            if not isinstance(text, str) or not text.strip():
                text = _get(chunk, "text")  # Legacy records without indexed text.
            if not isinstance(text, str):
                return None
            metadata = _get(chunk, "chunk_metadata", "metadata_json", default={})
            if isinstance(metadata, Mapping):
                uncertainties = metadata.get("figure_uncertainties", [])
                if isinstance(uncertainties, list):
                    notes = [
                        note
                        for note in uncertainties
                        if isinstance(note, str) and note.strip() and note not in text
                    ]
                    if notes:
                        text += "\n不确定项：" + "；".join(notes)
            return SearchHit(
                chunk_id=chunk_id,
                document_id=document_id,
                product=str(_get(document, "product", default=_get(chunk, "product"))),
                version=str(_get(document, "version", default=_get(chunk, "version"))),
                section_path=_section_path(_get(chunk, "section_path")),
                page_start=page_start,
                page_end=page_end,
                kind=_as_kind(_get(chunk, "kind")),
                text=text,
                score=vector_score,
                vector_score=vector_score,
                rerank_score=None,
            )
        except (TypeError, ValueError):
            return None

    async def _assert_embedding_compatible(self) -> None:
        if self._store is None or self._embedding_provider is None:
            raise RagUnavailableError("embedding search is unavailable")

        provider_dimension = _get(self._embedding_provider, "dimension")
        configured_dimension = int(
            _setting(
                self._settings,
                "embedding_dimension",
                provider_dimension if provider_dimension is not None else 1024,
            )
        )
        if provider_dimension is not None and int(provider_dimension) != configured_dimension:
            raise RagUnavailableError("embedding dimension does not match configuration")
        store_dimension = _get(self._store, "dimension", "embedding_dimension")
        if store_dimension is not None and int(store_dimension) != configured_dimension:
            raise RagUnavailableError("embedding dimension does not match the index")

        provider_fingerprint = _get(
            self._embedding_provider,
            "fingerprint",
            "model_fingerprint",
            "model_id",
        )
        assertion = getattr(self._store, "assert_embedding_fingerprint", None)
        if assertion is not None:
            try:
                await _maybe_await(assertion(provider_fingerprint))
            except Exception as exc:
                raise RagUnavailableError("embedding fingerprint does not match the index") from exc
            return

        store_fingerprint = _get(
            self._store,
            "embedding_fingerprint",
            "embedding_model_id",
            "model_fingerprint",
            "fingerprint",
        )
        if (
            provider_fingerprint is not None
            and store_fingerprint is not None
            and str(provider_fingerprint) != str(store_fingerprint)
        ):
            raise RagUnavailableError("embedding fingerprint does not match the index")

    @observed("rag.embed", "EMBEDDING")
    async def _embed_question(self, question: str) -> list[float]:
        span = trace.get_current_span()
        span.set_attribute("embedding.model_name", str(_setting(
            self._settings, "embedding_model",
            _get(self._embedding_provider, "model_id", default=""),
        )))
        try:
            vectors = await self._embedding_provider.embed([question])
        except Exception as exc:
            # A request must never switch to a different embedding implementation.
            raise RagUnavailableError("embedding provider is unavailable") from exc
        if len(vectors) != 1:
            raise RagUnavailableError("embedding provider returned an invalid batch")
        try:
            vector = [float(value) for value in vectors[0]]
        except (TypeError, ValueError) as exc:
            raise RagUnavailableError("embedding provider returned an invalid vector") from exc
        provider_dimension = _get(self._embedding_provider, "dimension")
        expected_dimension = int(
            _setting(
                self._settings,
                "embedding_dimension",
                provider_dimension if provider_dimension is not None else len(vector),
            )
        )
        if len(vector) != expected_dimension or not all(math.isfinite(value) for value in vector):
            raise RagUnavailableError("embedding provider returned an invalid vector")
        span.set_attribute("embedding.dimension", expected_dimension)
        return vector

    @observed("rag.query")
    async def query(self, request: QueryRequest) -> QueryResponse:
        trace_enabled = bool(_setting(self._settings, "query_trace_enabled", False))
        trace = QueryTrace() if trace_enabled else None
        trace_started = time.perf_counter() if trace is not None else 0.0
        try:
            return await self._query_impl(request, trace)
        except Exception as exc:
            if trace is not None:
                trace.record("error", trace_started, error=type(exc).__name__)
            raise
        finally:
            if trace is not None:
                await self._finish_trace(trace)

    async def _query_impl(
        self, request: QueryRequest, trace: QueryTrace | None
    ) -> QueryResponse:
        started = time.perf_counter()
        effective_request = request
        planner_degraded: list[str] = []
        planned_queries: tuple[str, ...] = ()
        planned_filters: tuple[FilterCondition, ...] = ()
        planner_started = self._trace_start(trace)
        planner_enabled = bool(
            _setting(self._settings, "planner_enabled", self._planner is not None)
        )
        if self._planner is not None and planner_enabled:
            use_planner, _gate_reasons = should_use_planner(
                request.question, request.product, request.version
            )
            if use_planner:
                try:
                    llm_plan = await self._planner.plan(
                        request.question, request.product, request.version
                    )
                    if not llm_plan.valid:
                        raise ValueError("invalid planner output")
                    planned_queries = llm_plan.queries
                    planned_filters = llm_plan.filters
                    if llm_plan.needs_clarification:
                        planner_degraded.append("planner_clarification_required")
                    if (
                        request.product is None
                        and request.version is None
                        and llm_plan.intent in {"lookup", "multi_evidence"}
                        and llm_plan.product
                        and llm_plan.version
                    ):
                        effective_request = request.model_copy(
                            update={
                                "question": llm_plan.rewritten_query or request.question,
                                "product": llm_plan.product,
                                "version": llm_plan.version,
                            }
                        )
                        planner_degraded.append("planner_scope_completed")
                except Exception:
                    planner_degraded.append("planner_unavailable")
        self._trace_record(
            trace,
            "plan",
            planner_started,
            enabled=planner_enabled,
            queries=len(planned_queries),
            filters=len(planned_filters),
        )

        plan = build_plan(
            effective_request.question, effective_request.product, effective_request.version
        )
        retrieval_started = self._trace_start(trace)
        try:
            hits = await self._retrieve_queries(
                effective_request, planned_queries, planned_filters
            )
        except Exception as exc:
            self._trace_record(
                trace, "retrieve", retrieval_started, error=type(exc).__name__
            )
            raise
        if not hits and planned_filters and self._store_accepts_filters():
            pushed_clause, _remaining = translate_filters(planned_filters)
            if pushed_clause:
                # A pushed-down clause can be narrower than the relaxed
                # post-filter; ask again without pushdown before giving up.
                relaxed_hits = await self._retrieve_queries(effective_request, planned_queries)
                if relaxed_hits:
                    hits = relaxed_hits
                    planner_degraded.append("query_filter_pushdown_relaxed")
        if planned_filters:
            hits, filter_reasons = self._apply_planned_filters(hits, planned_filters)
            planner_degraded.extend(filter_reasons)
            if "query_filter_hard_scope_no_match" in filter_reasons:
                raise RagNotFoundError(
                    "no evidence matched the planner product/version scope"
                )
        if self._bm25_degraded:
            planner_degraded.append("bm25_unavailable")
        self._trace_record(
            trace,
            "retrieve",
            retrieval_started,
            hits=len(hits),
            bm25_degraded=self._bm25_degraded,
            error="no_hits" if not hits else None,
        )
        if not hits:
            raise RagNotFoundError("no evidence matched the requested scope")

        selection_started = self._trace_start(trace)
        try:
            selected, rerank_degraded = await self._select_for_generation(plan, hits)
        except Exception as exc:
            self._trace_record(
                trace, "select", selection_started, error=type(exc).__name__
            )
            raise
        neighbor_window = int(_setting(self._settings, "context_neighbor_window", 0))
        try:
            if neighbor_window > 0:
                selected = await self._expand_neighbors(selected, neighbor_window)
            contexts, context_hits = self._build_contexts(selected)
        except Exception as exc:
            self._trace_record(
                trace, "select", selection_started, error=type(exc).__name__
            )
            raise
        if not contexts:
            self._trace_record(trace, "select", selection_started, error="empty_context")
            raise RagNotFoundError("no evidence fits the generation context budget")
        citations = [
            self._citation(hit, context.text)
            for hit, context in zip(context_hits, contexts, strict=True)
        ]
        self._trace_record(
            trace,
            "select",
            selection_started,
            selected=len(selected),
            contexts=len(contexts),
            rerank_degraded=rerank_degraded,
        )

        failure_mode = (
            str(_setting(self._settings, "generation_failure_mode", "error"))
            .lower()
            .replace("-", "_")
        )
        llm_enabled = bool(_setting(self._settings, "llm_enabled", self._llm_provider is not None))
        degraded_reasons: list[str] = list(planner_degraded)
        if rerank_degraded:
            degraded_reasons.append("reranker_unavailable")

        generation_started = self._trace_start(trace)
        if not llm_enabled or self._llm_provider is None:
            if failure_mode not in {"retrieval_only", "degraded"}:
                raise GenerationUnavailableError("generation is disabled")
            degraded_reasons.append("generation_disabled")
            self._trace_record(
                trace,
                "generate",
                generation_started,
                attempted=False,
                called=False,
                success=False,
            )
            checks_started = self._trace_start(trace)
            checks = self._query_checks(citations, "")
            self._trace_record(
                trace,
                "checks",
                checks_started,
                evidence=len(checks.evidence),
                answer=len(checks.answer),
            )
            return self._retrieval_only_response(started, citations, degraded_reasons, checks)

        try:
            with observe("rag.generate", "LLM", **{
                "llm.model_name": str(_get(self._llm_provider, "model_id", default="configured")),
            }):
                generation = await self._llm_provider.generate(effective_request.question, contexts)
                answer = self._generation_text(generation)
                if not answer:
                    raise ValueError("empty generation")
        except Exception as exc:
            self._trace_record(
                trace,
                "generate",
                generation_started,
                attempted=True,
                called=True,
                success=False,
                error=type(exc).__name__,
            )
            if failure_mode not in {"retrieval_only", "degraded"}:
                raise GenerationUnavailableError("generation provider is unavailable") from exc
            degraded_reasons.append("generation_unavailable")
            metrics.DEGRADED.labels("generation").inc()
            checks_started = self._trace_start(trace)
            checks = self._query_checks(citations, "")
            self._trace_record(
                trace,
                "checks",
                checks_started,
                evidence=len(checks.evidence),
                answer=len(checks.answer),
            )
            return self._retrieval_only_response(started, citations, degraded_reasons, checks)

        if bool(_get(generation, "degraded", default=False)):
            degraded_reasons.append("generation_degraded")

        model = _get(generation, "model", "model_id")
        if model is None:
            model = _get(self._llm_provider, "model_id", default="configured-llm")
        self._trace_record(
            trace,
            "generate",
            generation_started,
            attempted=True,
            called=True,
            success=True,
            model=str(model),
            answer_chars=len(answer),
        )
        checks_started = self._trace_start(trace)
        checks = self._query_checks(citations, answer)
        self._trace_record(
            trace,
            "checks",
            checks_started,
            evidence=len(checks.evidence),
            answer=len(checks.answer),
        )
        return QueryResponse(
            answer=answer,
            citations=citations,
            model=str(model),
            latency_ms=self._latency_ms(started),
            degraded=bool(degraded_reasons),
            degraded_reason=",".join(degraded_reasons) or None,
            checks=checks,
        )

    @property
    def _selection_pipeline(self) -> SelectionPipeline:
        pipeline = self.__dict__.get("_selection_pipeline_instance")
        if pipeline is None:
            pipeline = SelectionPipeline(self)
            self.__dict__["_selection_pipeline_instance"] = pipeline
        return pipeline

    @property
    def _neighbor_expander(self) -> NeighborExpander:
        expander = self.__dict__.get("_neighbor_expander_instance")
        if expander is None:
            expander = NeighborExpander(self)
            self.__dict__["_neighbor_expander_instance"] = expander
        return expander

    @property
    def _context_builder(self) -> ContextBuilder:
        builder = self.__dict__.get("_context_builder_instance")
        if builder is None:
            builder = ContextBuilder(self)
            self.__dict__["_context_builder_instance"] = builder
        return builder

    async def _select_for_generation(
        self, plan: RetrievalPlan, hits: Sequence[SearchHit]
    ) -> tuple[list[SearchHit], bool]:
        return await self._selection_pipeline.select_for_generation(plan, hits)


    async def _expand_neighbors(
        self, selected: Sequence[SearchHit], radius: int
    ) -> list[SearchHit]:
        return await self._neighbor_expander.expand_neighbors(selected, radius)


    def _build_contexts(
        self, hits: Sequence[SearchHit]
    ) -> tuple[list[GenerationContext], list[SearchHit]]:
        return self._context_builder.build_contexts(hits)


    @staticmethod
    def source_tag(hit: SearchHit) -> str:
        pages = (
            str(hit.page_start)
            if hit.page_start == hit.page_end
            else f"{hit.page_start}-{hit.page_end}"
        )
        return (
            f"[source:{hit.chunk_id} "
            f"{hit.product}/{hit.version}/{hit.section_path}/{pages}/{hit.kind.value}]"
        )

    @staticmethod
    def _citation(hit: SearchHit, context_text: str) -> Citation:
        return build_citation(hit, context_text)

    @staticmethod
    def _trace_start(trace: QueryTrace | None) -> float:
        return time.perf_counter() if trace is not None else 0.0

    @staticmethod
    def _trace_record(
        trace: QueryTrace | None, name: str, started: float, **data: object
    ) -> None:
        if trace is not None:
            trace.record(name, started, **data)

    async def _finish_trace(self, trace: QueryTrace) -> None:
        sink = getattr(self, "_trace_sink", None)
        if sink is None:
            return
        try:
            await asyncio.to_thread(sink.write, trace)
        except Exception:
            # Export is advisory; never fail the request because of the sink.
            return

    def _query_checks(self, citations: Sequence[Citation], answer: str) -> QueryChecks:
        """Deterministic evidence/citation advisories for the response."""

        evidence_citations = [
            EvidenceCitation(
                chunk_id=str(citation.chunk_id),
                text=citation.text,
                section_path=citation.section_path,
                kind=citation.kind.value,
            )
            for citation in citations
        ]
        try:
            evidence_findings = (
                *detect_findings(evidence_citations),
                *detect_duplicate_registers(evidence_citations),
            )
            answer_findings = check_answer(answer, evidence_citations)
        except Exception:
            return QueryChecks()
        return to_query_checks(evidence_findings, answer_findings)

    @staticmethod
    def _generation_text(generation: object) -> str:
        if isinstance(generation, str):
            return generation.strip()
        answer = _get(generation, "answer", "text", "content")
        return answer.strip() if isinstance(answer, str) else ""

    @staticmethod
    def _latency_ms(started: float) -> int:
        return max(0, int(round((time.perf_counter() - started) * 1000)))

    def _retrieval_only_response(
        self,
        started: float,
        citations: list[Citation],
        reasons: Sequence[str],
        checks: QueryChecks | None = None,
    ) -> QueryResponse:
        return QueryResponse(
            answer="",
            citations=citations,
            model="retrieval-only",
            latency_ms=self._latency_ms(started),
            degraded=True,
            degraded_reason=",".join(dict.fromkeys(reasons)),
            checks=checks,
        )

    @observed("rag.context", "RETRIEVER")
    async def context(self, document_id: UUID, chunk_id: UUID) -> ChunkContext:
        async with self._repo_scope() as repo:
            canonical = await repo.get_context(document_id, chunk_id)
            document = None
            if canonical is not None:
                get_document = getattr(repo, "get_document", None)
                if get_document is not None:
                    document = await get_document(document_id)
        if canonical is None:
            raise RagNotFoundError("chunk context was not found")

        chunk = canonical
        if isinstance(canonical, tuple) and len(canonical) == 2:
            chunk, document = canonical
        if document is None:
            document = _get(chunk, "document")

        try:
            actual_document_id = _as_uuid(_get(chunk, "document_id"))
            if actual_document_id != document_id:
                raise RagNotFoundError("chunk context was not found")
            return ChunkContext(
                chunk_id=_as_uuid(_get(chunk, "id", "chunk_id")),
                document_id=actual_document_id,
                product=str(_get(document, "product", default=_get(chunk, "product"))),
                version=str(_get(document, "version", default=_get(chunk, "version"))),
                section_path=_section_path(_get(chunk, "section_path")),
                section_title=_get(chunk, "section_title"),
                page_start=int(_get(chunk, "page_start")),
                page_end=int(_get(chunk, "page_end")),
                kind=_as_kind(_get(chunk, "kind")),
                text=str(_get(chunk, "text")),
                table_repr=_get(chunk, "table_repr"),
                image_ref=_get(chunk, "image_ref"),
                chunk_metadata=_get(chunk, "chunk_metadata", "metadata_json"),
            )
        except RagNotFoundError:
            raise
        except (TypeError, ValueError) as exc:
            raise RagUnavailableError("canonical chunk data is invalid") from exc


__all__ = [
    "GenerationContext",
    "GenerationUnavailableError",
    "RagNotFoundError",
    "RagQueryService",
    "RagServiceError",
    "RagUnavailableError",
    "RagValidationError",
    "RerankInput",
    "SearchScope",
    "VectorSearchScope",
]
