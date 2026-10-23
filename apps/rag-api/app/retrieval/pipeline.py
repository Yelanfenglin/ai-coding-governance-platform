"""Retrieval pipeline: the five retrieval steps, extracted verbatim from the service.

Nothing here is re-implemented.  The bodies are the same code that used to live on
``RagQueryService``; the pipeline proxies host attributes (store, repo scope, hit
assembly, bm25 cache, ``_bm25_degraded``) so behaviour cannot drift.  Host-only
helpers and the two error classes are imported lazily to avoid a circular import,
because ``rag_service`` imports this module.

Scope: retrieval only.  selection / context / generation / checks stay on the
service.  Defaults, caching, filters and degradation behaviour are unchanged.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from ..query_filters import FilterCondition
from ..schemas import KnowledgeChunkKind, QueryRequest, SearchHit, SearchParams
from .bm25 import BM25Index, fuse_with_quota


class RetrievalPipeline:
    """Runs the retrieval steps against a host ``RagQueryService``."""

    def __init__(self, host: Any) -> None:
        object.__setattr__(self, "_host", host)

    def __getattr__(self, name: str) -> Any:
        # Only reached when the pipeline itself has no such attribute: store,
        # repository scope, hit assembly, bm25 cache and every host helper resolve
        # through here, so the moved bodies keep their original `self._x` form.
        return getattr(self._host, name)

    @property
    def _bm25_degraded(self) -> bool:
        return bool(getattr(self._host, "_bm25_degraded", False))

    @_bm25_degraded.setter
    def _bm25_degraded(self, value: bool) -> None:
        # State stays on the host: query() reads it to build degraded_reason.
        self._host._bm25_degraded = value

    async def retrieve(
        self,
        request: SearchParams | QueryRequest,
        filters: Sequence[FilterCondition] = (),
    ) -> list[SearchHit]:
        from ..rag_service import _setting  # local import: rag_service imports this module
        self._validate_top_k(request.top_k)
        if not bool(_setting(self._settings, "multi_source_routing_enabled", True)):
            return await self._retrieve_one(request, filters)
        if request.product is not None:
            return await self._retrieve_one(request, filters)
        products = await self._matched_products(request.question)
        if len(products) < 2:
            return await self._retrieve_one(request, filters)
        return await self._retrieve_multi_source(request, products, filters)

    async def retrieve_multi_source(
        self,
        request: SearchParams | QueryRequest,
        products: Sequence[str],
        filters: Sequence[FilterCondition] = (),
    ) -> list[SearchHit]:
        from ..rag_service import _setting  # local import: rag_service imports this module
        limit = max(1, int(_setting(self._settings, "multi_source_max_products", 3)))
        batches: list[list[SearchHit]] = []
        degraded = False
        for product in products[:limit]:
            scoped_request = request.model_copy(
                update={"product": product, "version": None}
            )
            hits = await self._retrieve_one(scoped_request, filters)
            degraded = degraded or self._bm25_degraded
            if hits:
                batches.append(hits)
        self._bm25_degraded = degraded
        if not batches:
            return []
        return self._merge_multi_source_hits(
            batches,
            request.top_k,
            k0=float(_setting(self._settings, "rrf_k0", 60.0)),
        )

    async def retrieve_queries(
        self,
        request: SearchParams | QueryRequest,
        planned_queries: Sequence[str],
        filters: Sequence[FilterCondition] = (),
    ) -> list[SearchHit]:
        from ..rag_service import _setting  # local import: rag_service imports this module
        if not bool(_setting(self._settings, "multi_query_enabled", True)):
            return await self._retrieve(request, filters)
        limit = max(1, int(_setting(self._settings, "multi_query_max_queries", 3)))
        variants: list[str] = [request.question]
        for candidate in planned_queries:
            normalized = str(candidate).strip()
            if normalized and normalized not in variants:
                variants.append(normalized)
        if len(variants) == 1:
            for candidate in self._structural_query_variants(request.question):
                if candidate not in variants:
                    variants.append(candidate)
        variants = variants[:limit]
        if len(variants) <= 1:
            return await self._retrieve(request, filters)

        batches: list[list[SearchHit]] = []
        degraded = False
        for variant in variants:
            variant_request = request.model_copy(update={"question": variant})
            hits = await self._retrieve(variant_request, filters)
            degraded = degraded or self._bm25_degraded
            if hits:
                batches.append(hits)
        self._bm25_degraded = degraded
        if not batches:
            return []
        return self._merge_multi_source_hits(
            batches,
            request.top_k,
            k0=float(_setting(self._settings, "rrf_k0", 60.0)),
        )

    async def retrieve_one(
        self,
        request: SearchParams | QueryRequest,
        filters: Sequence[FilterCondition] = (),
    ) -> list[SearchHit]:
        from ..rag_service import (  # local import: rag_service imports this module
            RagServiceError,
            RagUnavailableError,
            SearchScope,
            _as_kind,
            _as_uuid,
            _get,
            _setting,
        )
        self._validate_top_k(request.top_k)
        self._bm25_degraded = False

        async with self._repo_scope() as repo:
            documents = await repo.resolve_search_documents(
                SearchScope(product=request.product, version=request.version)
            )

        if not documents:
            return []

        document_ids: list[UUID] = []
        documents_by_id: dict[UUID, object] = {}
        for document in documents:
            try:
                document_id = _as_uuid(_get(document, "id", "document_id"))
            except (TypeError, ValueError):
                continue
            document_ids.append(document_id)
            documents_by_id[document_id] = document
        if not document_ids:
            return []

        await self._assert_embedding_compatible()
        vector = await self._embed_question(request.question)
        kinds: tuple[KnowledgeChunkKind, ...] = ()
        kind = getattr(request, "kind", None)
        if kind is not None:
            kinds = (_as_kind(kind),)
        vector_scope = self._vector_scope(document_ids, kinds)

        pushed = tuple(filters) if self._store_accepts_filters() else ()
        try:
            if pushed:
                vector_hits = await self._store.search(
                    vector, vector_scope, request.top_k, filters=pushed
                )
            else:
                vector_hits = await self._store.search(vector, vector_scope, request.top_k)
        except RagServiceError:
            raise
        except Exception as exc:
            raise RagUnavailableError("vector search is unavailable") from exc
        if not vector_hits:
            return []

        embedding_ids = [
            str(_get(hit, "embedding_id", "id"))
            for hit in vector_hits
            if _get(hit, "embedding_id", "id") is not None
        ]
        if not embedding_ids:
            return []

        async with self._repo_scope() as repo:
            canonical = await repo.get_chunks_by_embedding_ids(embedding_ids)
        chunks_by_embedding = self._canonical_mapping(canonical)

        ranked: list[SearchHit] = []
        for vector_hit in vector_hits:
            embedding_id_value = _get(vector_hit, "embedding_id", "id")
            if embedding_id_value is None:
                continue
            chunk = chunks_by_embedding.get(str(embedding_id_value))
            if chunk is None:
                # Milvus can temporarily lead PG during compensation/reconcile.
                # Never return its preview as canonical evidence.
                continue
            assembled = self._assemble_hit(chunk, vector_hit, documents_by_id)
            if assembled is not None:
                ranked.append(assembled)
        if not bool(_setting(self._settings, "bm25_enabled", False)):
            return ranked
        try:
            return await self._fuse_bm25(request, ranked, document_ids, documents_by_id)
        except Exception:
            # Lexical recall is optional; never fail the request for it.
            self._bm25_degraded = True
            return ranked

    async def fuse_bm25(
        self,
        request: SearchParams | QueryRequest,
        dense: list[SearchHit],
        document_ids: Sequence[UUID],
        documents_by_id: Mapping[UUID, object],
    ) -> list[SearchHit]:
        from ..rag_service import (  # local import: rag_service imports this module
            _as_uuid,
            _get,
            _setting,
        )
        if not dense:
            return dense
        cache_key = tuple(sorted(str(document_id) for document_id in document_ids))
        now = time.monotonic()
        ttl = float(_setting(self._settings, "bm25_index_ttl_seconds", 600.0))
        cached = self._bm25_cache.get(cache_key)
        if cached is None or now - cached[0] > ttl:
            async with self._repo_scope() as repo:
                rows = await repo.list_chunks_for_documents(document_ids)
            chunk_map: dict[str, object] = {}
            records: list[tuple[str, str, str]] = []
            for row in rows:
                chunk_id = str(_get(row, "id", "chunk_id"))
                metadata = _get(row, "chunk_metadata", "metadata_json", default={})
                text = _get(row, "embedding_text") or _get(row, "text") or ""
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
                document = documents_by_id.get(_as_uuid(_get(row, "document_id")))
                product = str(_get(document, "product", default="")) if document else ""
                chunk_map[chunk_id] = row
                records.append((chunk_id, product, str(text)))
            index = await asyncio.to_thread(BM25Index.from_records, records)
            if len(self._bm25_cache) >= int(_setting(self._settings, "bm25_cache_max_scopes", 32)):
                oldest = min(self._bm25_cache, key=lambda key: self._bm25_cache[key][0])
                self._bm25_cache.pop(oldest, None)
            self._bm25_cache[cache_key] = (now, index, chunk_map)
            cached = self._bm25_cache[cache_key]
        index, chunk_map = cached[1], cached[2]

        products = sorted(
            {str(_get(doc, "product", default="")) for doc in documents_by_id.values()}
        )
        product_filter: str | tuple[str, ...] | None
        if not products:
            product_filter = None
        elif len(products) == 1:
            product_filter = products[0]
        else:
            product_filter = tuple(products)
        sparse_hits = index.search(
            request.question,
            product=product_filter,
            top_k=int(_setting(self._settings, "bm25_top_n", 30)),
        )
        sparse: list[SearchHit] = []
        for sparse_hit in sparse_hits:
            row = chunk_map.get(str(sparse_hit.chunk_id))
            if row is None:
                continue
            assembled = self._assemble_hit(
                row,
                {"vector_score": 0.0, "score": float(sparse_hit.score)},
                documents_by_id,
            )
            if assembled is not None:
                sparse.append(assembled)
        if not sparse:
            return dense
        fused = fuse_with_quota(
            [str(hit.chunk_id) for hit in dense],
            [str(hit.chunk_id) for hit in sparse],
            dense_slots=int(_setting(self._settings, "fusion_dense_slots", 3)),
            sparse_slots=int(_setting(self._settings, "fusion_sparse_slots", 2)),
            k0=float(_setting(self._settings, "rrf_k0", 60.0)),
            limit=request.top_k,
        )
        by_id: dict[str, SearchHit] = {str(hit.chunk_id): hit for hit in (*dense, *sparse)}
        return [by_id[str(item.chunk_id)] for item in fused if str(item.chunk_id) in by_id]
