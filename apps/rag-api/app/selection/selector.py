"""Selection step: pick the contexts that go to generation.

Extracted verbatim from ``RagQueryService._select_for_generation``.  Policy, defaults
and the rerank fallbacks are unchanged; the service keeps a delegating method.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..retrieval import RetrievalPlan, apply_rerank_results, rule_rerank
from ..schemas import SearchHit


class SelectionPipeline:
    def __init__(self, host: Any) -> None:
        object.__setattr__(self, "_host", host)

    def __getattr__(self, name: str) -> Any:
        # Host-owned dependencies (settings, reranker, statics) resolve here so the
        # moved body keeps its original `self._x` form.
        return getattr(self._host, name)

    async def select_for_generation(
        self, plan: RetrievalPlan, hits: Sequence[SearchHit]
    ) -> tuple[list[SearchHit], bool]:
        from ..rag_service import (  # local import: rag_service imports this module
            RerankInput,
            _get,
            _setting,
        )

        top_n = max(1, int(_setting(self._settings, "context_top_n", 4)))
        reranker_enabled = bool(
            _setting(self._settings, "reranker_enabled", self._reranker is not None)
        )
        if not reranker_enabled or self._reranker is None:
            return rule_rerank(list(hits), plan, top_n), False

        candidates = [
            RerankInput(
                id=hit.chunk_id,
                chunk=hit,
                text=hit.text,
                rerank_score=hit.vector_score,
            )
            for hit in hits
        ]
        try:
            results = await self._reranker.rerank(plan.normalized_question, candidates, top_n)
            reranked = apply_rerank_results(list(hits), results, top_n)
            if not reranked:
                raise ValueError("empty rerank result")
            provider_degraded = any(
                bool(_get(result, "degraded", default=False)) for result in results
            )
            if provider_degraded:
                # Do not apply vector-score ordering to a fused BM25 route: sparse
                # candidates legitimately have vector_score=0 and would be pushed
                # below worse dense candidates. Preserve the retrieval order.
                return rule_rerank(list(hits), plan, top_n), True
            return reranked, False
        except Exception:
            # Cosine order is a documented, visible query degradation.
            return rule_rerank(list(hits), plan, top_n), True
