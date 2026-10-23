"""Neighbour expansion: append at most two document-adjacent chunks around the best hit.

Extracted verbatim from ``RagQueryService._expand_neighbors``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..schemas import SearchHit


class NeighborExpander:
    def __init__(self, host: Any) -> None:
        object.__setattr__(self, "_host", host)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._host, name)

    async def expand_neighbors(
        self, selected: Sequence[SearchHit], radius: int
    ) -> list[SearchHit]:
        """Append document-adjacent chunks around the best hit, at most two.

        This is a retrieval-selection policy, not a separate context budget: it
        targets evidence that spans adjacent register/table blocks (e.g. R050).
        """

        from ..rag_service import _as_uuid, _get  # local import: rag_service imports this module

        if not selected or radius <= 0:
            return list(selected)
        try:
            async with self._repo_scope() as repo:
                pairs = await repo.get_chunk_neighbors(selected[0].chunk_id, radius)
        except Exception:
            return list(selected)
        seen = {str(hit.chunk_id) for hit in selected}
        neighbors: list[SearchHit] = []
        for chunk, document in pairs:
            document_id = _as_uuid(_get(document, "id", "document_id"))
            assembled = self._assemble_hit(
                chunk,
                {"vector_score": 0.0, "score": 0.0},
                {document_id: document},
            )
            if assembled is None or str(assembled.chunk_id) in seen:
                continue
            seen.add(str(assembled.chunk_id))
            neighbors.append(assembled)
            if len(neighbors) >= 2:
                break
        return [selected[0], *neighbors, *selected[1:]]
