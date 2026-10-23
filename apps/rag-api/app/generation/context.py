"""Context assembly: turn selected hits into generation contexts.

Extracted verbatim from ``RagQueryService._build_contexts``.  There is deliberately no
separate character/token budget here; the caller decides how many hits to pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..schemas import SearchHit

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..rag_service import GenerationContext


class ContextBuilder:
    def __init__(self, host: Any) -> None:
        object.__setattr__(self, "_host", host)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._host, name)

    def build_contexts(
        self, hits: Sequence[SearchHit]
    ) -> tuple[list[GenerationContext], list[SearchHit]]:
        from ..rag_service import GenerationContext  # local import: rag_service imports this module

        # Context size is controlled by the selected top-N hits; there is deliberately
        # no separate character/token budget. The caller decides how many hits to pass.
        contexts: list[GenerationContext] = []
        selected_hits: list[SearchHit] = []
        for hit in hits:
            contexts.append(
                GenerationContext(
                    chunk_id=hit.chunk_id,
                    source_tag=self.source_tag(hit),
                    text=hit.text,
                )
            )
            selected_hits.append(hit)
        return contexts, selected_hits
