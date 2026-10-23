"""Retrieval planning and candidate selection (Phase 1).

Online-chain front half, deterministic and provider-free: query normalization and
scope planning -> rule-based candidate selection (dedupe + evidence-aware section
diversity) -> context assembly.  Recall itself stays in ``RagQueryService`` (dense
Milvus search); BM25/RRF and model-based rerank are later phases.

Deliberately no query-word intent lexicon: a keyword list misclassifies too easily
("单位"/"位置" matched the register marker "位", "不要看图" matched the figure
marker, "程序存储器" matched the code marker).  Section diversity is decided from
the retrieved evidence (hit kind), not from words in the question.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    """Structured plan produced from a user question and explicit scope."""

    question: str
    normalized_question: str
    product: str | None
    version: str | None
    strategy: str
    reasons: tuple[str, ...]

    @property
    def needs_clarification(self) -> bool:
        """True when no explicit product/version was supplied.

        Phase 1 does not enforce clarification yet: callers may use this flag to
        warn/route.  The retrieval scope stays exactly the request's scope so the
        existing API behaviour is preserved.
        """

        return self.product is None or self.version is None


def normalize_question(question: str) -> str:
    """NFKC + whitespace normalization; keeps wording untouched."""

    text = unicodedata.normalize("NFKC", question or "")
    return re.sub(r"\s+", " ", text).strip()


def build_plan(
    question: str, product: str | None = None, version: str | None = None
) -> RetrievalPlan:
    normalized = normalize_question(question)
    reasons: list[str] = []
    if product:
        reasons.append("explicit_product")
    if version:
        reasons.append("explicit_version")
    return RetrievalPlan(
        question=question,
        normalized_question=normalized,
        product=product,
        version=version,
        strategy="dense",
        reasons=tuple(reasons),
    )


def _chunk_id(hit: object) -> str:
    return str(getattr(hit, "chunk_id", "")) or id(hit).__repr__()


def _section_key(hit: object) -> str:
    value = getattr(hit, "section_path", "") or ""
    return str(value)


def _section_cap(hit: object) -> int:
    """How many candidates one section may contribute.

    Figure/visual sections are allowed one extra slot because multi-figure questions
    legitimately need several figures from the same section; register/table/prose
    sections stay narrower so a template block cannot occupy the whole context.
    This is decided from the retrieved hit, not from the question wording.
    """

    return 3 if str(getattr(hit, "kind", "")) == "VISUAL" else 2


def rule_rerank(hits: list[object], plan: RetrievalPlan, limit: int) -> list[object]:
    """Deterministic candidate selection used when model rerank is disabled.

    - dedupe by chunk_id, preserving cosine order;
    - cap the number of candidates from one section, based on the retrieved hit
      kind rather than a query keyword list;
    - never invent, reorder, or score candidates.
    """

    if limit <= 0 or not hits:
        return []
    seen: set[str] = set()
    per_section: dict[str, int] = {}
    selected: list[object] = []
    for hit in hits:
        chunk_id = _chunk_id(hit)
        if chunk_id in seen:
            continue
        section = _section_key(hit)
        if section and per_section.get(section, 0) >= _section_cap(hit):
            continue
        seen.add(chunk_id)
        if section:
            per_section[section] = per_section.get(section, 0) + 1
        selected.append(hit)
        if len(selected) >= limit:
            break
    return selected


def apply_rerank_results(hits: list[object], results: object, top_n: int) -> list[object]:
    """Apply provider rerank results (kept provider-agnostic for tests/fakes)."""

    import math
    from collections.abc import Sequence

    if not isinstance(results, Sequence):
        return []
    by_id = {str(getattr(hit, "chunk_id", "")): hit for hit in hits}
    scored: list[tuple[object, float, int]] = []
    for ordinal, result in enumerate(results):
        hit = None
        raw_score: object
        if isinstance(result, (int, float)):
            if ordinal >= len(hits):
                continue
            hit = hits[ordinal]
            raw_score = result
        else:
            index = getattr(result, "index", None) or getattr(result, "candidate_index", None)
            candidate_id = (
                getattr(result, "candidate_id", None)
                or getattr(result, "chunk_id", None)
                or getattr(result, "id", None)
            )
            if index is not None:
                try:
                    hit = hits[int(index)]
                except (IndexError, TypeError, ValueError):
                    hit = None
            elif candidate_id is not None:
                hit = by_id.get(str(candidate_id))
            raw_score = getattr(result, "rerank_score", None) or getattr(result, "score", None)
        if hit is None:
            continue
        try:
            rerank_score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(rerank_score):
            continue
        scored.append((hit, rerank_score, ordinal))

    scored.sort(key=lambda item: (-item[1], item[2]))
    seen: set[str] = set()
    reranked: list[object] = []
    for hit, score, _ in scored:
        chunk_id = str(getattr(hit, "chunk_id", ""))
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        update = getattr(hit, "model_copy", None)
        reranked.append(
            update(update={"score": score, "rerank_score": score}) if callable(update) else hit
        )
        if len(reranked) == top_n:
            break
    return reranked
