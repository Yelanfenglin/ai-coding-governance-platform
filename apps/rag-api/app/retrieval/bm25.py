"""Pure-Python BM25 sparse recall plus RRF fusion for the retrieval chain.

Position in the chain (Phase 2): ``app.retrieval`` already normalizes the query
and the dense route lives in ``RagQueryService``.  This module adds a second,
provider-free recall route (lexical BM25 over the scope's chunk texts) and a
deterministic fusion step (Reciprocal Rank Fusion, optionally with per-route
reserved slots).  Nothing here touches Milvus, PostgreSQL, the LLM, or settings:
callers pass plain records in and get ordered chunk ids out.

Design choices:

- **Tokenization** is ``NFKC`` + ``lowercase``, then CJK runs become overlapping
  bigrams (a single CJK character is kept as a unigram) and ASCII identifiers
  (``pwm1dh``, ``adcon0``, ``8+4``, ``0.000v``) stay whole.  This matches how the
  corpus spells register names and figures values, and it degrades gracefully
  for a one-character question.
- **Scoring** is classic Okapi BM25 with ``k1=1.2`` / ``b=0.75``.  Corpus
  statistics (document count, average length, document frequencies) are global
  to the index; the product filter is applied *before* scoring, exactly like the
  production scope filter, so a scope never sees another product's chunks.
- **Determinism**: ties break on ``chunk_id`` ascending, so repeated runs and
  tests produce identical lists.  No hash ordering leaks into results.

The inverted index stores postings as packed ``unsigned int`` pairs
``(doc_index, term_frequency)``; document frequencies and IDF are derived on the
fly from posting length, which keeps the 2,867-chunk corpus index small enough
to hold in the API worker (~50 MB) without NumPy or an extra service.
"""

from __future__ import annotations

import math
import re
import unicodedata
from array import array
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

BM25_K1 = 1.2
BM25_B = 0.75
RRF_K0 = 60.0

# Lowercased input only: NFKC folding happens before this pattern runs.
_ASCII_TOKEN = re.compile(r"[a-z0-9][a-z0-9_./+\-]*")
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")

Record = tuple[str, str, str]
RankedItem = object


def tokenize(text: str) -> list[str]:
    """Split text into searchable tokens.

    NFKC + lowercase, CJK bigrams (single CJK characters kept as unigrams), and
    ASCII identifiers kept whole.  Returns occurrences in text order; callers
    that need term frequencies use :class:`collections.Counter`.
    """

    if not text:
        return []
    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for match in _ASCII_TOKEN.finditer(normalized):
        token = match.group(0).strip("._/+-\u2013")
        if token:
            tokens.append(token)
    for match in _CJK_RUN.finditer(normalized):
        run = match.group(0)
        if len(run) == 1:
            tokens.append(run)
            continue
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


@dataclass(frozen=True, slots=True)
class BM25Hit:
    """One scored chunk.  Ordering of the returned list is the ranking."""

    chunk_id: str
    score: float


@dataclass(frozen=True, slots=True)
class FusedHit:
    """One fused candidate, with the routes that contributed and the RRF score."""

    chunk_id: str
    score: float
    routes: tuple[str, ...] = ()


class BM25Index:
    """Inverted BM25 index over ``(chunk_id, product, text)`` records."""

    def __init__(
        self,
        records: Iterable[Record],
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        if not math.isfinite(k1) or k1 < 0:
            raise ValueError("k1 must be a finite non-negative number")
        if not math.isfinite(b) or not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1")
        self._k1 = float(k1)
        self._b = float(b)

        chunk_ids: list[str] = []
        doc_lens: array = array("I")
        by_product: dict[str, array] = {}
        postings: dict[str, array] = {}
        seen: set[str] = set()

        for record in records:
            chunk_id, product, text = record
            chunk_id = str(chunk_id)
            if chunk_id in seen:
                # One chunk id may appear once; duplicates would double-count.
                continue
            seen.add(chunk_id)
            product = str(product)
            doc_index = len(chunk_ids)
            chunk_ids.append(chunk_id)
            by_product.setdefault(product, array("I")).append(doc_index)

            tokens = tokenize(str(text or ""))
            doc_lens.append(len(tokens))
            for token, frequency in Counter(tokens).items():
                packed = postings.get(token)
                if packed is None:
                    packed = postings[token] = array("I")
                packed.append(doc_index)
                packed.append(frequency)

        self._chunk_ids = chunk_ids
        self._by_product = by_product
        self._postings = postings
        self._doc_lens = doc_lens
        self._size = len(chunk_ids)
        self._avgdl = (sum(doc_lens) / self._size) if self._size else 0.0

    @classmethod
    def from_records(cls, records: Iterable[Record], **options: float) -> BM25Index:
        """Named constructor for readability at call sites."""

        return cls(records, **options)

    def __len__(self) -> int:
        return self._size

    @property
    def size(self) -> int:
        """Number of indexed chunks."""

        return self._size

    @property
    def vocabulary_size(self) -> int:
        """Number of distinct tokens."""

        return len(self._postings)

    def search(
        self,
        query: str,
        product: str | Sequence[str] | None = None,
        top_k: int = 10,
    ) -> list[BM25Hit]:
        """Rank chunks for ``query`` after filtering to one or more products.

        ``product`` accepts a single product string, a sequence of products (for
        multi-source questions), or ``None`` for no product filter.  An empty
        query, empty index, unknown product, or non-positive ``top_k`` yields an
        empty list rather than an exception: sparse recall failing must never
        break the online chain, the caller falls back to dense-only.
        """

        if top_k <= 0 or self._size == 0 or self._avgdl <= 0:
            return []
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []

        allowed = self._allowed_indices(product)
        if allowed is not None and not allowed:
            return []

        k1, b, avgdl = self._k1, self._b, self._avgdl
        n_docs = self._size
        scores: dict[int, float] = {}
        for token in query_tokens:
            packed = self._postings.get(token)
            if not packed:
                continue
            frequency = len(packed) // 2
            idf = math.log(1.0 + (n_docs - frequency + 0.5) / (frequency + 0.5))
            for offset in range(0, len(packed), 2):
                doc_index = packed[offset]
                if allowed is not None and doc_index not in allowed:
                    continue
                term_frequency = packed[offset + 1]
                length_norm = 1.0 - b + b * (self._doc_lens[doc_index] / avgdl)
                scores[doc_index] = scores.get(doc_index, 0.0) + (
                    idf * term_frequency * (k1 + 1.0) / (term_frequency + k1 * length_norm)
                )
        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda item: (-item[1], self._chunk_ids[item[0]]))
        return [BM25Hit(self._chunk_ids[index], score) for index, score in ranked[:top_k]]

    def _allowed_indices(self, product: str | Sequence[str] | None) -> set[int] | None:
        if product is None:
            return None
        products: tuple[str, ...] = (
            (product,) if isinstance(product, str) else tuple(str(item) for item in product)
        )
        allowed: set[int] = set()
        for name in products:
            packed = self._by_product.get(name)
            if packed is not None:
                allowed.update(packed)
        return allowed


def _ranked_ids(ranked: Iterable[RankedItem]) -> list[str]:
    """Extract chunk ids from a route, dropping repeats after their first rank."""

    ids: list[str] = []
    seen: set[str] = set()
    for item in ranked:
        raw = item if isinstance(item, str) else getattr(item, "chunk_id", None)
        if raw is None:
            raise TypeError("ranked items must be chunk-id strings or expose chunk_id")
        chunk_id = str(raw)
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        ids.append(chunk_id)
    return ids


def _validate_fusion_options(
    k0: float, weights: Sequence[float]
) -> tuple[float, tuple[float, ...]]:
    if not math.isfinite(k0) or k0 < 0:
        raise ValueError("k0 must be a finite non-negative number")
    if len(weights) != 2:
        raise ValueError("weights must contain exactly two entries (dense, sparse)")
    normalized: tuple[float, ...] = tuple(float(weight) for weight in weights)
    if any(not math.isfinite(weight) or weight < 0 for weight in normalized):
        raise ValueError("weights must be finite non-negative numbers")
    return float(k0), normalized


def _rrf_scores(
    dense: Sequence[str],
    sparse: Sequence[str],
    *,
    k0: float,
    weights: tuple[float, ...],
) -> dict[str, tuple[float, list[str]]]:
    fused: dict[str, tuple[float, list[str]]] = {}
    for name, ids, weight in (("dense", dense, weights[0]), ("sparse", sparse, weights[1])):
        if weight == 0:
            continue
        for rank, chunk_id in enumerate(ids, 1):
            score, routes = fused.setdefault(chunk_id, (0.0, []))
            fused[chunk_id] = (score + weight / (k0 + rank), [*routes, name])
    return fused


def fuse_routes(
    dense_ranked: Iterable[RankedItem],
    sparse_ranked: Iterable[RankedItem],
    *,
    k0: float = RRF_K0,
    weights: Sequence[float] = (1.0, 1.0),
    limit: int | None = None,
) -> list[FusedHit]:
    """Reciprocal Rank Fusion of the dense and sparse routes.

    ``score(chunk) = sum_routes weight_route / (k0 + rank_route)`` with rank
    starting at 1.  A chunk missing from one route simply gets no contribution
    from it, which is why a sparse-only hit can outrank a mid-ranked dense hit.
    Duplicate chunk ids inside one route keep their first (best) rank only.
    """

    k0_value, weight_values = _validate_fusion_options(k0, weights)
    dense = _ranked_ids(dense_ranked)
    sparse = _ranked_ids(sparse_ranked)
    fused = _rrf_scores(dense, sparse, k0=k0_value, weights=weight_values)
    ordered = sorted(fused.items(), key=lambda item: (-item[1][0], item[0]))
    hits = [FusedHit(chunk_id, score, tuple(routes)) for chunk_id, (score, routes) in ordered]
    if limit is not None:
        if limit <= 0:
            return []
        return hits[:limit]
    return hits


def fuse_with_quota(
    dense_ranked: Iterable[RankedItem],
    sparse_ranked: Iterable[RankedItem],
    *,
    dense_slots: int = 3,
    sparse_slots: int = 2,
    k0: float = RRF_K0,
    weights: Sequence[float] = (1.0, 1.0),
    limit: int | None = None,
) -> list[FusedHit]:
    """RRF fusion that reserves the head of the list for route quotas.

    The default ``dense_slots=3, sparse_slots=2`` reserves the first five
    positions for the top-3 dense and top-2 BM25 candidates (the shape used by
    the offline evaluation, so a strong lexical hit cannot be pushed out of the
    context window by five dense near-duplicates).  Reserved candidates are
    ordered among themselves by their RRF score, then the rest of the fused list
    fills the remaining positions in RRF order.
    """

    if dense_slots < 0 or sparse_slots < 0:
        raise ValueError("quota slots must be non-negative")
    k0_value, weight_values = _validate_fusion_options(k0, weights)
    dense = _ranked_ids(dense_ranked)
    sparse = _ranked_ids(sparse_ranked)
    fused = _rrf_scores(dense, sparse, k0=k0_value, weights=weight_values)

    reserved: dict[str, None] = {}
    for chunk_id in dense[:dense_slots]:
        reserved.setdefault(chunk_id, None)
    for chunk_id in sparse[:sparse_slots]:
        reserved.setdefault(chunk_id, None)

    head = sorted(
        ((chunk_id, fused[chunk_id]) for chunk_id in reserved),
        key=lambda item: (-item[1][0], item[0]),
    )
    tail = sorted(
        ((chunk_id, value) for chunk_id, value in fused.items() if chunk_id not in reserved),
        key=lambda item: (-item[1][0], item[0]),
    )
    hits = [
        FusedHit(chunk_id, score, tuple(routes))
        for chunk_id, (score, routes) in (*head, *tail)
    ]
    if limit is not None:
        if limit <= 0:
            return []
        return hits[:limit]
    return hits


__all__ = [
    "BM25_B",
    "BM25_K1",
    "RRF_K0",
    "BM25Hit",
    "BM25Index",
    "FusedHit",
    "fuse_routes",
    "fuse_with_quota",
    "tokenize",
]
