"""Canonical scalar metadata contract for the v2 Milvus collection.

This module is the single source of truth for the three additive scalar fields
(``section_id`` / ``register`` / ``package``).  Both writing paths must use it:

* offline shadow backfill (``output/**/backfill_v2_sample.py`` and its full
  corpus successor);
* live ingestion (``IngestPipeline.embed``).

Contract version 2
------------------
``section_id``
    Trimmed string.  Taken from ``chunk_metadata["section_id"]`` when present,
    otherwise derived from the first numeric heading path segment
    (``"5 端口 > 5.3 PUA"`` -> ``"5.3"``).  Empty string when neither exists.

``register``
    Upper-cased, whitespace-collapsed register mnemonic
    (``"pwm1cr "`` -> ``"PWM1CR"``).  One register per chunk: a register chunk
    for PWM1CR does not also answer PWM2CR.  Empty string when the chunk is not
    register-scoped.

``packages``
    A *set* of normalised package codes, stored as ``ARRAY<VARCHAR>``
    (``max_capacity=8``, element ``max_length=32``).  Order is first appearance
    in the chunk text, de-duplicated afterwards.

    Matching is *membership*, never substring: ``array_contains(packages, "SOP20")``
    matches a chunk that lists ``SOP16 / SOP20 / TSSOP20`` but does not match
    ``"SOP2"`` or ``"TSSOP20"``.  Measured on Milvus 2.6.21: a delimited VARCHAR
    cannot express this — bare infix ``like "%SOP2%"`` also matched ``SSOP28``.

    Canonicalisation strips every non-alphanumeric character and upper-cases:
    ``"SOP-8"``/``"sop 8"`` -> ``SOP8``, ``"SOT23-6"`` -> ``SOT236``.  This keeps
    the ``SOT23-6`` datasheet suffix comparable with a planner that emits
    ``SOT236``.

    Overflow drops whole trailing tokens instead of truncating a token, so a
    stored value never contains a half package code.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Final

SCALAR_METADATA_CONTRACT: Final[str] = "v2"

# Physical limits of the ``packages ARRAY<VARCHAR>`` field in rag_specs_v2.
PACKAGE_FIELD_MAX_COUNT: Final[int] = 8
PACKAGE_TOKEN_MAX_CHARS: Final[int] = 32
REGISTER_MAX_CHARS: Final[int] = 256
SECTION_ID_MAX_CHARS: Final[int] = 64
PACKAGE_SEPARATOR: Final[str] = ","

# Frozen vocabulary.  Longest-first ordering matters: regex alternation is
# tried left to right, so "TSSOP" must be attempted before "SOP"/"SSOP".
_PACKAGE_BASES: Final[tuple[str, ...]] = (
    "WLCSP",
    "VFBGA",
    "TSSOP",
    "LQFP",
    "TQFP",
    "UQFN",
    "SSOP",
    "MSOP",
    "QSOP",
    "SOIC",
    "FBGA",
    "PDIP",
    "SOP",
    "DFN",
    "QFN",
    "SOT",
    "BGA",
    "LGA",
    "DIP",
)

# A base plus an optional body such as "16", "23-6", "-8", "20".  Both sides are
# anchored on non-alphanumerics so "SSOP28" never yields a "SOP28" match and
# "M8P515" never yields anything at all.  A space may separate the base from the
# number but never join a second numeric group, so "SOP8 1.27mm" is SOP8 and not
# "SOP81"; only a hyphen joins groups ("SOT23-6").
_PACKAGE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    + "|".join(_PACKAGE_BASES)
    + r")(?:[-\s]?\d{1,4}(?:-\d{1,3})?)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_NON_ALNUM: Final[re.Pattern[str]] = re.compile(r"[^A-Z0-9]+")
_SECTION_NUMBER: Final[re.Pattern[str]] = re.compile(r"(?<![\d.])(\d+(?:\.\d+)*)(?![\d])")
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


def canonical_token(value: object) -> str:
    """Upper-case and strip every non-alphanumeric character."""

    if value is None:
        return ""
    return _NON_ALNUM.sub("", str(value).upper())


def iter_package_matches(
    text: str | None, *, max_tokens: int | None = PACKAGE_FIELD_MAX_COUNT
) -> tuple[tuple[str, int], ...]:
    """Canonical package code plus its start offset, in first-appearance order.

    Callers that need to reason about *where* a code came from (negation cues,
    highlighting) use this; ``extract_packages`` is the token-only wrapper.
    """

    if not text:
        return ()
    seen: dict[str, int] = {}
    for match in _PACKAGE_PATTERN.finditer(str(text)):
        token = canonical_token(match.group(0))
        if not token or token in seen or len(token) > PACKAGE_TOKEN_MAX_CHARS:
            continue
        seen[token] = match.start()
        if max_tokens is not None and len(seen) >= max_tokens:
            break
    return tuple(seen.items())


def extract_packages(
    text: str | None, *, max_tokens: int | None = PACKAGE_FIELD_MAX_COUNT
) -> tuple[str, ...]:
    """Return canonical package codes in first-appearance order, de-duplicated.

    ``max_tokens=None`` disables the field capacity (used for matching, where
    nothing is persisted).  A code longer than the ARRAY element limit is dropped
    whole; it is never truncated into a different, plausible-looking code.
    """

    return tuple(token for token, _ in iter_package_matches(text, max_tokens=max_tokens))


def packages_field_value(packages: str | Iterable[object] | None) -> str:
    """Render a package set as the canonical display string.

    The stored form is ``ARRAY<VARCHAR>``; this comma form is only for reports,
    logs and the legacy 2026-10-19 dump.  Never persist it.
    """

    return PACKAGE_SEPARATOR.join(parse_package_field(packages))


def parse_package_field(value: object) -> tuple[str, ...]:
    """Read a stored delimited value (or a sequence) back into canonical tokens."""

    if value is None:
        return ()
    if isinstance(value, str):
        parts: Iterable[object] = value.split(PACKAGE_SEPARATOR)
    elif isinstance(value, Iterable):
        parts = value
    else:
        return ()
    tokens: dict[str, None] = {}
    for part in parts:
        token = canonical_token(part)
        if token and token not in tokens:
            tokens[token] = None
    return tuple(tokens)


def package_matches(stored: object, wanted: object) -> bool:
    """Membership test between a stored package value and one wanted code."""

    target = canonical_token(wanted)
    if not target:
        return False
    return target in parse_package_field(stored)


def package_mentioned_in(text: str | None, wanted: object) -> bool:
    """True when ``wanted`` appears in free text as a complete package code.

    Same vocabulary and canonicalisation as extraction, so the post-retrieval
    fallback cannot disagree with what a backfill would have stored.  Extraction
    here is unbounded on purpose: matching must not inherit the 64-char budget.
    """

    target = canonical_token(wanted)
    if not target or not text:
        return False
    return target in extract_packages(text, max_tokens=None)


def normalize_register(value: object) -> str:
    """Upper-case and collapse whitespace for a single register mnemonic."""

    if value is None:
        return ""
    text = _WHITESPACE.sub("", str(value)).upper()
    return text[:REGISTER_MAX_CHARS]


def section_id_from_path(section_path: object) -> str:
    """Derive a numeric section id from a heading path, or return ``""``.

    A dotted heading wins over a bare chapter number: ``"8 定时器 > 8.7 PWM1CR"``
    resolves to ``8.7``, not ``8``.  When nothing is dotted the deepest number is
    used, which is the closest heading to the chunk.
    """

    if not section_path:
        return ""
    candidates = _SECTION_NUMBER.findall(str(section_path))
    if not candidates:
        return ""
    dotted = [candidate for candidate in candidates if "." in candidate]
    chosen = dotted[-1] if dotted else candidates[-1]
    return chosen[:SECTION_ID_MAX_CHARS]


def normalize_section_id(value: object, *, section_path: object = "") -> str:
    """Prefer an explicit section id, then fall back to the heading path."""

    text = str(value).strip() if value is not None else ""
    if text:
        return text[:SECTION_ID_MAX_CHARS]
    return section_id_from_path(section_path)


@dataclass(frozen=True, slots=True)
class ScalarMetadata:
    """The three additive scalar fields for one chunk."""

    section_id: str = ""
    register: str = ""
    packages: tuple[str, ...] = field(default_factory=tuple)

    @property
    def package_display(self) -> str:
        """Comma form for logs and reports; not the stored representation."""

        return packages_field_value(self.packages)

    def as_fields(self) -> dict[str, object]:
        """Values as written to the collection (packages stays a list)."""

        return {
            "section_id": self.section_id,
            "register": self.register,
            "packages": list(self.packages),
        }


def scalar_metadata(
    *,
    chunk_metadata: Mapping[str, object] | None = None,
    text: str = "",
    extra_text: str = "",
    section_path: str = "",
) -> ScalarMetadata:
    """Build the canonical scalar metadata for one chunk.

    ``chunk_metadata`` is the persisted chunk metadata produced by the chunker,
    which already carries ``section_id`` and ``register``.  Package codes are not
    stored there; they are derived from the chunk text itself so both writing
    paths see exactly the same evidence.
    """

    meta = chunk_metadata or {}
    meta_path = meta.get("section_path")
    if section_path:
        resolved_path = section_path
    elif isinstance(meta_path, (list, tuple)):
        resolved_path = " > ".join(str(part) for part in meta_path)
    else:
        resolved_path = str(meta_path or "")
    evidence = "\n".join(part for part in (text, extra_text) if part)
    return ScalarMetadata(
        section_id=normalize_section_id(meta.get("section_id"), section_path=resolved_path),
        register=normalize_register(meta.get("register")),
        packages=extract_packages(evidence),
    )
