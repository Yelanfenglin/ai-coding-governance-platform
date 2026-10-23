"""Deterministic filter extraction from explicit question mentions.

This is the cheap, reproducible arm of the filter A/B: no LLM, no planner, no
tokens.  It only emits a filter when the question *names* a value the catalog
knows, so an unknown or ambiguous mention degrades to "no filter" instead of
narrowing the search on a guess.

Rules (v1)
----------
``package``
    Emitted only when the question also carries a packaging cue (封装 / 外形 / 引脚 /
    脚 / 焊盘 / 尺寸 / 俯视 / 间距).  A bare code inside an unrelated sentence is not
    enough.  Codes come from the same extractor the collection was built with and
    must exist in the catalog.

``register``
    Upper-case mnemonic tokens that exist in the catalog's register list.  Pin
    names (GND / VDD) are not registers and are ignored by construction.

``section_id``
    Only the explicit "第 6.3 节" / "6.3 节" form.  A bare "6.3" is ignored: too easy
    to confuse with a timing value or a pin number.

Negation
--------
A code directly after 别用 / 不要用 / 而非 / 排除 etc. is dropped:
"SSOP24 封装的第 13 脚…别用 SOP28 的编号" filters on SSOP24 only.

Multiple values in one field become ``in`` (OR), never ``and``: one chunk rarely
carries every package a question names, and AND would drop gold evidence.

Cross-product questions emit nothing
------------------------------------
A question that names two or more products is a comparison, and its evidence
spans both documents.  A register or section that exists in one product does not
exist in the other, so any such filter silently deletes half the required
evidence.  Measured on the development set: without this guard, "M8P515 的 OPTION
寄存器中 GIE 位与 M8P613 的中断向量地址" lost one of its two evidence groups for
7 questions; with the guard they are untouched.
"""

from __future__ import annotations

import re

from .query_catalog import QueryCatalog
from .query_filters import FilterCondition
from .scalar_metadata import iter_package_matches

PACKAGE_CUES: tuple[str, ...] = (
    "封装",
    "外形",
    "引脚",
    "管脚",
    "脚",
    "焊盘",
    "尺寸",
    "俯视",
    "间距",
)

NEGATION_CUES: tuple[str, ...] = ("别用", "不要用", "不用", "而非", "排除", "不是", "忽略")
NEGATION_WINDOW = 6

_SECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"第\s*(\d+(?:\.\d+)+)\s*节"),
    re.compile(r"(\d+(?:\.\d+)+)\s*节"),
)
_REGISTER_TOKEN = re.compile(r"(?<![A-Za-z0-9])([A-Z][A-Z0-9]{2,9})(?![A-Za-z0-9])")


def _negated(question: str, start: int) -> bool:
    return any(cue in question[max(0, start - NEGATION_WINDOW) : start] for cue in NEGATION_CUES)


def product_mentions(question: str, catalog: QueryCatalog) -> tuple[str, ...]:
    """Products named in the question, in catalog order, de-duplicated."""

    found: list[str] = []
    for product in sorted(catalog.products):
        pattern = rf"(?<![A-Za-z0-9]){re.escape(product)}(?![A-Za-z0-9])"
        if re.search(pattern, question, re.IGNORECASE):
            found.append(product)
    return tuple(found)


def _condition(field: str, values: list[str]) -> FilterCondition | None:
    if not values:
        return None
    if len(values) == 1:
        return FilterCondition(field, "eq", values[0])  # type: ignore[arg-type]
    return FilterCondition(field, "in", tuple(values))  # type: ignore[arg-type]


def _package_condition(question: str, catalog: QueryCatalog) -> FilterCondition | None:
    if not any(cue in question for cue in PACKAGE_CUES):
        return None
    tokens = [
        token
        for token, start in iter_package_matches(question, max_tokens=None)
        if token in catalog.packages and not _negated(question, start)
    ]
    return _condition("package", tokens)


def _register_condition(question: str, catalog: QueryCatalog) -> FilterCondition | None:
    tokens: list[str] = []
    for match in _REGISTER_TOKEN.finditer(question):
        token = match.group(1)
        if token not in catalog.registers or _negated(question, match.start()):
            continue
        if token not in tokens:
            tokens.append(token)
    return _condition("register", tokens)


def _section_condition(question: str, catalog: QueryCatalog) -> FilterCondition | None:
    sections: list[str] = []
    for pattern in _SECTION_PATTERNS:
        for match in pattern.finditer(question):
            value = match.group(1)
            if value in sections:
                continue
            if not any(
                re.search(rf"(?<![\d.]){re.escape(value)}(?![\d])", section)
                for section in catalog.sections
            ):
                continue
            sections.append(value)
    return _condition("section_id", sections)


def extract_rule_filters(question: str, catalog: QueryCatalog) -> tuple[FilterCondition, ...]:
    """Return the filters an explicit-mention extractor would emit."""

    if not isinstance(question, str):
        raise TypeError("question must be a string")
    if not isinstance(catalog, QueryCatalog):
        raise TypeError("catalog must be a QueryCatalog")
    if len(product_mentions(question, catalog)) >= 2:
        # Cross-product comparison: metadata filters are single-document by nature.
        return ()
    conditions = (
        _package_condition(question, catalog),
        _register_condition(question, catalog),
        _section_condition(question, catalog),
    )
    return tuple(condition for condition in conditions if condition is not None)
