"""Translate typed planner filters into a Milvus boolean expression.

This is the only place that turns ``FilterCondition`` values into Milvus
expression syntax.  Nothing here accepts a caller-supplied expression string.

Which fields may be pushed down (contract v1)
---------------------------------------------
``package``
    Pushed as ``array_contains(packages, "<canonical>")``.  Exactly the membership
    semantics the contract defines, so a pushed filter never drops a row the
    post-retrieval check would have kept: the stored set is derived from
    ``text + embedding_text``, which is a superset of the ``hit.text`` the
    post-filter scans.

``section_id``
    Matches the heading and its descendants: ``section_id == "2.3" or
    section_id like "2.3.%"``.  The dot keeps sibling headings such as ``2.30``
    outside the scope.  ``in`` is an OR of these parent/descendant groups.

Deliberately NOT pushed
-----------------------
``register``
    The stored scalar is chunk-level metadata (empty for most table/figure
    chunks), while the post-filter also accepts a textual mention anywhere in the
    chunk.  Pushing ``register == X`` would therefore drop chunks the current
    behaviour keeps — a silent recall loss.  It stays a post-retrieval filter.

``product`` / ``version``
    Scope is already resolved through PostgreSQL into ``document_id``s, and
    ``version`` needs ``normalize_version`` on both sides, which an equality
    expression cannot express.  Pushing the raw planner value could disagree with
    the resolved scope.

Anything unparseable is simply left for the post-filter: translation degrades to
"no clause", never to "wrong clause".
"""

from __future__ import annotations

import json
import re

from .query_filters import FilterCondition
from .scalar_metadata import PACKAGE_TOKEN_MAX_CHARS, canonical_token

PUSHABLE_FIELDS: frozenset[str] = frozenset({"package", "section_id"})
POST_ONLY_FIELDS: frozenset[str] = frozenset({"register", "product", "version"})

_ALWAYS_TRUE = None


def _literal(value: str) -> str:
    """Render a Milvus string literal; JSON escaping is also valid Milvus syntax."""

    return json.dumps(value, ensure_ascii=False)


def _package_clause(condition: FilterCondition) -> str | None:
    tokens: list[str] = []
    for value in condition.values:
        token = canonical_token(value)
        if not token or len(token) > PACKAGE_TOKEN_MAX_CHARS:
            # Cannot be a stored code; let the post-filter make the call.
            return None
        if token not in tokens:
            tokens.append(token)
    if not tokens:
        return None
    membership = [f"array_contains(packages, {_literal(token)})" for token in tokens]
    return membership[0] if len(membership) == 1 else "(" + " or ".join(membership) + ")"


def _section_clause(condition: FilterCondition) -> str | None:
    values = [value.strip() for value in condition.values if value.strip()]
    if not values or any(re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", value) is None for value in values):
        # LIKE wildcard characters are not valid section numbers.
        return None
    clauses = [
        f"(section_id == {_literal(value)} or section_id like {_literal(value + '.%')})"
        for value in dict.fromkeys(values)
    ]
    return clauses[0] if len(clauses) == 1 else "(" + " or ".join(clauses) + ")"


def translate_condition(condition: FilterCondition) -> str | None:
    """Return one Milvus clause, or ``None`` when the condition stays post-only."""

    if not isinstance(condition, FilterCondition):
        raise TypeError("condition must be a FilterCondition")
    if condition.field == "package":
        return _package_clause(condition)
    if condition.field == "section_id":
        return _section_clause(condition)
    return _ALWAYS_TRUE


def translate_filters(
    conditions: tuple[FilterCondition, ...] | list[FilterCondition],
) -> tuple[str | None, tuple[FilterCondition, ...]]:
    """Split filters into (pushed clause, conditions left for post-filtering).

    The returned clause is ``None`` when nothing can be pushed; callers must then
    behave exactly as before.
    """

    if isinstance(conditions, (str, bytes)) or not isinstance(conditions, (tuple, list)):
        raise TypeError("conditions must be a sequence of FilterCondition values")

    clauses: list[str] = []
    remaining: list[FilterCondition] = []
    for condition in conditions:
        clause = translate_condition(condition)
        if clause is None:
            remaining.append(condition)
        else:
            clauses.append(clause)

    if not clauses:
        return None, tuple(conditions)
    return " and ".join(clauses), tuple(remaining)


def pushed_fields(
    conditions: tuple[FilterCondition, ...] | list[FilterCondition],
) -> tuple[str, ...]:
    """Field names that would be pushed, for logging/diagnostics."""

    pushed = {
        condition.field
        for condition in conditions
        if translate_condition(condition) is not None
    }
    return tuple(sorted(pushed))
