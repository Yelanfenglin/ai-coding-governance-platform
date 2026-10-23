"""Typed metadata filters for self-query style retrieval.

This module is the contract layer between the LLM query planner and retrieval.
It deliberately only accepts a small, known field/operator vocabulary; callers
must validate values against the live catalog before pushing them into a store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FilterField = Literal["product", "version", "register", "section_id", "package"]
FilterOperator = Literal["eq", "in"]

ALLOWED_FIELDS: frozenset[str] = frozenset(
    {"product", "version", "register", "section_id", "package"}
)
ALLOWED_OPERATORS: frozenset[str] = frozenset({"eq", "in"})


class QueryFilterError(ValueError):
    """Raised when a planner-produced filter is malformed."""


@dataclass(frozen=True, slots=True)
class FilterCondition:
    field: FilterField
    operator: FilterOperator
    value: str | tuple[str, ...]

    @property
    def values(self) -> tuple[str, ...]:
        return (self.value,) if isinstance(self.value, str) else self.value


def parse_filter_conditions(payload: object) -> tuple[FilterCondition, ...]:
    if payload is None:
        return ()
    if not isinstance(payload, list):
        raise QueryFilterError("filters must be a list")
    conditions: list[FilterCondition] = []
    for raw in payload:
        if not isinstance(raw, dict):
            raise QueryFilterError("each filter must be an object")
        field = str(raw.get("field") or "").strip().lower()
        operator = str(raw.get("operator") or "").strip().lower()
        value = raw.get("value")
        if field not in ALLOWED_FIELDS:
            raise QueryFilterError(f"unsupported filter field: {field!r}")
        if operator not in ALLOWED_OPERATORS:
            raise QueryFilterError(f"unsupported filter operator: {operator!r}")
        if operator == "eq":
            if not isinstance(value, str) or not value.strip():
                raise QueryFilterError("eq filter value must be a non-empty string")
            normalized: str | tuple[str, ...] = value.strip()
        else:
            if not isinstance(value, list) or not value:
                raise QueryFilterError("in filter value must be a non-empty list")
            values = tuple(
                str(item).strip()
                for item in value
                if isinstance(item, str) and str(item).strip()
            )
            if not values:
                raise QueryFilterError("in filter value must contain non-empty strings")
            normalized = values
        conditions.append(
            FilterCondition(field=field, operator=operator, value=normalized)  # type: ignore[arg-type]
        )
    return tuple(conditions)
