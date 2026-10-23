"""Local catalog validation for planner-produced metadata filters."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .query_filters import FilterCondition, QueryFilterError
from .schemas import normalize_version


def _section_matches(value: str, section: str) -> bool:
    pattern = rf"(?<![\d.]){re.escape(value)}(?![\d])"
    return re.search(pattern, section, flags=re.IGNORECASE) is not None


@dataclass(frozen=True, slots=True)
class QueryCatalog:
    products: frozenset[str]
    versions: frozenset[str]
    registers: frozenset[str]
    sections: tuple[str, ...]
    packages: frozenset[str]

    @classmethod
    def from_mapping(cls, payload: dict[str, object]) -> QueryCatalog:
        def values(key: str) -> tuple[str, ...]:
            raw = payload.get(key) or []
            return tuple(str(item) for item in raw if str(item).strip())

        return cls(
            products=frozenset(value.upper() for value in values("products")),
            versions=frozenset(normalize_version(value) for value in values("versions")),
            registers=frozenset(value.upper() for value in values("registers")),
            sections=values("sections"),
            packages=frozenset(value.upper() for value in values("packages")),
        )

    @classmethod
    def load(cls, path: Path) -> QueryCatalog:
        return cls.from_mapping(json.loads(path.read_text(encoding="utf-8")))

    def validate(self, filters: tuple[FilterCondition, ...]) -> tuple[FilterCondition, ...]:
        for condition in filters:
            if condition.field == "product":
                unknown = {value.upper() for value in condition.values} - self.products
            elif condition.field == "version":
                unknown = {
                    normalize_version(value) for value in condition.values
                } - self.versions
            elif condition.field == "register":
                unknown = {value.upper() for value in condition.values} - self.registers
            elif condition.field == "package":
                unknown = {value.upper() for value in condition.values} - self.packages
            elif condition.field == "section_id":
                unknown = {
                    value
                    for value in condition.values
                    if not re.fullmatch(r"\d+(?:\.\d+)*", value)
                    or not any(_section_matches(value, section) for section in self.sections)
                }
            else:  # pragma: no cover - parser rejects unknown fields
                unknown = set()
            if unknown:
                raise QueryFilterError(
                    f"filter value(s) not in catalog for {condition.field}: {sorted(unknown)}"
                )
        return filters
