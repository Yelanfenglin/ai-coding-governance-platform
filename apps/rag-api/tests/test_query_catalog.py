from __future__ import annotations

import pytest

from app.query_catalog import QueryCatalog
from app.query_filters import FilterCondition, QueryFilterError


def _catalog() -> QueryCatalog:
    return QueryCatalog.from_mapping(
        {
            "products": ["M8P5170"],
            "versions": ["V1.6"],
            "registers": ["PUA", "PUB"],
            "sections": ["6 端口 > 6.3 PUA 上拉使能寄存器"],
            "packages": ["SOP16"],
        }
    )


def test_catalog_accepts_known_values() -> None:
    filters = (
        FilterCondition("product", "eq", "M8P5170"),
        FilterCondition("version", "eq", "V1.6"),
        FilterCondition("register", "in", ("PUA", "PUB")),
        FilterCondition("package", "eq", "SOP16"),
        FilterCondition("section_id", "eq", "6.3"),
    )
    assert _catalog().validate(filters) == filters


def test_catalog_rejects_unknown_register() -> None:
    with pytest.raises(QueryFilterError):
        _catalog().validate((FilterCondition("register", "eq", "NOT_A_REGISTER"),))


def test_catalog_normalizes_version_values() -> None:
    filters = _catalog().validate((FilterCondition("version", "eq", "V1.06"),))
    assert filters == (FilterCondition("version", "eq", "V1.06"),)


def test_catalog_rejects_unknown_package() -> None:
    with pytest.raises(QueryFilterError):
        _catalog().validate((FilterCondition("package", "eq", "BGA999"),))
