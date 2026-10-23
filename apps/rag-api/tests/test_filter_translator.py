"""Behavioural tests for the Milvus filter translator (pushdown v1)."""

from __future__ import annotations

import pytest

from app.filter_translator import (
    POST_ONLY_FIELDS,
    PUSHABLE_FIELDS,
    pushed_fields,
    translate_condition,
    translate_filters,
)
from app.query_filters import FilterCondition, parse_filter_conditions
from app.vectordb.milvus_store import scalar_filter_clause


def _package(operator: str, value) -> FilterCondition:
    return FilterCondition("package", operator, value)


def test_package_eq_uses_membership_and_canonicalises() -> None:
    assert translate_condition(_package("eq", "SOP20")) == 'array_contains(packages, "SOP20")'
    # The planner may emit the datasheet form; the stored array is canonical.
    assert translate_condition(_package("eq", "SOP-8")) == 'array_contains(packages, "SOP8")'
    assert translate_condition(_package("eq", "sot23-6")) == (
        'array_contains(packages, "SOT236")'
    )


def test_package_in_becomes_an_or_of_membership_checks() -> None:
    clause = translate_condition(_package("in", ("SOP16", "SOP20")))
    assert clause == (
        '(array_contains(packages, "SOP16") or array_contains(packages, "SOP20"))'
    )


def test_section_id_keeps_parent_and_descendants_with_a_dot_boundary() -> None:
    assert translate_condition(FilterCondition("section_id", "eq", "6.3")) == (
        '(section_id == "6.3" or section_id like "6.3.%")'
    )
    assert translate_condition(FilterCondition("section_id", "in", ("6.3", "6.4"))) == (
        '((section_id == "6.3" or section_id like "6.3.%") or '
        '(section_id == "6.4" or section_id like "6.4.%"))'
    )


@pytest.mark.parametrize("field", sorted(POST_ONLY_FIELDS))
def test_post_only_fields_are_never_pushed(field: str) -> None:
    condition = FilterCondition(field, "eq", "VALUE")
    assert translate_condition(condition) is None
    assert field not in PUSHABLE_FIELDS


def test_unusable_package_value_falls_back_to_post_filtering() -> None:
    oversized = FilterCondition("package", "eq", "S" + "1" * 40)
    assert translate_condition(oversized) is None


def test_translate_filters_splits_pushed_and_remaining() -> None:
    conditions = (
        FilterCondition("product", "eq", "M8P5170"),
        FilterCondition("package", "eq", "SOP20"),
        FilterCondition("register", "eq", "PWM1CR"),
        FilterCondition("section_id", "eq", "6.3"),
    )
    clause, remaining = translate_filters(conditions)
    assert clause == (
        'array_contains(packages, "SOP20") and '
        '(section_id == "6.3" or section_id like "6.3.%")'
    )
    assert remaining == (
        FilterCondition("product", "eq", "M8P5170"),
        FilterCondition("register", "eq", "PWM1CR"),
    )
    assert pushed_fields(conditions) == ("package", "section_id")


def test_nothing_pushable_returns_none_and_keeps_every_condition() -> None:
    conditions = (FilterCondition("register", "eq", "PWM1CR"),)
    assert translate_filters(conditions) == (None, conditions)


def test_parser_output_is_accepted_directly() -> None:
    conditions = parse_filter_conditions(
        [
            {"field": "package", "operator": "eq", "value": "SOP20"},
            {"field": "section_id", "operator": "in", "value": ["6.3", "6.4"]},
        ]
    )
    clause, remaining = translate_filters(conditions)
    assert clause == (
        'array_contains(packages, "SOP20") and '
        '((section_id == "6.3" or section_id like "6.3.%") or '
        '(section_id == "6.4" or section_id like "6.4.%"))'
    )
    assert remaining == ()


@pytest.mark.parametrize("value", ['6.3" or 1==1', "2.3%", "2.3_", "端口", "6..3", "6.3."])
def test_invalid_section_values_are_not_pushed(value: str) -> None:
    condition = FilterCondition("section_id", "eq", value)
    assert translate_condition(condition) is None
    assert translate_filters((condition,)) == (None, (condition,))


def test_section_in_preserves_grouping_when_combined_with_another_condition() -> None:
    conditions = (
        FilterCondition("section_id", "in", ("2.2", "2.3")),
        _package("eq", "SOP20"),
    )
    clause, remaining = translate_filters(conditions)
    assert clause == (
        '((section_id == "2.2" or section_id like "2.2.%") or '
        '(section_id == "2.3" or section_id like "2.3.%")) '
        'and array_contains(packages, "SOP20")'
    )
    assert remaining == ()


def test_store_refuses_pushdown_on_the_legacy_collection() -> None:
    conditions = (_package("eq", "SOP20"),)
    assert scalar_filter_clause(conditions, "rag_specs") is None
    assert scalar_filter_clause(conditions, "rag_specs_v2") == (
        'array_contains(packages, "SOP20")'
    )
    assert scalar_filter_clause((), "rag_specs_v2") is None


def test_store_refuses_pushdown_when_nothing_is_pushable() -> None:
    conditions = (FilterCondition("register", "eq", "PWM1CR"),)
    assert scalar_filter_clause(conditions, "rag_specs_v2") is None
