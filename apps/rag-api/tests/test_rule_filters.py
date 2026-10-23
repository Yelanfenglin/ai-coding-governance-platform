"""Tests for the deterministic filter extractor (no LLM, catalog-validated)."""

from __future__ import annotations

import pytest

from app.query_catalog import QueryCatalog
from app.rule_filters import NEGATION_WINDOW, extract_rule_filters


def _catalog() -> QueryCatalog:
    return QueryCatalog(
        products=frozenset({"M8P515", "M8P632"}),
        versions=frozenset({"V2.3"}),
        registers=frozenset({"PWM1CR", "PUA", "IOACR"}),
        sections=(
            "1 产品简述 > 1.2 引脚图",
            "6 端口 > 6.3 PUA",
            "19 封装信息 > 19.1 SOP16",
        ),
        packages=frozenset({"SOP16", "SOP8", "SSOP24", "SOP28", "DIP8"}),
    )


def _fields(question: str) -> dict[str, object]:
    return {
        condition.field: condition.value
        for condition in extract_rule_filters(question, _catalog())
    }


def test_package_requires_a_packaging_cue() -> None:
    assert _fields("M8P515：SOP16 的引脚间距是多少？")["package"] == "SOP16"
    # Same code, no packaging cue: not a packaging question, so no filter.
    assert "package" not in _fields("M8P515：SOP16 这个编号出现在哪一节？")


def test_codes_are_canonicalised_before_matching_the_catalog() -> None:
    assert _fields("M8P515：SOP-8 封装的引脚 2 是什么功能？")["package"] == "SOP8"


def test_multiple_packages_become_in_not_and() -> None:
    fields = _fields("M8P515：DIP8/SOP8 封装中，引脚 2 和引脚 7 分别是什么？")
    assert fields["package"] == ("DIP8", "SOP8")


def test_negated_code_is_dropped() -> None:
    question = "M8P634：SSOP24 封装的第 13 脚标了哪个 PWM 复用名？别用 SOP28 的编号。"
    assert _fields(question)["package"] == "SSOP24"


def test_unknown_code_is_ignored_rather_than_guessed() -> None:
    assert "package" not in _fields("M8P515：QFN99 封装的引脚定义是什么？")


def test_register_only_fires_for_catalog_registers() -> None:
    assert _fields("M8P515：PWM1CR 的 bit7 是什么？")["register"] == "PWM1CR"
    # GND/VDD are pin names, not registers.
    assert "register" not in _fields("M8P515：SSOP24 封装里 GND 和 VDD 在第几脚？")


def test_section_requires_the_explicit_form() -> None:
    assert _fields("M8P515：第 6.3 节讲了什么？")["section_id"] == "6.3"
    assert "section_id" not in _fields("M8P515：6.3 V 的阈值是多少？")
    assert "section_id" not in _fields("M8P515：第 9.9 节讲了什么？")


def test_multiple_sections_become_in() -> None:
    assert _fields("M8P515：第 6.3 节和第 1.2 节各讲什么？")["section_id"] == ("6.3", "1.2")


def test_negation_window_is_bounded() -> None:
    far = "别用" + "x" * (NEGATION_WINDOW + 1) + " SOP28 封装的编号"
    assert _fields(far)["package"] == "SOP28"


def test_question_without_mentions_produces_no_filters() -> None:
    assert extract_rule_filters("M8P515：这个芯片掉电后数据会丢吗？", _catalog()) == ()


@pytest.mark.parametrize("bad", [None, 3])
def test_non_string_question_is_rejected(bad: object) -> None:
    with pytest.raises(TypeError):
        extract_rule_filters(bad, _catalog())  # type: ignore[arg-type]


def _catalog_with_option() -> QueryCatalog:
    base = _catalog()
    return QueryCatalog(
        products=base.products,
        versions=base.versions,
        registers=base.registers | {"OPTION"},
        sections=base.sections,
        packages=base.packages,
    )


def test_cross_product_question_emits_no_filters() -> None:
    """A comparison needs both documents; a single-product filter would delete one."""

    catalog = _catalog_with_option()
    cross = "M8P515 的 OPTION 寄存器中 GIE 位与 M8P632 的中断向量地址分别是什么？请并列说明。"
    assert extract_rule_filters(cross, catalog) == ()
    # Same register mention, only one product: the filter is emitted.
    single = "M8P515 的 OPTION 寄存器中 GIE 位是什么？"
    assert _fields_for(single, catalog)["register"] == "OPTION"


def _fields_for(question: str, catalog: QueryCatalog) -> dict[str, object]:
    return {
        condition.field: condition.value
        for condition in extract_rule_filters(question, catalog)
    }
