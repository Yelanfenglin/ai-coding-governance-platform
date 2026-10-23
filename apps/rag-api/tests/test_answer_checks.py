"""Tests for the lightweight answer/citation checks (P1c)."""

from __future__ import annotations

import pytest

from app.answer_checks import (
    AnswerFinding,
    check_addresses,
    check_answer,
    check_structure,
    check_symbols,
    summarise,
    symbol_mentions,
)
from app.evidence_checks import Citation

REGISTERS = frozenset({"PUA", "PUB", "INTCR0", "PWM1CR"})

CITATION_TEXT = """M8P5170 Version 1.06 > 6 端口 > 6.3 PUA 上拉使能寄存器
类型：寄存器定义
寄存器：PUA
位定义：
Bit7 PDA5
Bit6 PDA3
|  | Bit7 | Bit6 | Bit5 | Bit4 | Bit3 | Bit2 | Bit1 | Bit0 |
| PUA | PDA5 | PDA3 | PUA5 | PUA4 | PUA3 | PUA2 | PUA1 | PUA0 |
注：IOA5 上电后默认下拉。
中断向量地址为 0008H。
"""


def _citations(text: str = CITATION_TEXT, kind: str = "MIXED") -> list[Citation]:
    return [Citation(chunk_id="c1", text=text, section_path="6 端口 > 6.3 PUA", kind=kind)]


def test_symbols_present_in_citations_are_not_flagged() -> None:
    answer = "PUA 的 Bit7 是 PDA5，对应 IOA5 的下拉使能。"
    assert check_symbols(answer, _citations(), REGISTERS) == ()


def test_register_and_symbol_absent_from_citations_are_flagged() -> None:
    answer = "INTCR0 的 IOAIE 位使能 IOA4 中断。"
    findings = check_symbols(answer, _citations(), REGISTERS)
    kinds = summarise(findings)
    assert kinds.get("unsupported_register") == 1  # INTCR0
    assert kinds.get("unsupported_symbol") == 1  # IOA4 (IOAIE has no bit index)


def test_bit_symbol_absent_from_citations_is_flagged() -> None:
    answer = "该位对应 IOA5 的下拉使能。"
    findings = check_symbols(answer, _citations("寄存器：PUA\n"), REGISTERS)
    assert [f.value for f in findings] == ["IOA5"]


def test_catalog_registers_are_checked_even_without_a_bit_index() -> None:
    answer = "见 PWM1CR 的定义。"
    assert [f.kind for f in check_symbols(answer, _citations(), REGISTERS)] == [
        "unsupported_register"
    ]


def test_plain_words_are_not_treated_as_symbols() -> None:
    answer = "根据 GND 与 VDD 的连接方式，CPU 在复位后读取该位。"
    assert symbol_mentions(answer, REGISTERS) == ()


def test_address_present_and_absent() -> None:
    assert check_addresses("跳转到 0008H。", _citations()) == ()
    assert [f.value for f in check_addresses("跳转到 0009H。", _citations())] == ["0009H"]


def test_unbalanced_code_fence_is_flagged() -> None:
    answer = "示例：\n```c\nGOTO Main_Program;\n"
    findings = check_structure(answer, _citations())
    assert [f.kind for f in findings] == ["code_block_unbalanced"]


def test_balanced_code_fence_passes() -> None:
    answer = "示例：\n```c\nGOTO Main_Program;\n```\n"
    assert check_structure(answer, _citations()) == ()


def test_ragged_table_is_flagged() -> None:
    answer = "| Bit7 | Bit6 |\n| --- | --- |\n| PDA5 |\n"
    findings = check_structure(answer, _citations())
    assert [f.kind for f in findings] == ["table_ragged"]


def test_figure_reference_without_figure_citation_is_flagged() -> None:
    findings = check_structure("如图所示，外形尺寸见下。", _citations())
    assert [f.kind for f in findings] == ["image_without_figure_citation"]


def test_figure_reference_with_visual_citation_passes() -> None:
    visual = [Citation(chunk_id="c2", text="![图](snapshots/p071_s1_image.png)", kind="VISUAL")]
    assert check_structure("如图所示。", visual) == ()


def test_clean_answer_produces_no_findings() -> None:
    answer = "PUA 的 Bit7 是 PDA5，复位值为 1。中断向量在 0008H。"
    assert check_answer(answer, _citations(), registers=REGISTERS) == ()


def test_empty_answer_or_missing_citations_is_not_checked() -> None:
    assert check_answer("", _citations(), registers=REGISTERS) == ()
    assert check_answer("PUA Bit7 是 PDA5", [], registers=REGISTERS) == ()


@pytest.mark.parametrize("bad", [None, 3])
def test_non_string_answer_is_rejected(bad: object) -> None:
    with pytest.raises(TypeError):
        check_answer(bad, _citations())  # type: ignore[arg-type]


def test_findings_are_immutable_values() -> None:
    finding = AnswerFinding(kind="unsupported_register", value="PUA", detail="x")
    with pytest.raises(AttributeError):
        finding.kind = "other"  # type: ignore[misc]
