"""Tests for the deterministic register-block / conflict checks (P1b)."""

from __future__ import annotations

from app.evidence_checks import (
    Citation,
    detect_findings,
    parse_register_block,
    summarise,
)


def _citation(chunk_id: str, text: str, section_path: str = "") -> Citation:
    return Citation(chunk_id=chunk_id, text=text, section_path=section_path)

M8P6272_MIXED = """M8P6272 Version 1.10 > 6 端口 > 6.3 端口上下拉控制
类型：寄存器定义
寄存器：PUA
位定义：
Bit7 PUA7
Bit6 PUA6
6.3 端口上下拉控制

IOA 上拉使能寄存器

|  | Bit7 | Bit6 | Bit5 | Bit4 | Bit3 | Bit2 | Bit1 | Bit0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PUA | PUA7 | PUA6 | PUA5 | PUA4 | PUA3 | PUA2 | PUA1 | PUA0 |
| 复位后 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

IOB 上拉使能寄存器

|  | Bit7 | Bit6 | Bit5 | Bit4 | Bit3 | Bit2 | Bit1 | Bit0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PUB | PDA2 | PDA0 | PUB5 | PUB4 | PUB3 | PUB2 | PUB1 | PUB0 |
| 复位后 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
"""

M8P5170_PUA = """M8P5170 Version 1.06 > 6 端口 > 6.3 PUA 上拉使能寄存器
类型：寄存器定义
寄存器：PUA
位定义：
Bit7 PDA5
Bit6 PDA3
|  | Bit7 | Bit6 | Bit5 | Bit4 | Bit3 | Bit2 | Bit1 | Bit0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PUA | PDA5 | PDA3 | PUA5 | PUA4 | PUA3 | PUA2 | PUA1 | PUA0 |
| 复位后 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
"""

CLEAN_TIMER = """M8P5170 Version 1.06 > 7 定时器0(TC0) > 7.7 PWM0xD 数据位(x =0,1,2,3)
类型：寄存器定义
寄存器：PWM0xD
位定义：
Bit7 PWM0xD7
Bit0 PWM0xD0
|  | Bit7 | Bit6 | Bit5 | Bit4 | Bit3 | Bit2 | Bit1 | Bit0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PWM0xD | PWM0xD7 | PWM0xD6 | PWM0xD5 | PWM0xD4 | PWM0xD3 | PWM0xD2 | PWM0xD1 | PWM0xD0 |
"""


def test_parses_declared_register_bits_reset_and_tables() -> None:
    block = parse_register_block(M8P5170_PUA)
    assert block is not None
    assert block.product == "M8P5170"
    assert block.register == "PUA"
    assert block.bits[7] == "PDA5"
    assert block.bits[5] == "PUA5"
    assert block.reset[7] == "1"
    assert block.table_registers == ("PUA",)


def test_non_register_chunk_is_ignored() -> None:
    prose = "5.1 概述\nM8P5170有多路中断源: TC0定时器，TC1定时器，IO变化中断。"
    assert parse_register_block(prose) is None


def test_mixed_register_chunk_is_flagged_as_ambiguous_scope() -> None:
    findings = detect_findings([_citation("6272", M8P6272_MIXED)])
    kinds = summarise(findings)
    assert kinds.get("ambiguous_scope") == 1
    assert "PUB" in findings[0].detail
    # Nothing contradicts itself, so this must not be reported as a hard conflict.
    assert kinds.get("conflict") is None


def test_same_product_register_bit_with_different_symbol_is_a_conflict() -> None:
    conflicting = M8P5170_PUA.replace("Bit7 PDA5", "Bit7 PUA7").replace(
        "| PUA | PDA5 |", "| PUA | PUA7 |"
    )
    findings = detect_findings(
        [
            _citation("a", M8P5170_PUA, "6 端口 > 6.3 PUA 上拉使能寄存器"),
            _citation("b", conflicting, "6 端口 > 6.3 PUA 上拉使能寄存器"),
        ]
    )
    conflicts = [finding for finding in findings if finding.kind == "conflict"]
    assert conflicts, "expected a hard conflict"
    assert conflicts[0].bit == 7 and conflicts[0].register == "PUA"


def test_same_register_different_products_are_not_conflicts() -> None:
    """PUA.bit7 legitimately differs between products; that is not a contradiction."""

    findings = detect_findings(
        [_citation("5170", M8P5170_PUA), _citation("6272", M8P6272_MIXED)]
    )
    assert [finding for finding in findings if finding.kind == "conflict"] == []


def test_reset_value_conflict_is_detected() -> None:
    other = M8P5170_PUA.replace("| 复位后 | 1 |", "| 复位后 | 0 |")
    findings = detect_findings(
        [
            _citation("a", M8P5170_PUA, "6 端口 > 6.3 PUA 上拉使能寄存器"),
            _citation("b", other, "6 端口 > 6.3 PUA 上拉使能寄存器"),
        ]
    )
    assert any("resets to" in finding.detail for finding in findings)


def test_clean_single_register_chunk_produces_nothing() -> None:
    assert detect_findings([_citation("timer", CLEAN_TIMER)]) == ()


def test_different_sections_become_variance_not_conflict() -> None:
    """ADH has a different layout per alignment mode; that is not a contradiction."""

    left = M8P5170_PUA.replace("| 复位后 | 1 |", "| 复位后 | x |")
    findings = detect_findings(
        [
            _citation("left", M8P5170_PUA, "10 模数转换器 > 10.5 ADH/ADL > 10.5.1 左对齐"),
            _citation("right", left, "10 模数转换器 > 10.5 ADH/ADL > 10.5.2 右对齐"),
        ]
    )
    kinds = summarise(findings)
    assert kinds.get("conflict") is None
    assert kinds.get("cross_section_variance")
