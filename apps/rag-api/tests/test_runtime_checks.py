from __future__ import annotations

from uuid import uuid4

from app.rag_service import RagQueryService
from app.schemas import Citation, KnowledgeChunkKind

PUA_MIXED = """M8P6272 Version 1.10 > 6 端口 > 6.3 端口上下拉控制
类型：寄存器定义
寄存器：PUA
位定义：
Bit7 PUA7
Bit6 PUA6
|  | Bit7 | Bit6 | Bit5 | Bit4 | Bit3 | Bit2 | Bit1 | Bit0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PUA | PUA7 | PUA6 | PUA5 | PUA4 | PUA3 | PUA2 | PUA1 | PUA0 |
| PUB | PDA2 | PDA0 | PUB5 | PUB4 | PUB3 | PUB2 | PUB1 | PUB0 |
"""

IOAICR_53 = """M8P5170 Version 1.06 > 5 中断 > 5.3 IOAICR 变化中断使能寄存器
类型：寄存器定义
寄存器：IOAICR
位定义：
Bit4 IOAICR4
| IOAICR | - | - | - | IOAICR4 | - | - | - | - |
"""

IOAICR_64 = """M8P5170 Version 1.06 > 6 端口 > 6.4 IOAICR 变化中断使能寄存器
类型：寄存器定义
寄存器：IOAICR
位定义：
Bit4 IOAICR4
| IOAICR | - | - | - | IOAICR4 | - | - | - | - |
"""


def _citation(text: str, section: str) -> Citation:
    return Citation(
        product="M8P6272",
        version="V1.10",
        section_path=section,
        page=1,
        kind=KnowledgeChunkKind.MIXED,
        text=text,
        score=1.0,
        chunk_id=uuid4(),
        vector_score=1.0,
    )


def _checks(citations: list[Citation], answer: str):
    service = RagQueryService.__new__(RagQueryService)
    return service._query_checks(citations, answer)


def test_runtime_checks_report_ambiguous_register_table() -> None:
    checks = _checks([_citation(PUA_MIXED, "6 端口 > 6.3")], "PUA.bit7=PDA2")
    assert any(item.kind == "ambiguous_scope" for item in checks.evidence)


def test_runtime_checks_report_duplicate_register_and_unsupported_symbol() -> None:
    checks = _checks(
        [
            _citation(IOAICR_53, "5 中断 > 5.3"),
            _citation(IOAICR_64, "6 端口 > 6.4"),
        ],
        "IOA4 对应的使能位是 IOAICR4。",
    )
    assert any(item.kind == "duplicate_register" for item in checks.evidence)
    assert any(item.kind == "unsupported_symbol" for item in checks.answer)


def test_runtime_checks_are_empty_for_clean_answer() -> None:
    checks = _checks([_citation(PUA_MIXED, "6 端口 > 6.3")], "根据上下文无法确认。")
    assert checks.evidence
    # PUA/PUB is inherently ambiguous; it remains visible rather than silently cleaned.
    assert all(item.kind != "conflict" for item in checks.evidence)
