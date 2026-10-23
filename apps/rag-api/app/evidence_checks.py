"""Deterministic evidence checks: register-block parsing and source-table conflicts.

No LLM, no runtime calls.  Two kinds of findings, deliberately separated so a
"needs care when citing" signal is never reported as "the sources contradict each
other":

``conflict``
    Two citations state *different* facts for the same (product, register, bit):
    a different bit symbol, or a different reset value.  This is a real
    contradiction and must block a confident answer.

``cross_section_variance``
    The same (product, register, bit) is defined differently in *different*
    sections.  Usually legitimate: ADH has a left-aligned layout in 10.5.1 and a
    right-aligned one in 10.5.2.  Informational, never "the sources contradict".

``ambiguous_scope``
    A citation declares one register but also carries another register's table
    whose cells are **not self-describing** — e.g. header ``寄存器：PUA`` plus a
    ``PUB`` row reading ``PDA2`` for bit7 (the R056 shape).  Quoting "PUA.bit7"
    from that chunk can silently pick up the other register's symbol.

    Sibling tables whose cells carry their own register prefix (``IOC7``,
    ``OEC7``, ``PWM01CR7``) are *not* flagged: the symbol already says which
    register it belongs to, so there is nothing to confuse.  That difference is
    what keeps this signal from firing on every port-configuration chunk.

Parsing targets the shape the chunker actually emits:

```text
M8P6272 Version 1.10 > 6 端口 > 6.3 端口上下拉控制
类型：寄存器定义
寄存器：PUA
位定义：
Bit7 PUA7
...
| PUA | PUA7 | PUA6 | ... | PUA0 |
| 复位后 | 0 | 0 | ... | 0 |
```
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .scalar_metadata import section_id_from_path

_REGISTER_HEADER = re.compile(r"^\s*寄存器\s*[:：]\s*([A-Za-z][A-Za-z0-9_]*)", re.MULTILINE)
_BIT_LINE = re.compile(r"^\s*Bit\s*(\d{1,2})\s*[:：]?\s*([A-Za-z][A-Za-z0-9_]*)\s*$")
_PRODUCT_PREFIX = re.compile(r"^\s*([A-Za-z][A-Za-z0-9]*)\s+Version\b")
_TABLE_ROW = re.compile(r"^\s*\|\s*([^|]+?)\s*\|(.+)\|\s*$")
_RESET_LABEL = "复位后"
_REGISTER_LABEL = re.compile(r"^[A-Z][A-Z0-9_]{1,15}$")
# A cell that looks like another register's bit symbol: "PDA2", "LTS1", "IOC1".
# Function names such as "PWM01EN" or "SMTENB" do not match and are not flagged.
_FOREIGN_BIT_SYMBOL = re.compile(r"^[A-Z]{2,}[0-9]{1,2}$")


@dataclass(frozen=True, slots=True)
class Citation:
    """One retrieved chunk with the section it came from."""

    chunk_id: str
    text: str
    section_path: str = ""
    kind: str = ""

    @property
    def section_id(self) -> str:
        return section_id_from_path(self.section_path)


@dataclass(frozen=True, slots=True)
class RegisterBlock:
    """One declared register plus whatever bit facts the chunk states."""

    product: str
    register: str
    section_id: str = ""
    bits: dict[int, str] = field(default_factory=dict)
    reset: dict[int, str] = field(default_factory=dict)
    table_registers: tuple[str, ...] = ()
    table_cells: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def declared(self) -> str:
        return self.register


@dataclass(frozen=True, slots=True)
class Finding:
    kind: str  # "conflict" | "ambiguous_scope"
    detail: str
    product: str
    register: str
    bit: int | None = None


def _product_of(text: str) -> str:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    match = _PRODUCT_PREFIX.match(first_line)
    return match.group(1).upper() if match else ""


def _table_rows(text: str) -> list[tuple[str, list[str]]]:
    rows: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        match = _TABLE_ROW.match(line)
        if not match:
            continue
        label = match.group(1).strip()
        cells = [cell.strip() for cell in match.group(2).split("|")]
        rows.append((label, cells))
    return rows


def parse_register_block(text: str, *, section_path: str = "") -> RegisterBlock | None:
    """Parse one chunk into a register block, or ``None`` when it is not one."""

    header = _REGISTER_HEADER.search(text)
    if header is None:
        return None
    register = header.group(1).upper()

    bits: dict[int, str] = {}
    for line in text.splitlines():
        match = _BIT_LINE.match(line)
        if match:
            bit = int(match.group(1))
            bits.setdefault(bit, match.group(2))

    table_registers: list[str] = []
    table_cells: dict[str, tuple[str, ...]] = {}
    reset: dict[int, str] = {}
    for label, cells in _table_rows(text):
        if not label:
            # Header row ("| | Bit7 | ... |"): not a data row.
            continue
        if label in {"读/写", "读写"}:
            continue
        if label == _RESET_LABEL:
            values = [cell for cell in cells][:8]
            for index, value in enumerate(values):
                bit = 7 - index
                if value and bit not in reset:
                    reset[bit] = value
            continue
        if not _REGISTER_LABEL.fullmatch(label):
            continue
        symbols = [cell for cell in cells if cell][:8]
        if len(symbols) < 8 or not any(re.match(r"^[A-Za-z]", cell) for cell in symbols):
            continue
        table_registers.append(label)
        table_cells.setdefault(label, tuple(symbols))
        for index, symbol in enumerate(symbols):
            bit = 7 - index
            if bit not in bits:
                bits[bit] = symbol

    return RegisterBlock(
        product=_product_of(text),
        register=register,
        section_id=section_id_from_path(section_path),
        bits=bits,
        reset=reset,
        table_registers=tuple(dict.fromkeys(table_registers)),
        table_cells=table_cells,
    )


def ambiguous_scope(block: RegisterBlock) -> Finding | None:
    """Header register vs a foreign table whose cells do not name themselves."""

    confusable: list[str] = []
    for label, cells in block.table_cells.items():
        if label == block.register:
            continue
        foreign = [
            cell
            for cell in cells
            if _FOREIGN_BIT_SYMBOL.fullmatch(cell.upper())
            and not cell.upper().startswith(label.upper())
        ]
        if foreign:
            confusable.append(f"{label} ({foreign[0]})")
    if not confusable:
        return None
    return Finding(
        kind="ambiguous_scope",
        detail=(
            f"chunk declares 寄存器：{block.register} but its other table(s) do not "
            f"name their register in every cell: {', '.join(confusable)}"
        ),
        product=block.product,
        register=block.register,
    )


def detect_findings(citations: Sequence[Citation]) -> tuple[Finding, ...]:
    """Run all deterministic checks over the retrieved citations."""

    blocks = [
        block
        for block in (
            parse_register_block(citation.text, section_path=citation.section_path)
            for citation in citations
        )
        if block is not None
    ]

    findings: list[Finding] = []
    for block in blocks:
        finding = ambiguous_scope(block)
        if finding is not None:
            findings.append(finding)

    def compare(
        facts: Iterable[tuple[str, str, str, int, str]],
        template: str,
    ) -> None:
        """facts = (product, register, section_id, bit, value)."""

        seen: dict[tuple[str, str, int], tuple[str, str]] = {}
        for product, register, section, bit, value in facts:
            key = (product, register, bit)
            previous = seen.get(key)
            if previous is not None and previous[0] != value:
                same_section = previous[1] == section
                findings.append(
                    Finding(
                        kind="conflict" if same_section else "cross_section_variance",
                        detail=template.format(
                            product=product,
                            register=register,
                            bit=bit,
                            previous=previous[0],
                            value=value,
                            section_a=previous[1] or "-",
                            section_b=section or "-",
                        ),
                        product=product,
                        register=register,
                        bit=bit,
                    )
                )
            seen.setdefault(key, (value, section))

    compare(
        (
            (block.product, block.register, block.section_id, bit, symbol)
            for block in blocks
            for bit, symbol in block.bits.items()
        ),
        "{product} {register}.bit{bit} is {previous} in section {section_a} "
        "and {value} in section {section_b}",
    )
    compare(
        (
            (block.product, block.register, block.section_id, bit, value)
            for block in blocks
            for bit, value in block.reset.items()
        ),
        "{product} {register}.bit{bit} resets to {previous} in section {section_a} "
        "and {value} in section {section_b}",
    )

    return tuple(findings)


def detect_duplicate_registers(citations: Sequence[Citation]) -> tuple[Finding, ...]:
    """Flag the same register appearing as multiple blocks in the context.

    This is the runtime-visible proxy for the R050 selection failure: two chunks
    define the same register in different sections, consuming budget without
    adding a new evidence group.
    """

    groups: dict[tuple[str, str], list[RegisterBlock]] = {}
    for citation in citations:
        block = parse_register_block(citation.text, section_path=citation.section_path)
        if block is not None:
            groups.setdefault((block.product, block.register), []).append(block)
    findings: list[Finding] = []
    for (product, register), blocks in groups.items():
        if len(blocks) < 2:
            continue
        sections = ", ".join(block.section_id or "-" for block in blocks)
        findings.append(
            Finding(
                kind="duplicate_register",
                detail=(
                    f"{product} {register} appears in multiple citation blocks "
                    f"(sections: {sections})"
                ),
                product=product,
                register=register,
            )
        )
    return tuple(findings)


def summarise(findings: Iterable[Finding]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.kind] = counts.get(finding.kind, 0) + 1
    return counts
