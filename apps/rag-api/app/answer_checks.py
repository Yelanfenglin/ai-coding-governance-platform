"""P1c: lightweight, deterministic answer/citation checks.

No LLM, no extra generation or judge calls.  It answers three narrow questions
about an already-produced answer:

1. does every register / bit symbol the answer names actually appear in a citation?
2. are hex addresses in the answer present in a citation?
3. is the answer's own structure intact (code fences, markdown tables, figures)?

Everything here is advisory.  A finding means "this claim is not visibly backed by
the cited text", which is a prompt for a human or a later grounded check — not a
verdict that the answer is wrong.

Deliberately *not* checked
--------------------------
* numbers in prose (5.1K, 30%, 12 位): too easy to paraphrase, would flood with
  false positives;
* semantic entailment: that is the judge's job, not a regex's;
* image contents: only whether a figure is cited at all.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .evidence_checks import Citation

# Register-like symbols that carry a bit index, e.g. PDA5, IOA5, PUA7.
_SYMBOL_WITH_INDEX = re.compile(r"^[A-Z]{2,9}[0-9]{1,2}$")
_UPPERCASE_TOKEN = re.compile(r"(?<![A-Za-z0-9])([A-Z][A-Z0-9]{1,11})(?![A-Za-z0-9])")
_HEX_ADDRESS = re.compile(
    r"(?<![A-Za-z0-9])((?:[0-9A-F]{4}H)|(?:0x[0-9A-Fa-f]{2,4}))(?![A-Za-z0-9])"
)
_CODE_FENCE = re.compile(r"^\s*```", re.MULTILINE)
_IMAGE_CUE = ("如图", "见图", "上图", "下图", "图片", "图中", "示意图", "框图", "外形图")
_VISUAL_HINT = ("snapshot", "![", "image", "figure")


@dataclass(frozen=True, slots=True)
class AnswerFinding:
    kind: str
    value: str
    detail: str


def _citation_text(citations: Sequence[Citation]) -> str:
    return "\n".join(citation.text for citation in citations)


def symbol_mentions(answer: str, registers: frozenset[str]) -> tuple[str, ...]:
    """Register-like symbols the answer names, in first-appearance order."""

    found: dict[str, None] = {}
    for match in _UPPERCASE_TOKEN.finditer(answer):
        token = match.group(1)
        if token in registers or _SYMBOL_WITH_INDEX.fullmatch(token):
            found.setdefault(token, None)
    return tuple(found)


def check_symbols(
    answer: str, citations: Sequence[Citation], registers: frozenset[str]
) -> tuple[AnswerFinding, ...]:
    cited = _citation_text(citations)
    findings: list[AnswerFinding] = []
    for token in symbol_mentions(answer, registers):
        if token in cited:
            continue
        is_register = token in registers
        findings.append(
            AnswerFinding(
                kind="unsupported_register" if is_register else "unsupported_symbol",
                value=token,
                detail=(
                    f"answer names {token} but no citation contains it"
                    + (" (catalog register)" if is_register else " (bit symbol)")
                ),
            )
        )
    return tuple(findings)


def check_addresses(answer: str, citations: Sequence[Citation]) -> tuple[AnswerFinding, ...]:
    cited = _citation_text(citations).upper()
    findings: list[AnswerFinding] = []
    for match in _HEX_ADDRESS.finditer(answer):
        token = match.group(1)
        if token.upper() in cited:
            continue
        findings.append(
            AnswerFinding(
                kind="unsupported_address",
                value=token,
                detail=f"answer states address {token} but no citation contains it",
            )
        )
    return tuple(findings)


def check_structure(answer: str, citations: Sequence[Citation]) -> tuple[AnswerFinding, ...]:
    findings: list[AnswerFinding] = []

    fences = len(_CODE_FENCE.findall(answer))
    if fences % 2:
        findings.append(
            AnswerFinding(
                kind="code_block_unbalanced",
                value=str(fences),
                detail=f"answer has {fences} code fences, so one block is never closed",
            )
        )

    table_lines = [line for line in answer.splitlines() if line.strip().startswith("|")]
    if table_lines:
        header = table_lines[0]
        expected = header.count("|") - 1
        for line in table_lines[2:]:
            if set(line.replace("|", "").strip()) <= {"-", ":", " "}:
                continue
            if line.count("|") - 1 != expected:
                findings.append(
                    AnswerFinding(
                        kind="table_ragged",
                        value=str(expected),
                        detail=(
                            "answer table row has "
                            f"{line.count('|') - 1} columns but the header has {expected}"
                        ),
                    )
                )
                break

    if any(cue in answer for cue in _IMAGE_CUE):
        has_figure = any(
            citation.kind.upper() in {"VISUAL", "FIGURE"}
            or any(hint in citation.text.lower() for hint in _VISUAL_HINT)
            for citation in citations
        )
        if not has_figure:
            findings.append(
                AnswerFinding(
                    kind="image_without_figure_citation",
                    value="",
                    detail="answer refers to a figure but no citation carries image content",
                )
            )

    return tuple(findings)


def check_answer(
    answer: str,
    citations: Sequence[Citation],
    *,
    registers: frozenset[str] = frozenset(),
) -> tuple[AnswerFinding, ...]:
    """Run every lightweight check; returns an empty tuple for a clean answer."""

    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    for citation in citations:
        if not isinstance(citation, Citation):
            raise TypeError("citations must contain Citation values")
    if not answer.strip() or not citations:
        return ()
    return (
        check_symbols(answer, citations, registers)
        + check_addresses(answer, citations)
        + check_structure(answer, citations)
    )


def summarise(findings: Iterable[AnswerFinding]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.kind] = counts.get(finding.kind, 0) + 1
    return counts
