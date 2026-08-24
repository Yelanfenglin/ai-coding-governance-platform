"""Deterministic semantic cleanup after layout ordering.

The cleanup keeps geometry and reading order intact.  Numbered headings are
recovered at document scope, where typography and the surrounding section
sequence can distinguish a chapter number from values such as ``0 = off``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_NUMBERED_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,5})\s+(.+?)\s*$")
_EXISTING_HEADING_RE = re.compile(
    r"^\s*(\d+(?:\.\d+){0,5})(?:\s+|(?=[A-Za-z\u3400-\u9fff]))(.+?)\s*$"
)
_NEGATIVE_HEADING_RE = re.compile(r"^\s*(?:注|说明|备注|注意|例)\s*[:：]")
_TOC_LEADER_RE = re.compile(r"(?:\.{3,}|…{2,})")
_BULLET_RE = re.compile(r"^\s*(?:[■●•◼▪◆◇▶►]|[-*+]\s+|\d+[、.)）]\s*)")
_VALUE_ENUM_RE = re.compile(r"^\s*[0-9A-Fa-fxX]+\s*(?:=|[:：])\s*\S")
_VALUE_PREFIX_RE = re.compile(r"^[=<>≤≥+*/%:]|^[-+]?(?:\d+(?:\.\d+)?)\s*(?:V|A|Hz|s|%)\b", re.I)
_SENTENCE_END = frozenset("。！？；!?;：:")

_ROOT_TYPOGRAPHY_RATIO = 1.8
_NESTED_TYPOGRAPHY_RATIO = 1.3
_LOOKAHEAD_PAGES = 1
_LOOKAHEAD_BLOCKS = 30


@dataclass(frozen=True)
class _HeadingCandidate:
    section_id: str
    title: str

    @property
    def parts(self) -> tuple[int, ...]:
        return tuple(int(part) for part in self.section_id.split("."))

    @property
    def depth(self) -> int:
        return len(self.parts)

    @property
    def short_title(self) -> bool:
        return len(self.title.replace(" ", "")) < 2


def normalize_text_block(kind: str, text: str) -> tuple[str, str]:
    """Reflow one block and normalize layout-provided headings.

    Paragraphs are not promoted here: isolated regex matching has no access to
    typography or neighboring section numbers.  ``recover_document_headings``
    performs that decision after every page has an established reading order.
    """
    normalized = reflow_text(text)
    if not normalized:
        return kind, normalized
    if kind == "heading" and _NEGATIVE_HEADING_RE.match(normalized):
        return "paragraph", normalized
    if kind == "heading":
        compact = " ".join(normalized.split())
        match = _EXISTING_HEADING_RE.match(compact)
        if match:
            compact = f"{match.group(1)} {match.group(2).strip()}"
        return kind, compact
    return kind, normalized


def recover_document_headings(pages: list) -> None:
    """Recover and validate numbered headings with document context.

    Evidence is intentionally domain-neutral: layout label, relative font
    size, the previously accepted hierarchy, and nearby descendant numbers.
    No document name, page number, chip field, or title vocabulary is used.
    Existing layout headings that look like numeric values but lack contextual
    support are demoted as well.
    """
    entries = [
        (page_index, page, block)
        for page_index, page in enumerate(pages)
        for block in page.blocks
        if hasattr(block, "kind") and hasattr(block, "text")
    ]
    candidates = [_parse_heading_candidate(block.text) for _, _, block in entries]
    accepted_by_depth: dict[int, tuple[int, ...]] = {}
    current_root: int | None = None

    for index, ((page_index, page, block), candidate) in enumerate(
        zip(entries, candidates, strict=True)
    ):
        if candidate is None:
            continue
        original_heading = block.kind == "heading" or getattr(block, "pp_label", "") == (
            "paragraph_title"
        )
        parts = candidate.parts
        body_size = max(float(getattr(page, "body_font_size", 0.0)), 0.0)
        font_size = max(float(getattr(block, "font_size", 0.0)), 0.0)
        typography_ratio = font_size / body_size if body_size > 0.0 else 0.0
        descendant_ahead = _has_nearby_descendant(
            index,
            page_index,
            candidate.section_id,
            entries,
            candidates,
        )

        if candidate.depth == 1:
            number = parts[0]
            sequence = (current_root is None and number == 1) or (
                current_root is not None and number == current_root + 1
            )
            evidence = sum(
                (
                    original_heading,
                    typography_ratio >= _ROOT_TYPOGRAPHY_RATIO,
                    sequence,
                    descendant_ahead,
                )
            )
            # Very short labels are ambiguous, but they are not forbidden:
            # a genuine chapter such as ``1 序`` survives when typography,
            # sequence, layout, or descendant context provide a third signal.
            accepted = evidence >= (3 if candidate.short_title else 2)
        else:
            parent = parts[:-1]
            same_root = current_root is not None and parts[0] == current_root
            known_parent = accepted_by_depth.get(candidate.depth - 1) == parent
            previous_peer = accepted_by_depth.get(candidate.depth)
            peer_sequence = (
                previous_peer is not None
                and previous_peer[:-1] == parent
                and parts[-1] == previous_peer[-1] + 1
            )
            evidence = sum(
                (
                    original_heading,
                    typography_ratio >= _NESTED_TYPOGRAPHY_RATIO,
                    same_root,
                    known_parent,
                    peer_sequence,
                    descendant_ahead,
                )
            )
            accepted = evidence >= (3 if candidate.short_title else 2)

        if not accepted:
            if block.kind == "heading":
                block.kind = "paragraph"
            continue

        block.kind = "heading"
        block.text = f"{candidate.section_id} {candidate.title}"
        if candidate.depth == 1:
            current_root = parts[0]
        for depth in [depth for depth in accepted_by_depth if depth >= candidate.depth]:
            del accepted_by_depth[depth]
        accepted_by_depth[candidate.depth] = parts


def markdown_heading_level(text: str, kind: str) -> int | None:
    if kind != "heading":
        return None
    match = _EXISTING_HEADING_RE.match(" ".join(text.split()))
    if not match:
        return 2
    depth = match.group(1).count(".") + 1
    return min(6, depth + 1)


def reflow_text(text: str) -> str:
    """Join visual wraps while preserving bullets and semantic boundaries."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    output = [lines[0]]
    for line in lines[1:]:
        previous = output[-1]
        if _starts_new_structure(line) or _ends_semantic_unit(previous):
            output.append(line)
        else:
            output[-1] = _join_line(previous, line)
    return "\n".join(output)


def _is_numbered_heading(text: str) -> bool:
    return _parse_heading_candidate(text) is not None


def _starts_new_structure(line: str) -> bool:
    return bool(_BULLET_RE.match(line) or _VALUE_ENUM_RE.match(line) or _is_numbered_heading(line))


def _ends_semantic_unit(line: str) -> bool:
    return bool(line) and line[-1] in _SENTENCE_END


def _join_line(left: str, right: str) -> str:
    if not left or not right:
        return left + right
    left_char, right_char = left[-1], right[0]
    if (
        left_char.isascii()
        and right_char.isascii()
        and left_char.isalnum()
        and right_char.isalnum()
    ):
        return f"{left} {right}"
    return left + right


def _parse_heading_candidate(text: str) -> _HeadingCandidate | None:
    compact = " ".join(text.split())
    match = _NUMBERED_HEADING_RE.match(compact)
    if not match or len(compact) > 80:
        return None
    if _TOC_LEADER_RE.search(compact) or _NEGATIVE_HEADING_RE.match(compact):
        return None
    title = match.group(2).strip()
    if not title or title[-1] in "。；，,;" or _VALUE_PREFIX_RE.match(title):
        return None
    return _HeadingCandidate(match.group(1), title)


def _has_nearby_descendant(
    index: int,
    page_index: int,
    section_id: str,
    entries: list,
    candidates: list[_HeadingCandidate | None],
) -> bool:
    prefix = f"{section_id}."
    for future_index in range(index + 1, min(len(entries), index + _LOOKAHEAD_BLOCKS + 1)):
        future_page_index = entries[future_index][0]
        if future_page_index - page_index > _LOOKAHEAD_PAGES:
            break
        future = candidates[future_index]
        if future is not None and future.section_id.startswith(prefix):
            return True
    return False
