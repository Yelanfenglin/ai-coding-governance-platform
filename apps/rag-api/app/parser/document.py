"""跨页后处理：章节路径与可审计的表格续页链接。"""

from __future__ import annotations

import re
from itertools import pairwise

_SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+\S")


class SectionTracker:
    def __init__(self) -> None:
        self._levels: dict[int, tuple[str, str]] = {}

    def annotate(self, page: dict) -> dict:
        for block in page.get("blocks", []):
            if block.get("type") == "text" and block.get("kind") == "heading":
                self._consume_heading(block.get("text", ""))
            ordered = [self._levels[level] for level in sorted(self._levels)]
            block["section_id"] = ordered[-1][0] if ordered else ""
            block["section_path"] = [title for _, title in ordered]
        return page

    def _consume_heading(self, text: str) -> None:
        title = " ".join(text.split())
        match = _SECTION_RE.match(title)
        if match:
            section_id = match.group(1)
            depth = section_id.count(".") + 1
            parts = section_id.split(".")
        elif title.lower().startswith(("附录", "appendix")):
            section_id = title
            depth = 1
            parts = []
        else:
            # PP 的 paragraph_title 还会命中步骤提示、图题等非章节文本；
            # 无编号时不污染已有数字章节树。
            return
        kept: dict[int, tuple[str, str]] = {}
        for level, value in self._levels.items():
            if level >= depth:
                continue
            expected = ".".join(parts[:level]) if parts else value[0]
            if value[0] == expected:
                kept[level] = value
        self._levels = kept
        self._levels[depth] = (section_id, title)


def link_table_continuations(pages: list[dict]) -> None:
    """链接相邻页同章节、同列数且贴近页边的表格。

    这里只写显式 metadata，不物理拼接 rows；这样可避免重复表头和跨页误并，
    下游 chunker 可按 ``table_group_id`` 合并或扩展上下文。
    """
    for previous, current in pairwise(pages):
        left = _edge_table(previous, at_bottom=True)
        right = _edge_table(current, at_bottom=False)
        if left is None or right is None:
            continue
        if left.get("n_cols") != right.get("n_cols"):
            continue
        if not left.get("section_id") or left.get("section_id") != right.get("section_id"):
            continue
        group = left.get("table_group_id") or (
            f"table-p{previous['page']:03d}-b{left.get('block_index', 0):03d}"
        )
        left["table_group_id"] = group
        right["table_group_id"] = group
        left["continues_on_page"] = current["page"]
        right["continued_from_page"] = previous["page"]


def _edge_table(page: dict, at_bottom: bool) -> dict | None:
    height = float(page.get("height") or 0.0)
    tables = [block for block in page.get("blocks", []) if block.get("type") == "table"]
    if not tables or height <= 0:
        return None
    table = (
        max(tables, key=lambda block: block["bbox"][3])
        if at_bottom
        else min(tables, key=lambda block: block["bbox"][1])
    )
    if at_bottom and table["bbox"][3] < height * 0.82:
        return None
    if not at_bottom and table["bbox"][1] > height * 0.22:
        return None
    return table
