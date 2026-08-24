"""PageIR -> portable Markdown/HTML；快照文件名由单一函数生成。"""

from __future__ import annotations

import html

from .page import PageIR, SnapshotBlock, TableBlock, TextBlock
from .postprocess import markdown_heading_level


def snapshot_filename(page_number: int, ordinal: int, hint: str) -> str:
    safe_hint = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in hint)
    return f"p{page_number:03d}_s{ordinal}_{safe_hint or 'visual'}.png"


def blocks_to_markdown(page: PageIR, snapshot_prefix: str = "../snapshots") -> str:
    out: list[str] = []
    snapshot_ordinal = 0
    for block in page.blocks:
        if isinstance(block, SnapshotBlock):
            snapshot_ordinal += 1
            rendered = _render_snapshot(
                block,
                page.page_number,
                snapshot_ordinal,
                snapshot_prefix,
            )
        else:
            rendered = _render_block(block)
        if rendered:
            out.append(rendered)
    return "\n\n".join(out)


def _render_block(block) -> str:
    if isinstance(block, TextBlock):
        text = block.text.strip()
        level = markdown_heading_level(text, block.kind)
        return f"{'#' * level} {text}" if level is not None else text
    if isinstance(block, TableBlock):
        return _render_table(block)
    return ""


def _render_table(block: TableBlock) -> str:
    table = block.table
    if not table.rows:
        return ""
    prefix = ""
    if block.source_pages and len(block.source_pages) > 1:
        prefix = f"<!-- pdf-pages: {'-'.join(str(p) for p in block.source_pages)} -->\n\n"
    if any(cell.rowspan > 1 or cell.colspan > 1 for cell in table.origin_cells()):
        return prefix + _render_html_table(block)
    rows = table.semantic_text_rows()
    lines: list[str] = []
    for row_index, row in enumerate(rows):
        cells = [_markdown_cell(text) for text in row]
        lines.append("| " + " | ".join(cells) + " |")
        if row_index == 0:
            lines.append("| " + " | ".join(["---"] * table.n_cols) + " |")
    return prefix + "\n".join(lines)


def _render_html_table(block: TableBlock) -> str:
    table = block.table
    lines = ["<table>"]
    aligned = {}
    for subrow in table.aligned_subrows:
        aligned.setdefault(subrow.parent_row, []).append(subrow)
    for row_index, row in enumerate(table.rows):
        recovered = sorted(aligned.get(row_index, []), key=lambda item: item.subrow)
        if recovered:
            for subrow in recovered:
                lines.append("  <tr>")
                for value in subrow.values:
                    lines.append(f"    <td>{html.escape(value.strip())}</td>")
                lines.append("  </tr>")
            continue
        lines.append("  <tr>")
        for cell in row:
            if cell is None or cell.col >= len(row) or row[cell.col] is not cell:
                continue
            attrs = []
            if cell.rowspan > 1:
                attrs.append(f'rowspan="{cell.rowspan}"')
            if cell.colspan > 1:
                attrs.append(f'colspan="{cell.colspan}"')
            attr_text = (" " + " ".join(attrs)) if attrs else ""
            value = html.escape(cell.text.strip()).replace("\n", "<br>")
            lines.append(f"    <td{attr_text}>{value}</td>")
        lines.append("  </tr>")
    lines.append("</table>")
    return "\n".join(lines)


def _markdown_cell(text: str) -> str:
    return text.strip().replace("|", "\\|").replace("\n", "<br>")


def _render_snapshot(
    block: SnapshotBlock,
    page_number: int,
    ordinal: int,
    snapshot_prefix: str,
) -> str:
    bbox = block.bbox
    filename = snapshot_filename(page_number, ordinal, block.hint)
    prefix = snapshot_prefix.rstrip("/")
    ref = f"{prefix}/{filename}" if prefix else filename
    rendered = (
        f"![p{page_number} {block.hint}]({ref})  "
        f"<!-- type={block.hint} source={block.source} "
        f"bbox=({bbox.x0:.1f},{bbox.y0:.1f},{bbox.x1:.1f},{bbox.y1:.1f}) -->"
    )
    if block.annotations.strip():
        annotation = block.annotations.strip().replace("--", "-")[:2000]
        rendered += f"\n<!-- annotations: {annotation} -->"
    return rendered
