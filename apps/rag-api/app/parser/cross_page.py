"""Cross-page table stitcher: join physical page fragments into logical tables.

Runs on ``PageIR`` blocks after each page is built and before rendering.  It
only merges when the evidence is strong:

  * the right page's **first** body block is exactly the continuation table
    (this kills the many identical ``Bit7..Bit0`` register tables, whose
    headers repeat but are introduced by a real heading on the next page),
  * the repeated header prefix matches after whitespace normalisation
    (``单 位`` == ``单位``),
  * column count and horizontal span agree (bbox is an insurance check, not
    the primary signal).

It is deliberately conservative (matches the "rather not merge than merge
wrong" policy): it never splits a mid-row page break and never merges a
snapshot/image.  A multi-page continuation is folded into one table whose
``source_pages`` records the page span.
"""

from __future__ import annotations

from .page import PageIR, TableBlock
from .table import AlignedSubrow, Cell, Table

_EPS_FRAC = 0.03
_MAX_HEADER_ROWS = 3


def _norm(text: str) -> str:
    """Normalise a header cell for comparison.

    Collapse *all* whitespace so ``单位`` and ``单 位`` compare equal; short
    header labels carry no meaningful whitespace for matching purposes.
    """
    return "".join(text.split())


def _head_rows(table: Table) -> list[list[str]]:
    return [[_norm(cell) for cell in row] for row in table.text_rows()]


def common_header_rows(left: Table, right: Table, max_rows: int = _MAX_HEADER_ROWS) -> int:
    """Longest identical leading row prefix (the repeated header)."""
    a = _head_rows(left)
    b = _head_rows(right)
    n = 0
    for ra, rb in zip(a[:max_rows], b[:max_rows], strict=False):
        if ra != rb:
            break
        n += 1
    return n


def same_grid(left: TableBlock, right: TableBlock, page_width: float) -> bool:
    """Column count matches and the horizontal span lines up (insurance)."""
    if left.table.n_cols != right.table.n_cols:
        return False
    eps = page_width * _EPS_FRAC
    return abs(left.bbox.x0 - right.bbox.x0) < eps and abs(left.bbox.x1 - right.bbox.x1) < eps


def _offset_cell(cell: Cell, offset: int) -> Cell:
    return Cell(
        row=cell.row + offset,
        col=cell.col,
        rowspan=cell.rowspan,
        colspan=cell.colspan,
        text=cell.text,
        bbox=cell.bbox,
        physical_lines=list(cell.physical_lines),
    )


def _merge_is_safe(base: Table, cont: Table, header_n: int) -> bool:
    """Guard against a rowspan that would be severed by dropping the header.

    Dropping ``cont.rows[:header_n]`` must not orphan a rowspan that reaches
    into the kept rows; and appending must not interrupt a base rowspan (in a
    well-formed base table this never happens, so it is a defensive check).
    """
    for row_index in range(header_n):
        for cell in cont.rows[row_index]:
            if cell is not None and cell.row + cell.rowspan > header_n:
                return False
    for row in base.rows:
        for cell in row:
            if cell is not None and cell.row + cell.rowspan > base.n_rows:
                return False
    return True


def _append_continuation(base: Table, cont: Table, header_n: int) -> None:
    offset = base.n_rows
    kept = cont.rows[header_n:]
    # A kept row originally at ``cont`` row ``header_n + k`` must land at merged
    # row ``offset + k``; a cell's own row index therefore shifts by
    # ``offset - header_n`` (the same amount a continuation subrow moves).
    subrow_offset = offset - header_n
    for row in kept:
        base.rows.append(
            [None if cell is None else _offset_cell(cell, subrow_offset) for cell in row]
        )
    base.n_rows = offset + len(kept)
    base.aligned_subrows.extend(
        AlignedSubrow(
            parent_row=subrow.parent_row + subrow_offset,
            subrow=subrow.subrow,
            values=subrow.values,
            bboxes=subrow.bboxes,
            max_baseline_delta=subrow.max_baseline_delta,
        )
        for subrow in cont.aligned_subrows
        if subrow.parent_row >= header_n
    )
    base.split_merges += cont.split_merges


def stitch_cross_page_tables(page_irs: list[PageIR]) -> None:
    """Fold every confirmed cross-page table continuation into one logical table.

    Mutates ``page_irs`` in place: the first page's trailing ``TableBlock``
    absorbs the following pages' leading continuation ``TableBlock`` rows
    (minus the repeated header) and records ``source_pages``.  The continuation
    table is removed from its page, so a fully-absorbed page renders nothing.
    """
    i = 0
    while i < len(page_irs):
        left = page_irs[i]
        if not left.blocks or not isinstance(left.blocks[-1], TableBlock):
            i += 1
            continue
        left_last = left.blocks[-1]
        span_pages = [left.page_number]
        j = i + 1
        while j < len(page_irs):
            nxt = page_irs[j]
            if not nxt.blocks or not isinstance(nxt.blocks[0], TableBlock):
                break
            cont = nxt.blocks[0]
            header_n = common_header_rows(left_last.table, cont.table)
            if header_n < 1 or not same_grid(left_last, cont, left.width):
                break
            if not _merge_is_safe(left_last.table, cont.table, header_n):
                break
            _append_continuation(left_last.table, cont.table, header_n)
            span_pages.append(nxt.page_number)
            # A continuation page's implicit-subtable rows/groups now belong to
            # the merged table on the first page; relocate the counters so the
            # per-page stats stay consistent with the (merged) blocks.
            left.stats["implicit_subtable_groups"] = left.stats.get(
                "implicit_subtable_groups", 0
            ) + nxt.stats.get("implicit_subtable_groups", 0)
            left.stats["implicit_subrows"] = left.stats.get("implicit_subrows", 0) + nxt.stats.get(
                "implicit_subrows", 0
            )
            nxt.stats["implicit_subtable_groups"] = 0
            nxt.stats["implicit_subrows"] = 0
            nxt.blocks.pop(0)
            j += 1
        if len(span_pages) > 1:
            left_last.source_pages = span_pages
            left.stats["cross_page_merged_pages"] = span_pages
            i = j
        else:
            i += 1
