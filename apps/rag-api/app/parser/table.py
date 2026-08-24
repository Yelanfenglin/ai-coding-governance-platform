"""Ruled-table geometry reconstruction with conservative merge repair.

正确性关键：
  1. 用真实横/竖线聚类出 row/col 边界
  2. 判断每个 atomic cell 的四条 separator 是否真实存在（线是否覆盖该区间）
  3. 缺失 separator 的相邻 atomic cell 形成候选 rowspan/colspan
  4. 先按 atomic cell 分配 native spans，再拆掉吞入多行/多列独立文本的候选合并

sanity check（只留「是否形成重复二维记录」这一个原则）：
  record_rows = [row for row in rows if nonempty_cells(row) >= 2]
  valid = len(record_rows) >= 2 and recurring_columns(record_rows) >= 2

合并策略遵循「宁可不合并，也不要错合并」：一个候选 merge 内若多个
atomic 行或列各自有文字，就按这些文字起点重新分段；只有文字集中在一个
atomic 区域时才保留跨度。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .models import BBox, VectorRule
from .native_pdf import NativeChar, NativeLine, NativePage, NativeSpan

# 横/竖线聚类的判定：同一条边界线的微小抖动容差
_GAP = 4.0
# 判断一条线段覆盖区间的容差（pt）
_COVER_MARGIN = 2.0
_SEGMENT_JOIN_TOL = 3.0
_CONNECT_TOL = 3.0
# 文字只能作为边线模糊时的保守兜底。至少三个 atomic band、绝大多数 band
# 各自有文字，并且相邻行/列的 separator 节奏也支持，才按重复记录拆分。
_MIN_RECORD_REPAIR_SPAN = 3
_MIN_RECORD_ANCHOR_DENSITY = 0.8
_MIN_NEIGHBOR_SEPARATOR_RATIO = 0.5
# 无内部分隔线的外层 cell 有时实际包含一张按纵向基线对齐的子表。这里只在
# 多个相邻 body row 重复出现同一几何模式时恢复；单个多行 cell、两行表头和
# 普通视觉折行全部回退为原始 outer row。
_MIN_ALIGNED_LINES = 2
_STRONG_ALIGNED_LINES = 3
_MIN_ALIGNED_ROW_RUN = 3
_MIN_STRONG_ROWS_IN_RUN = 2
_ALIGNED_BASELINE_TOL_RATIO = 0.3
_ALIGNED_BASELINE_TOL_MIN = 1.25
_STACK_LINE_MAX_WIDTH_RATIO = 0.86


# ---------------------------------------------------------------------------
# 单元格模型
# ---------------------------------------------------------------------------
@dataclass
class CellLine:
    """One physical PDF text line inside a reconstructed cell."""

    text: str
    bbox: BBox
    source_lines: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class AlignedSubrow:
    """A high-confidence record recovered from aligned lines across cells."""

    parent_row: int
    subrow: int
    values: tuple[str, ...]
    bboxes: tuple[BBox, ...]
    max_baseline_delta: float = 0.0


@dataclass
class Cell:
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    text: str = ""
    bbox: BBox | None = None
    physical_lines: list[CellLine] = field(default_factory=list)


def _is_origin(row: list, c: Cell) -> bool:
    """判断 c 是否为它所在列的起始位（cell.col 与它在 row 里的列索引一致）。"""
    return c.col is not None and c.col < len(row) and row[c.col] is c


@dataclass
class Table:
    # rows 为逻辑网格：rows[r] 是与第 r 行对齐的 cell（跨行的 cell 会在其 span 的
    # 非起始行以 None/引用占位，便于统计二维记录）。实现时用占位 None。
    rows: list[list[Cell | None]] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    bbox: BBox | None = None
    split_merges: int = 0
    validated: bool | None = None
    aligned_subrows: list[AlignedSubrow] = field(default_factory=list)

    @property
    def record_rows(self) -> int:
        # 只统计 cell 起始位的实际文本（spanne 位置的引用不计）
        def row_content(r):
            return [c for c in r if c is not None and c.text.strip() and _is_origin(r, c)]

        return sum(1 for r in self.rows if len(row_content(r)) >= 2)

    @property
    def recurring_columns(self) -> int:
        ncols = self.n_cols
        recur = 0
        for j in range(ncols):
            filled = sum(
                1
                for r in self.rows
                if j < len(r) and r[j] is not None and r[j].text.strip() and _is_origin(r, r[j])
            )
            if filled >= 2:
                recur += 1
        return recur

    @property
    def valid(self) -> bool:
        """是否形成重复二维记录。"""
        if self.validated is not None:
            return self.validated
        return self.record_rows >= 2 and self.recurring_columns >= 2

    def origin_cells(self) -> list[Cell]:
        return [
            cell for row in self.rows for cell in row if cell is not None and _is_origin(row, cell)
        ]

    def text_rows(self) -> list[list[str]]:
        return [
            [cell.text if cell is not None and _is_origin(row, cell) else "" for cell in row]
            for row in self.rows
        ]

    @property
    def aligned_parent_rows(self) -> list[int]:
        return sorted({subrow.parent_row for subrow in self.aligned_subrows})

    def semantic_text_rows(self) -> list[list[str]]:
        """Expand only parser-confirmed implicit subtable rows.

        ``text_rows`` remains the lossless outer PDF topology.  This view is a
        canonical retrieval representation, not another downstream guess.
        """
        if not self.aligned_subrows:
            return self.text_rows()
        by_parent: dict[int, list[AlignedSubrow]] = {}
        for subrow in self.aligned_subrows:
            by_parent.setdefault(subrow.parent_row, []).append(subrow)
        semantic: list[list[str]] = []
        for row_index, raw_row in enumerate(self.text_rows()):
            recovered = sorted(
                by_parent.get(row_index, []),
                key=lambda item: item.subrow,
            )
            if recovered:
                semantic.extend([list(item.values) for item in recovered])
            else:
                semantic.append(raw_row)
        return semantic


@dataclass
class _Grid:
    cells: dict[tuple[int, int], Cell]
    atomic: list[list[str]]
    row_edges: list[float]
    column_edges: list[float]
    right_sep: list[list[bool]]
    down_sep: list[list[bool]]
    n_rows: int
    n_cols: int


# ---------------------------------------------------------------------------
# 聚类
# ---------------------------------------------------------------------------
def _cluster(vals: list[float], gap: float) -> list[list[float]]:
    if not vals:
        return []
    vals = sorted(vals)
    out = [[vals[0]]]
    for x in vals[1:]:
        if x - out[-1][-1] <= gap:
            out[-1].append(x)
        else:
            out.append([x])
    return out


def _rule_inside(r: VectorRule, roi: BBox, margin: float = 8.0) -> bool:
    return (
        r.bbox.x0 >= roi.x0 - margin
        and r.bbox.x1 <= roi.x1 + margin
        and r.bbox.y0 >= roi.y0 - margin
        and r.bbox.y1 <= roi.y1 + margin
    )


def connected_rule_components(
    rules: list[VectorRule], tolerance: float = _CONNECT_TOL
) -> list[list[VectorRule]]:
    """按真实相交/相接关系拆分向量线，避免把整页多个网格笛卡尔积成一张表。"""
    if not rules:
        return []
    uf = _UF(len(rules))
    for i, left in enumerate(rules):
        for j in range(i + 1, len(rules)):
            if _rules_connect(left, rules[j], tolerance):
                uf.union(i, j)
    groups: dict[int, list[VectorRule]] = {}
    for i, rule in enumerate(rules):
        groups.setdefault(uf.find(i), []).append(rule)
    return list(groups.values())


def rules_bbox(rules: list[VectorRule]) -> BBox | None:
    if not rules:
        return None
    return BBox(
        min(r.bbox.x0 for r in rules),
        min(r.bbox.y0 for r in rules),
        max(r.bbox.x1 for r in rules),
        max(r.bbox.y1 for r in rules),
    )


def rule_boundary_counts(rules: list[VectorRule]) -> tuple[int, int]:
    h = _cluster([round(r.bbox.cy, 1) for r in rules if r.horizontal], _GAP)
    v = _cluster([round(r.bbox.cx, 1) for r in rules if not r.horizontal], _GAP)
    return len(h), len(v)


def _rules_connect(a: VectorRule, b: VectorRule, tol: float) -> bool:
    if a.horizontal != b.horizontal:
        h, v = (a, b) if a.horizontal else (b, a)
        return (
            h.bbox.x0 - tol <= v.bbox.cx <= h.bbox.x1 + tol
            and v.bbox.y0 - tol <= h.bbox.cy <= v.bbox.y1 + tol
        )
    if a.horizontal:
        return abs(a.bbox.cy - b.bbox.cy) <= tol and _ranges_touch(
            a.bbox.x0, a.bbox.x1, b.bbox.x0, b.bbox.x1, tol
        )
    return abs(a.bbox.cx - b.bbox.cx) <= tol and _ranges_touch(
        a.bbox.y0, a.bbox.y1, b.bbox.y0, b.bbox.y1, tol
    )


def _ranges_touch(a0: float, a1: float, b0: float, b1: float, tol: float) -> bool:
    return max(a0, b0) <= min(a1, b1) + tol


# ---------------------------------------------------------------------------
# separator 覆盖检测
#   水平分隔线（row separator）：横线段，y≈边界，x 覆盖区间
#   垂直分隔线（col separator）：竖线段，x≈边界，y 覆盖区间
# ---------------------------------------------------------------------------
def _horiz_cover(
    hlines: list[VectorRule], yy: float, v0: float, v1: float, tol: float = 4.5
) -> bool:
    """同一 y 边界附近的横线区间并集是否覆盖 ``[v0, v1]``。"""
    intervals = [(h.bbox.x0, h.bbox.x1) for h in hlines if abs(h.bbox.cy - yy) <= tol]
    return _intervals_cover(intervals, v0, v1)


def _vert_cover(
    vlines: list[VectorRule], xx: float, h0: float, h1: float, tol: float = 4.5
) -> bool:
    """同一 x 边界附近的竖线区间并集是否覆盖 ``[h0, h1]``。"""
    intervals = [(v.bbox.y0, v.bbox.y1) for v in vlines if abs(v.bbox.cx - xx) <= tol]
    return _intervals_cover(intervals, h0, h1)


def _horiz_single_cover(
    hlines: list[VectorRule], yy: float, v0: float, v1: float, tol: float = 4.5
) -> bool:
    """旧版单片段覆盖判断，仅用于冻结既有分类结果。"""
    return any(
        abs(line.bbox.y0 - yy) <= tol
        and abs(line.bbox.y1 - yy) <= tol
        and line.bbox.x0 <= v0 + _COVER_MARGIN
        and line.bbox.x1 >= v1 - _COVER_MARGIN
        for line in hlines
    )


def _vert_single_cover(
    vlines: list[VectorRule], xx: float, h0: float, h1: float, tol: float = 4.5
) -> bool:
    """旧版单片段覆盖判断，仅用于冻结既有分类结果。"""
    return any(
        abs(line.bbox.x0 - xx) <= tol
        and abs(line.bbox.x1 - xx) <= tol
        and line.bbox.y0 <= h0 + _COVER_MARGIN
        and line.bbox.y1 >= h1 - _COVER_MARGIN
        for line in vlines
    )


def _intervals_cover(intervals: list[tuple[float, float]], start: float, stop: float) -> bool:
    """判断若干共线片段的区间并集是否覆盖目标边界。

    PDF 常把一条表格边界切成多个填充矩形，并在交叉点留下与线宽相当的
    小间隙。这里只跨越很小的几何间隙；更大的空档仍被视为真实的 merge。
    """
    if stop < start:
        start, stop = stop, start
    furthest = start
    for left, right in sorted(intervals):
        if right < start - _COVER_MARGIN:
            continue
        if left > furthest + _SEGMENT_JOIN_TOL:
            return False
        furthest = max(furthest, right)
        if furthest >= stop - _COVER_MARGIN:
            return True
    return False


# ---------------------------------------------------------------------------
# union-find：只用于把全页向量线拆成连通组件
# ---------------------------------------------------------------------------
class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


# ---------------------------------------------------------------------------
# 主重建
# ---------------------------------------------------------------------------
def reconstruct_table(
    roi: BBox,
    native: NativePage,
    *,
    validate_with_topology: bool = False,
) -> Table | None:
    grid = _reconstruct_grid(roi, native, native.rules)
    if grid is None:
        return None

    # 分类证据与拓扑证据分离。真实 PDF 会提供旧版高置信 rule，用它复现
    # 已有 table/snapshot 决策；增强 rule 只重建已确认表格的内部 topology。
    classification_grid = (
        grid
        if native.classification_rules is None or validate_with_topology
        else _reconstruct_grid(
            roi,
            native,
            native.classification_rules,
            combine_fragments=False,
        )
    )
    validated = _grid_valid(classification_grid)

    table_cells = grid.cells
    atomic = grid.atomic
    hy = grid.row_edges
    vx = grid.column_edges
    right_sep = grid.right_sep
    down_sep = grid.down_sep

    # 文字只处理几何仍含糊的候选 merge。只有高密度 atomic 文字锚点与
    # 邻行/邻列 separator 节奏同时支持重复记录时才分段；多行表头、稀疏
    # 分组标签和短纵排文字保持原跨度。
    repaired_cells: dict[tuple[int, int], Cell] = {}
    split_merges = 0
    for cell in table_cells.values():
        pieces = _partition_merge_by_atomic_text(
            cell,
            atomic,
            hy,
            vx,
            right_sep,
            down_sep,
        )
        if len(pieces) > 1:
            split_merges += 1
        for piece in pieces:
            repaired_cells[(piece.row, piece.col)] = piece

    # 逻辑网格：merge cell 放在其起始位，spanne 格填 None（统计/渲染都只认起始位）
    logic = _logic_grid(repaired_cells.values(), grid.n_rows, grid.n_cols)
    table = Table(
        rows=logic,
        n_rows=grid.n_rows,
        n_cols=grid.n_cols,
        bbox=BBox(vx[0], hy[0], vx[-1], hy[-1]),
        split_merges=split_merges,
        validated=validated,
    )
    _annotate_physical_cell_lines(table, native)
    table.aligned_subrows = _recover_aligned_subrows(table)
    return table


def _reconstruct_grid(
    roi: BBox,
    native: NativePage,
    rules: list[VectorRule],
    *,
    combine_fragments: bool = True,
) -> _Grid | None:
    roi_rules = [rule for rule in rules if _rule_inside(rule, roi)]
    if not roi_rules:
        return None
    # 正常 PP table ROI 已经是可靠的局部边界；使用其中全部 rule 才能保留
    # 因 rowspan/colspan 而未与主组件相交的短分隔线。连通组件只用于全页漏表发现。
    hlines = [r for r in roi_rules if r.horizontal]
    vlines = [r for r in roi_rules if not r.horizontal]

    hys = sorted({round(r.bbox.cy, 1) for r in hlines})
    vxs = sorted({round(r.bbox.cx, 1) for r in vlines})
    h_cl = _cluster(hys, _GAP)
    v_cl = _cluster(vxs, _GAP)

    # 单元格边界必须来自真实 rule。PP bbox 往往比外框宽 2~20pt；把 ROI
    # 再塞进 edges 会制造首尾空行/空列（P57/P64/P76 的旧结果即如此）。
    hy = _with_missing_outer_edges([(c[0] + c[-1]) / 2 for c in h_cl], roi.y0, roi.y1)
    vx = _with_missing_outer_edges([(c[0] + c[-1]) / 2 for c in v_cl], roi.x0, roi.x1)
    hy = _unique_sorted(hy)
    vx = _unique_sorted(vx)

    n_rows = len(hy) - 1
    n_cols = len(vx) - 1
    if n_rows < 2 or n_cols < 2:
        return None

    # 判断每个 atomic cell 的右分隔(right=竖线 in vx[j+1]) 与下分隔(down=横线 in hy[i+1])
    right_sep = [[False] * (n_cols - 1) for _ in range(n_rows)]
    vertical_cover = _vert_cover if combine_fragments else _vert_single_cover
    for i in range(n_rows):
        y0, y1 = hy[i], hy[i + 1]
        for j in range(n_cols - 1):
            right_sep[i][j] = vertical_cover(vlines, vx[j + 1], y0, y1)
    down_sep = [[False] * n_cols for _ in range(n_rows - 1)]
    horizontal_cover = _horiz_cover if combine_fragments else _horiz_single_cover
    for i in range(n_rows - 1):
        for j in range(n_cols):
            x0, x1 = vx[j], vx[j + 1]
            down_sep[i][j] = horizontal_cover(hlines, hy[i + 1], x0, x1)

    # 二维铺瓦：从左上角开始，把未覆盖格子向右、向下延伸成一个 merge cell。
    # 用 seen 标记，避免十字交叉把整个表吞进一个 cell。
    covered = [[False] * n_cols for _ in range(n_rows)]
    table_cells: dict[tuple[int, int], Cell] = {}
    for i in range(n_rows):
        for j in range(n_cols):
            if covered[i][j]:
                continue
            # 向右延伸 colspan：沿当前行，遇到存在竖线分隔则停
            cs = j
            while cs < n_cols - 1 and covered[i][cs + 1] is False and not right_sep[i][cs]:
                cs += 1
            colspan = cs - j + 1
            # 向下延伸 rowspan：对合并的每一列，要求没有横线分隔
            rs = i
            while rs < n_rows - 1:
                # 仅检查当前待跨越的横边界；下方原子格还必须未被占用。
                block_ok = all(
                    not down_sep[rs][cc] and not covered[rs + 1][cc] for cc in range(j, j + colspan)
                )
                if not block_ok:
                    break
                rs += 1
            rowspan = rs - i + 1
            cell = Cell(
                row=i,
                col=j,
                rowspan=rowspan,
                colspan=colspan,
                bbox=BBox(vx[j], hy[i], vx[j + colspan], hy[i + rowspan]),
            )
            table_cells[(i, j)] = cell
            for di in range(rowspan):
                for dj in range(colspan):
                    covered[i + di][j + dj] = True

    # atomic text：先按原子 cell 分配（不依赖 merge）
    atomic = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    origin_by_slot = _origin_by_atomic_slot(table_cells.values())
    for span in _iter_spans(native, roi):
        for row, col, text in _span_atomic_fragments(
            span,
            hy,
            vx,
            origin_by_slot,
        ):
            atomic[row][col] = (atomic[row][col] + " " + text).strip()

    for cell in table_cells.values():
        cell.text = _atomic_region_text(cell, atomic)
    return _Grid(
        cells=table_cells,
        atomic=atomic,
        row_edges=hy,
        column_edges=vx,
        right_sep=right_sep,
        down_sep=down_sep,
        n_rows=n_rows,
        n_cols=n_cols,
    )


def _grid_valid(grid: _Grid | None) -> bool:
    if grid is None:
        return False
    logic = _logic_grid(grid.cells.values(), grid.n_rows, grid.n_cols)
    return Table(rows=logic, n_rows=grid.n_rows, n_cols=grid.n_cols).valid


def _logic_grid(cells, n_rows: int, n_cols: int) -> list[list[Cell | None]]:
    logic: list[list[Cell | None]] = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    for cell in cells:
        logic[cell.row][cell.col] = cell
    return logic


def _atomic_region_text(cell: Cell, atomic: list[list[str]]) -> str:
    return " ".join(
        atomic[row][col]
        for row in range(cell.row, cell.row + cell.rowspan)
        for col in range(cell.col, cell.col + cell.colspan)
        if atomic[row][col]
    )


def _partition_merge_by_atomic_text(
    cell: Cell,
    atomic: list[list[str]],
    row_edges: list[float],
    column_edges: list[float],
    right_sep: list[list[bool]],
    down_sep: list[list[bool]],
) -> list[Cell]:
    row_start, row_stop = cell.row, cell.row + cell.rowspan
    col_start, col_stop = cell.col, cell.col + cell.colspan
    occupied_rows = [
        row
        for row in range(row_start, row_stop)
        if any(atomic[row][col].strip() for col in range(col_start, col_stop))
    ]
    row_segments = _segments_at_content(
        row_start,
        row_stop,
        occupied_rows,
        boundary_supported=lambda row: _row_boundary_has_neighbor_support(
            row,
            cell,
            down_sep,
            len(column_edges) - 1,
        ),
        preserve_short_vertical_text=_looks_like_short_vertical_text(
            cell,
            atomic,
            occupied_rows,
        ),
    )

    pieces: list[Cell] = []
    for segment_row_start, segment_row_stop in row_segments:
        occupied_columns = [
            col
            for col in range(col_start, col_stop)
            if any(atomic[row][col].strip() for row in range(segment_row_start, segment_row_stop))
        ]
        column_segments = _segments_at_content(
            col_start,
            col_stop,
            occupied_columns,
            boundary_supported=lambda col: _column_boundary_has_neighbor_support(
                col,
                cell,
                right_sep,
                len(row_edges) - 1,
            ),
        )
        for segment_col_start, segment_col_stop in column_segments:
            text = " ".join(
                atomic[row][col]
                for row in range(segment_row_start, segment_row_stop)
                for col in range(segment_col_start, segment_col_stop)
                if atomic[row][col]
            )
            pieces.append(
                Cell(
                    row=segment_row_start,
                    col=segment_col_start,
                    rowspan=segment_row_stop - segment_row_start,
                    colspan=segment_col_stop - segment_col_start,
                    text=text,
                    bbox=BBox(
                        column_edges[segment_col_start],
                        row_edges[segment_row_start],
                        column_edges[segment_col_stop],
                        row_edges[segment_row_stop],
                    ),
                )
            )
    return pieces


def _segments_at_content(
    start: int,
    stop: int,
    occupied: list[int],
    boundary_supported,
    *,
    preserve_short_vertical_text: bool = False,
) -> list[tuple[int, int]]:
    """在重复记录证据充分时，按 atomic band 中的文字起点分段。

    单凭多个文字锚点无法区分独立记录和多行表头/纵排分组文字。这里同时要求
    高锚点密度和邻接行列的 separator 节奏；两行歧义跨度不会被兜底规则拆开。
    """
    span = stop - start
    if (
        preserve_short_vertical_text
        or span < _MIN_RECORD_REPAIR_SPAN
        or len(occupied) < _MIN_RECORD_REPAIR_SPAN
        or len(occupied) / max(span, 1) < _MIN_RECORD_ANCHOR_DENSITY
    ):
        return [(start, stop)]
    supported = [position for position in occupied[1:] if boundary_supported(position)]
    if len(supported) < _MIN_RECORD_REPAIR_SPAN - 1:
        return [(start, stop)]
    boundaries = [start, *supported, stop]
    return list(zip(boundaries, boundaries[1:], strict=False))


def _row_boundary_has_neighbor_support(
    row: int,
    cell: Cell,
    down_sep: list[list[bool]],
    n_cols: int,
) -> bool:
    boundary = row - 1
    if boundary < 0 or boundary >= len(down_sep):
        return False
    neighbors = [col for col in range(n_cols) if not cell.col <= col < cell.col + cell.colspan]
    return _majority_true(down_sep[boundary][col] for col in neighbors)


def _column_boundary_has_neighbor_support(
    col: int,
    cell: Cell,
    right_sep: list[list[bool]],
    n_rows: int,
) -> bool:
    boundary = col - 1
    if boundary < 0:
        return False
    neighbors = [row for row in range(n_rows) if not cell.row <= row < cell.row + cell.rowspan]
    return _majority_true(
        right_sep[row][boundary] for row in neighbors if boundary < len(right_sep[row])
    )


def _majority_true(values) -> bool:
    observations = list(values)
    return bool(observations) and sum(observations) / len(observations) >= (
        _MIN_NEIGHBOR_SEPARATOR_RATIO
    )


def _looks_like_short_vertical_text(
    cell: Cell,
    atomic: list[list[str]],
    occupied_rows: list[int],
) -> bool:
    """保护由短东亚文字片段组成的纵排分组标签。

    这是字符形态判断，不使用标题或业务字段词表。ASCII 标识、编号和数值记录
    不会命中，因此仍可由高密度记录规则拆分。
    """
    if cell.rowspan < _MIN_RECORD_REPAIR_SPAN or len(occupied_rows) < 3:
        return False
    fragments = [
        "".join(
            atomic[row][col] for col in range(cell.col, cell.col + cell.colspan) if atomic[row][col]
        ).replace(" ", "")
        for row in occupied_rows
    ]
    return all(
        0 < len(fragment) <= 2
        and all(not (character.isascii() and character.isalnum()) for character in fragment)
        for fragment in fragments
    )


@dataclass(frozen=True)
class _LineFragment:
    spans: tuple[NativeSpan, ...]
    bbox: BBox
    source_line: tuple[int, int]


@dataclass(frozen=True)
class _AlignedRowCandidate:
    row: int
    cells: tuple[Cell, ...]
    line_count: int
    max_baseline_delta: float


def _annotate_physical_cell_lines(table: Table, native: NativePage) -> None:
    """Attach native line geometry without changing the outer cell text.

    Span centers decide ownership so a native line that happens to span several
    columns is clipped per cell.  Fragments from separate PDF blocks are merged
    only when their vertical centers describe the same physical baseline.
    """
    for cell in table.origin_cells():
        if cell.bbox is None:
            continue
        fragments: list[_LineFragment] = []
        for line in native.lines:
            selected = tuple(
                fragment
                for span in line.spans
                if (fragment := _span_fragment_in_bbox(span, cell.bbox)) is not None
            )
            if not selected:
                continue
            fragments.append(
                _LineFragment(
                    spans=selected,
                    bbox=_spans_bbox(selected),
                    source_line=(line.block_id, line.line_id),
                )
            )
        cell.physical_lines = _merge_line_fragments(fragments)


def _merge_line_fragments(fragments: list[_LineFragment]) -> list[CellLine]:
    if not fragments:
        return []
    median_height = statistics.median(fragment.bbox.h for fragment in fragments)
    tolerance = max(0.75, median_height * 0.18)
    bands: list[list[_LineFragment]] = []
    for fragment in sorted(fragments, key=lambda item: (item.bbox.cy, item.bbox.x0)):
        if not bands:
            bands.append([fragment])
            continue
        band_center = statistics.median(item.bbox.cy for item in bands[-1])
        if abs(fragment.bbox.cy - band_center) <= tolerance:
            bands[-1].append(fragment)
        else:
            bands.append([fragment])

    physical_lines: list[CellLine] = []
    for band in bands:
        spans = tuple(span for fragment in band for span in fragment.spans)
        text = NativeLine(spans=list(spans)).text
        if not text:
            continue
        physical_lines.append(
            CellLine(
                text=text,
                bbox=_spans_bbox(spans),
                source_lines=tuple(sorted({fragment.source_line for fragment in band})),
            )
        )
    return physical_lines


def _recover_aligned_subrows(table: Table) -> list[AlignedSubrow]:
    """Recover borderless nested records from repeated cross-cell baselines.

    A single outer row is never enough evidence: ordinary wrapped prose can
    align by chance.  Candidate body rows must have the same number of physical
    lines in every column, aligned baselines, and deliberate horizontal slack
    in at least half the columns.  Only a consecutive run with either two
    3+-line rows or three 2+-line rows is promoted.
    """
    candidates: list[_AlignedRowCandidate] = []
    # The renderer and current table contract treat row zero as the header.
    # Keep it intact even when its labels wrap over several physical lines.
    for row_index in range(1, table.n_rows):
        cells = _plain_origin_row(table, row_index)
        if cells is None:
            continue
        line_counts = [len(cell.physical_lines) for cell in cells]
        if not line_counts or min(line_counts) < _MIN_ALIGNED_LINES or len(set(line_counts)) != 1:
            continue
        max_delta = _max_baseline_delta(cells)
        if max_delta is None:
            continue
        compact_columns = sum(_has_deliberate_line_breaks(cell) for cell in cells)
        if compact_columns < (table.n_cols + 1) // 2:
            continue
        candidates.append(
            _AlignedRowCandidate(
                row=row_index,
                cells=tuple(cells),
                line_count=line_counts[0],
                max_baseline_delta=max_delta,
            )
        )

    accepted: list[_AlignedRowCandidate] = []
    for run in _consecutive_candidate_runs(candidates):
        strong_rows = sum(candidate.line_count >= _STRONG_ALIGNED_LINES for candidate in run)
        if len(run) >= _MIN_ALIGNED_ROW_RUN or strong_rows >= _MIN_STRONG_ROWS_IN_RUN:
            accepted.extend(run)

    recovered: list[AlignedSubrow] = []
    for candidate in accepted:
        for subrow_index in range(candidate.line_count):
            lines = [cell.physical_lines[subrow_index] for cell in candidate.cells]
            centers = [line.bbox.cy for line in lines]
            recovered.append(
                AlignedSubrow(
                    parent_row=candidate.row,
                    subrow=subrow_index,
                    values=tuple(line.text for line in lines),
                    bboxes=tuple(line.bbox for line in lines),
                    max_baseline_delta=max(centers) - min(centers),
                )
            )
    return recovered


def _plain_origin_row(table: Table, row_index: int) -> list[Cell] | None:
    if row_index >= len(table.rows) or len(table.rows[row_index]) != table.n_cols:
        return None
    row = table.rows[row_index]
    cells: list[Cell] = []
    for column, cell in enumerate(row):
        if (
            cell is None
            or cell.row != row_index
            or cell.col != column
            or cell.rowspan != 1
            or cell.colspan != 1
            or not cell.text.strip()
            or cell.bbox is None
        ):
            return None
        cells.append(cell)
    return cells


def _max_baseline_delta(cells: list[Cell]) -> float | None:
    lines = [line for cell in cells for line in cell.physical_lines]
    if not lines:
        return None
    tolerance = max(
        _ALIGNED_BASELINE_TOL_MIN,
        statistics.median(line.bbox.h for line in lines) * _ALIGNED_BASELINE_TOL_RATIO,
    )
    worst = 0.0
    for line_index in range(len(cells[0].physical_lines)):
        centers = [cell.physical_lines[line_index].bbox.cy for cell in cells]
        delta = max(centers) - min(centers)
        if delta > tolerance:
            return None
        worst = max(worst, delta)
    return worst


def _has_deliberate_line_breaks(cell: Cell) -> bool:
    if cell.bbox is None or cell.bbox.w <= 0.0 or not cell.physical_lines:
        return False
    return max(line.bbox.w for line in cell.physical_lines) <= (
        cell.bbox.w * _STACK_LINE_MAX_WIDTH_RATIO
    )


def _consecutive_candidate_runs(
    candidates: list[_AlignedRowCandidate],
) -> list[list[_AlignedRowCandidate]]:
    runs: list[list[_AlignedRowCandidate]] = []
    for candidate in candidates:
        if runs and candidate.row == runs[-1][-1].row + 1:
            runs[-1].append(candidate)
        else:
            runs.append([candidate])
    return runs


def _spans_bbox(spans: tuple[NativeSpan, ...]) -> BBox:
    return BBox(
        min(span.bbox.x0 for span in spans),
        min(span.bbox.y0 for span in spans),
        max(span.bbox.x1 for span in spans),
        max(span.bbox.y1 for span in spans),
    )


def _origin_by_atomic_slot(cells) -> dict[tuple[int, int], tuple[int, int]]:
    origins: dict[tuple[int, int], tuple[int, int]] = {}
    for cell in cells:
        for row in range(cell.row, cell.row + cell.rowspan):
            for col in range(cell.col, cell.col + cell.colspan):
                origins[(row, col)] = (cell.row, cell.col)
    return origins


def _span_atomic_fragments(
    span: NativeSpan,
    row_edges: list[float],
    column_edges: list[float],
    origin_by_slot: dict[tuple[int, int], tuple[int, int]],
) -> list[tuple[int, int, str]]:
    """Split a span only when positioned characters cross real cell origins.

    PDF generators occasionally emit several visual cells as one text span.
    Whole-span center ownership then collapses every label into one cell.  A
    true colspan/rowspan has the same origin for all covered atomic slots and
    deliberately remains untouched here.
    """
    fallback_row = _bin(span.bbox.cy, row_edges)
    fallback_col = _bin(span.bbox.cx, column_edges)
    fallback = (
        [(fallback_row, fallback_col, span.text)]
        if fallback_row is not None and fallback_col is not None
        else []
    )
    if not span.characters:
        return fallback

    grouped: dict[tuple[int, int], list[NativeChar]] = {}
    for character in span.characters:
        row = _strict_bin(character.bbox.cy, row_edges)
        col = _strict_bin(character.bbox.cx, column_edges)
        if row is None or col is None:
            continue
        grouped.setdefault((row, col), []).append(character)
    visible_groups = {
        slot: characters
        for slot, characters in grouped.items()
        if any(character.text.strip() for character in characters)
    }
    origins = {origin_by_slot[slot] for slot in visible_groups if slot in origin_by_slot}
    if len(origins) < 2:
        return fallback

    fragments: list[tuple[int, int, str]] = []
    for (row, col), characters in sorted(visible_groups.items()):
        text = "".join(character.text for character in characters).strip()
        if text:
            fragments.append((row, col, text))
    return fragments or fallback


def _span_fragment_in_bbox(span: NativeSpan, bbox: BBox) -> NativeSpan | None:
    if not span.text.strip():
        return None
    if _bbox_inside(span.bbox, bbox):
        return span
    if not span.characters:
        return span if _span_centered_in(span, bbox) else None
    characters = [
        character
        for character in span.characters
        if bbox.x0 - 1.0 <= character.bbox.cx <= bbox.x1 + 1.0
        and bbox.y0 - 1.0 <= character.bbox.cy <= bbox.y1 + 1.0
    ]
    text = "".join(character.text for character in characters).strip()
    visible = [character for character in characters if character.text.strip()]
    if not text or not visible:
        return None
    return NativeSpan(
        text=text,
        bbox=BBox(
            min(character.bbox.x0 for character in visible),
            min(character.bbox.y0 for character in visible),
            max(character.bbox.x1 for character in visible),
            max(character.bbox.y1 for character in visible),
        ),
        font=span.font,
        size=span.size,
        flags=span.flags,
        block_id=span.block_id,
        line_id=span.line_id,
        span_id=span.span_id,
        characters=characters,
    )


def _bbox_inside(inner: BBox, outer: BBox, margin: float = 1.0) -> bool:
    return (
        inner.x0 >= outer.x0 - margin
        and inner.x1 <= outer.x1 + margin
        and inner.y0 >= outer.y0 - margin
        and inner.y1 <= outer.y1 + margin
    )


def _span_centered_in(span: NativeSpan, bbox: BBox, margin: float = 1.0) -> bool:
    return (
        bbox.x0 - margin <= span.bbox.cx <= bbox.x1 + margin
        and bbox.y0 - margin <= span.bbox.cy <= bbox.y1 + margin
    )


def _strict_bin(value: float, edges: list[float], margin: float = 1.0) -> int | None:
    if len(edges) < 2 or value < edges[0] - margin or value > edges[-1] + margin:
        return None
    if value <= edges[0]:
        return 0
    if value >= edges[-1]:
        return len(edges) - 2
    for index, (left, right) in enumerate(zip(edges, edges[1:], strict=False)):
        if left <= value < right:
            return index
    return None


def _iter_spans(native: NativePage, roi: BBox):
    for ln in native.lines:
        for sp in ln.spans:
            if sp.text.strip() and _span_inside(sp, roi):
                yield sp


def _span_inside(sp: NativeSpan, roi: BBox, margin: float = 2.0) -> bool:
    return (
        sp.bbox.x0 >= roi.x0 - margin
        and sp.bbox.x1 <= roi.x1 + margin
        and sp.bbox.y0 >= roi.y0 - margin
        and sp.bbox.y1 <= roi.y1 + margin
    )


def _bin(value: float, edges: list[float]) -> int | None:
    for i in range(len(edges) - 1):
        if edges[i] - 2.0 <= value <= edges[i + 1] + 2.0:
            return i
    return None


def _unique_sorted(vals: list[float]) -> list[float]:
    out = []
    for v in sorted(vals):
        if not out or v - out[-1] > 1.0:
            out.append(v)
    return out


def _with_missing_outer_edges(
    centers: list[float], low: float, high: float, tolerance: float = 8.0
) -> list[float]:
    """仅当 vector 层确实缺外框时才借用 PP ROI 边界。

    有些 PDF 把粗外框画成普通矩形，规则线提取只能看到内部 separator；另一些
    PDF 又有完整外框。无条件加入 ROI 会产生空首尾行，无条件不用则会少两列。
    """
    values = sorted(centers)
    if not values:
        return [low, high]
    if values[0] - low > tolerance:
        values.insert(0, low)
    if high - values[-1] > tolerance:
        values.append(high)
    return values
