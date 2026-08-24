"""PP 漏表的高精度 ruled-grid 补提（NativeTable fallback）。

PP-DocLayoutV3 偶尔会漏掉嵌在正文中的规则表格。本模块用原生 vector
网格、重复二维记录和邻近语义块联合补提，不依赖表内固定字段词表。

设计（宁漏勿错）：
- 只在「明显形成封闭矩形网格」的区域触发
- 已与 PP 任一 table / image / chart proposal 有较大重叠的区域一律跳过
- 候选区域走 reconstruct_table 的二维记录验证，成功才采纳
- 不做全页 vector 分类，不做置信度
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import BBox
from .native_pdf import NativePage
from .table import (
    Table,
    connected_rule_components,
    reconstruct_table,
    rule_boundary_counts,
    rules_bbox,
)

# 常规漏表至少要有 4 条聚类行边界 + 4 条聚类列边界。增强边线可以
# 额外恢复嵌在正文中的 3x2 小表，但必须再满足重复记录和局部上下文证据。
_MIN_ROWS = 4
_MIN_COLS = 4
_MIN_COMPACT_ROWS = 4
_MIN_COMPACT_COLS = 3
# 与 PP proposal 重叠比例超过该值则跳过
_SKIP_OVERLAP = 0.4
_CONTEXT_VERTICAL_GAP_RATIO = 0.22
_CONTEXT_HORIZONTAL_OVERLAP = 0.2


@dataclass
class GridProposal:
    bbox: BBox
    n_rows: int
    n_cols: int
    source: str = "classification"


def _cluster(vals: list, gap: float) -> list[list[float]]:
    vals = sorted(vals)
    if not vals:
        return []
    out = [[vals[0]]]
    for x in vals[1:]:
        if x - out[-1][-1] <= gap:
            out[-1].append(x)
        else:
            out.append([x])
    return out


def find_grid_proposals(
    native: NativePage,
    skip_bboxes: list[BBox],
    margin: float = 8.0,
) -> list[GridProposal]:
    """在全页找「PP 未覆盖的规则网格」候选区域。"""

    proposals: list[GridProposal] = []
    # 必须按线段真实相交关系分组件；独立聚类 x/y 再做笛卡尔积会把一页上
    # 相距很远的表、波形和页框合成 y=76..790 的伪整页表。
    rule_sets = [
        (
            native.classification_rules
            if native.classification_rules is not None
            else native.rules,
            _MIN_ROWS,
            _MIN_COLS,
            "classification",
        )
    ]
    if native.classification_rules is not None:
        rule_sets.append(
            (
                native.rules,
                _MIN_COMPACT_ROWS,
                _MIN_COMPACT_COLS,
                "enhanced",
            )
        )

    for detection_rules, min_rows, min_cols, source in rule_sets:
        for component in connected_rule_components(detection_rules):
            n_rows, n_cols = rule_boundary_counts(component)
            if n_rows < min_rows or n_cols < min_cols:
                continue
            bbox = rules_bbox(component)
            if bbox is None or bbox.w <= 0 or bbox.h <= 0:
                continue
            if _overlaps_any(bbox, skip_bboxes):
                continue
            proposal = GridProposal(bbox=bbox, n_rows=n_rows, n_cols=n_cols, source=source)
            if any(_same_region(proposal.bbox, known.bbox) for known in proposals):
                continue
            proposals.append(proposal)
    return proposals


def _same_region(left: BBox, right: BBox) -> bool:
    intersection = _inter_area(left, right)
    return intersection / max(min(left.area, right.area), 1e-6) >= 0.8


def _overlaps_any(b: BBox, skip_bboxes: list[BBox]) -> bool:
    for s in skip_bboxes:
        inter = _inter_area(b, s)
        if inter / max(b.area, 1e-6) >= _SKIP_OVERLAP:
            return True
    return False


def _inter_area(a: BBox, b: BBox) -> float:
    x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
    x1, y1 = min(a.x1, b.x1), min(a.y1, b.y1)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def recover_missing_tables(
    native: NativePage,
    covered_bboxes: list[BBox],
    page_width: float,
    page_height: float,
    context_bboxes: list[BBox] | None = None,
) -> list[Table]:
    """补提 PP 漏掉的真表格。covered_bboxes = 已由 PP table/image/chart 占据的区域。

    宁漏勿错：
    - 候选必须形成至少三行、两列重复记录；
    - 候选必须紧邻已有标题、正文或表格上下文；
    - 已由 PP 标成 table/image/chart 的区域不在这里重新分类。
    """
    proposals = find_grid_proposals(native, covered_bboxes)
    tables: list[Table] = []
    seen: list[BBox] = []
    for p in proposals:
        t = reconstruct_table(
            p.bbox,
            native,
            validate_with_topology=p.source == "enhanced",
        )
        if t is None or not t.valid:
            continue
        if t.record_rows < 3 or t.recurring_columns < 2:
            continue
        if not _has_local_context(
            p.bbox,
            context_bboxes or [],
            page_width,
            page_height,
        ):
            continue
        if any(_overlaps_any(p.bbox, [s]) for s in seen):
            continue
        tables.append(t)
        seen.append(p.bbox)
    return tables


def _has_local_context(
    candidate: BBox,
    contexts: list[BBox],
    page_width: float,
    page_height: float,
) -> bool:
    """Whether a compact grid belongs to a nearby semantic block.

    Context is geometric rather than lexical: a preceding heading, paragraph,
    or structured table must overlap the candidate's horizontal reading band
    and be close enough vertically.  This lets a small embedded value table be
    recovered without teaching the parser any chip-specific vocabulary.
    """
    max_gap = page_height * _CONTEXT_VERTICAL_GAP_RATIO
    for context in contexts:
        if context.y1 > candidate.y0 + 4.0:
            continue
        gap = candidate.y0 - context.y1
        if gap > max_gap:
            continue
        overlap = max(0.0, min(candidate.x1, context.x1) - max(candidate.x0, context.x0))
        denominator = max(min(candidate.w, context.w, page_width), 1e-6)
        if overlap / denominator >= _CONTEXT_HORIZONTAL_OVERLAP:
            return True
    return False
