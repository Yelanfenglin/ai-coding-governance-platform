"""PP-DocLayoutV3 proposal → Anchor，并把 native lines 绑定到最匹配 anchor。

含义（v1 关键变化）：
- PP proposal 不再只是「用来猜栏目」的 hint，而是**页面 block 的 anchor**。
- 绑定方向是「NativeLine 找最匹配的 PP anchor」，而非「PP bbox 里截取文字」。
- 若一条 native line 一个 anchor 都不匹配（PP 漏检），就自生成一个 TextAnchor。
  这样每条 line 天生只属于一个 owner，**无需 unowned 二次补充**。

匹配度量：A = line ∩ anchor 面积 / line 面积（line 有多少比例落在 anchor 内）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import BBox, LayoutProposal
from .native_pdf import NativeLine, NativePage


# ---- Anchor 类型（轻量，不做复杂分类体系）----
@dataclass
class Anchor:
    anchor_id: int
    bbox: BBox
    pp_label: str = "text"
    pp_order: int | None = None
    kind: str = "TEXT"  # TEXT | HEADING | VISUAL | TABLE | NOISE
    lines: list[NativeLine] = field(default_factory=list)
    source: str = "pp"  # pp | auto | native_table
    table: object | None = None
    table_upgraded: bool = False

    @property
    def text(self) -> str:
        return " | ".join(line.text for line in self.lines if line.text.strip())


def _anchor_kind(label: str) -> str:
    lbl = (label or "").lower()
    if lbl in ("header", "header_image", "footer", "number"):
        return "NOISE"
    if lbl == "table":
        return "TABLE"
    if lbl in ("image", "chart"):
        return "VISUAL"
    if lbl == "paragraph_title":
        return "HEADING"
    return "TEXT"


def make_anchors(
    proposals: list[LayoutProposal],
) -> list[Anchor]:
    """把每页 PP proposals 转成 anchors。"""
    anchors = []
    for i, p in enumerate(proposals):
        anchors.append(
            Anchor(
                anchor_id=i,
                bbox=p.bbox,
                pp_label=p.label,
                pp_order=p.order,
                kind=_anchor_kind(p.label),
            )
        )
    return anchors


def add_recovered_table_anchors(anchors: list[Anchor], tables: list[object]) -> list[Anchor]:
    """把 native ruled-grid 补提结果提升为真正 anchor，参与唯一 ownership。

    先补 anchor、后绑定 native line，可避免旧流程在 PageIR 生成后硬插表格，
    导致同一份寄存器位表同时出现在 TABLE 和 TEXT 中。
    """
    for table in tables:
        bbox = getattr(table, "bbox", None)
        if bbox is None:
            continue
        anchors.append(
            Anchor(
                anchor_id=_new_id(anchors),
                bbox=bbox,
                pp_label="native_table",
                pp_order=None,
                kind="TABLE",
                source="native_table",
                table=table,
            )
        )
    return anchors


def _intersection_area(a: BBox, b: BBox) -> float:
    x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
    x1, y1 = min(a.x1, b.x1), min(a.y1, b.y1)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _line_inside_ratio(line: NativeLine, anchor: BBox) -> float:
    if line.bbox.area <= 0:
        return 0.0
    return _intersection_area(line.bbox, anchor) / line.bbox.area


def bind_native_lines(
    native: NativePage,
    anchors: list[Anchor],
    mode: str = "line",
) -> list[Anchor]:
    """把 native lines 绑定到 anchors。

    每条 line 绑定到 overlap 最大的 anchor。若最佳 overlap < min_ratio（基本不重叠），
    说明 PP 漏检该行，则自生成 TextAnchor。
    """
    # 按 line 中心 y 给 anchors 排序存储，方便生成无主 anchor 时插入正确位置
    anchors.sort(key=lambda a: (a.bbox.y0, a.bbox.x0))

    for line in native.lines:
        if not line.text.strip():
            continue
        best_idx = None
        best_key = (0, 0, 0.0)
        for ai, a in enumerate(anchors):
            r = _line_inside_ratio(line, a.bbox)
            centered = _line_centered_in(line, a.bbox)
            # TABLE/VISUAL/NOISE 是排他区域：行中心落入时由它优先拥有。
            # 这既剔除宽空白导致的页眉页脚，也阻止表内文字重复进入正文。
            if a.kind in ("NOISE", "TABLE", "VISUAL") and (centered or r >= 0.25):
                priority = 3 if a.kind == "NOISE" else 2
                key = (2, priority, r)
            elif r >= 0.45:
                key = (1, 1 if a.kind == "HEADING" else 0, r)
            else:
                continue
            if key > best_key:
                best_key, best_idx = key, ai
        if best_idx is None and _is_margin_noise(line, native.height):
            noise_indices = [i for i, anchor in enumerate(anchors) if anchor.kind == "NOISE"]
            if noise_indices:
                best_idx = min(
                    noise_indices,
                    key=lambda index: abs(anchors[index].bbox.cy - line.bbox.cy),
                )
            else:
                anchors.append(
                    Anchor(
                        anchor_id=_new_id(anchors),
                        bbox=line.bbox,
                        pp_label="(auto-margin-noise)",
                        pp_order=None,
                        kind="NOISE",
                        source="auto",
                    )
                )
                best_idx = len(anchors) - 1
        if best_idx is not None:
            anchors[best_idx].lines.append(line)
        else:
            # PP 漏检 → 自生成 TextAnchor
            new_anchor = Anchor(
                anchor_id=_new_id(anchors),
                bbox=line.bbox,
                pp_label="(auto-text)",
                pp_order=None,
                kind="TEXT",
                lines=[line],
                source="auto",
            )
            _insert_by_position(anchors, new_anchor)
    return anchors


def _new_id(anchors: list[Anchor]) -> int:
    return max((a.anchor_id for a in anchors), default=-1) + 1


def _insert_by_position(anchors: list[Anchor], anchor: Anchor):
    """按 y 位置把自动生成的 anchor 插入排序列表。"""
    # re-sort whole list by (y0,x0) 保持稳定
    anchors.append(anchor)
    anchors.sort(key=lambda a: (a.bbox.y0, a.bbox.x0))


def _line_centered_in(line: NativeLine, bbox: BBox, margin: float = 2.5) -> bool:
    return (
        bbox.x0 - margin <= line.bbox.cx <= bbox.x1 + margin
        and bbox.y0 - margin <= line.bbox.cy <= bbox.y1 + margin
    )


def _is_margin_noise(line: NativeLine, page_height: float) -> bool:
    """PP 偶尔只报页码不报同一原生行上的 URL；极窄页边带作最终噪声兜底。"""
    return line.bbox.cy <= page_height * 0.09 or line.bbox.cy >= page_height * 0.92
