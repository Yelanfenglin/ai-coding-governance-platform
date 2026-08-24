"""PDF 原生信息提取——v1 版。

只忠实保留 PDF 的原生层次，不做任何 layout 推断：

    page.get_text("dict")
      └── blocks[ ]
            └── lines[ ]
                 └── spans[ ]   (font / size / flags / text / bbox)

NativeLine 来自 PDF 原生 line 本身，绝不再按 y 重叠把左右栏/同高文字拼成一条。
（v0 正是这一步把 P6 左右栏粘成一个 line，信息在进入任何逻辑前就毁掉了。）

同时修掉 v0 的核心 bug：`get_text("words")` 返回的是
`(x0,y0,x1,y1,word,block_no,line_no,word_no)` 8 元组，不是 11 元组，所以
之前把 block_no 当 font、line_no 当 size、word_no 当 flags 全是错的。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import fitz

from .models import BBox, NativeDrawing, NativeImage, VectorRule


@dataclass
class NativeChar:
    """One positioned character retained for cell-level table assignment."""

    text: str
    bbox: BBox


@dataclass
class NativeSpan:
    text: str
    bbox: BBox
    font: str = ""
    size: float = 0.0
    flags: int = 0
    block_id: int = -1
    line_id: int = -1
    span_id: int = -1
    characters: list[NativeChar] = field(default_factory=list)

    @property
    def dominant_style(self) -> tuple[float, int]:
        return (self.size, self.flags)


@dataclass
class NativeLine:
    """对应 PDF 原生的一行（同一条 dict line，可能跨多个 span）。"""

    spans: list[NativeSpan] = field(default_factory=list)
    block_id: int = -1
    line_id: int = -1
    bbox: BBox = field(default_factory=lambda: BBox(0, 0, 0, 0))

    @property
    def text(self) -> str:
        return _join_spans(self.spans)

    @property
    def font(self) -> str:
        return max(self.spans, key=lambda s: len(s.text)).font if self.spans else ""

    @property
    def size(self) -> float:
        if not self.spans:
            return 0.0
        # 取覆盖最长文本的 span 字号（避免空 span 干扰）
        return max((s.size for s in self.spans if s.text.strip()), default=0.0)

    @property
    def flags(self) -> int:
        return max((s.flags for s in self.spans), default=0)


@dataclass
class NativeBlock:
    block_id: int
    kind: str  # 0=text, 1=image
    bbox: BBox
    lines: list[NativeLine] = field(default_factory=list)


@dataclass
class NativePage:
    page_number: int
    width: float
    height: float
    blocks: list[NativeBlock] = field(default_factory=list)
    lines: list[NativeLine] = field(default_factory=list)
    words: list = field(default_factory=list)  # 保留可用 word 级位置（可选）
    rules: list[VectorRule] = field(default_factory=list)
    # 旧版高置信 rule 单独保留，只冻结 table/snapshot 分类。None 表示调用方
    # 没有提供独立分类证据（主要用于单元测试构造的 NativePage）。
    classification_rules: list[VectorRule] | None = None
    images: list[NativeImage] = field(default_factory=list)
    drawings: list[NativeDrawing] = field(default_factory=list)


# ---------------------------------------------------------------------------
# rules 提取（表格几何用）——沿用 v0 的确定性归一化
# ---------------------------------------------------------------------------
_ORTHO_DEG = 4.5
_MIN_RULE_LEN = 3.0
# 一些 PDF 生成器不用 stroke 绘制表格，而是把每段边框输出成填充矩形。
# 这类矩形的厚度通常只有几个 PDF point，但短边框的长宽比未必很大；仅靠
# aspect ratio 会系统性漏掉它们。这里同时约束绝对厚度和最小细长程度：
# 前者排除单元格底色，后者排除边框交点处的小方块。
_MAX_RULE_THICKNESS = 3.0
_MIN_RULE_ASPECT = 3.0
_CLASSIFICATION_RECT_ASPECT = 20.0


def _seg_to_rules(p0, p1, out):
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < _MIN_RULE_LEN:
        return
    angle = abs(math.degrees(math.atan2(abs(dy), abs(dx))))
    bx0, bx1 = (x0, x1) if x0 <= x1 else (x1, x0)
    by0, by1 = (y0, y1) if y0 <= y1 else (y1, y0)
    if angle < _ORTHO_DEG or angle > 180 - _ORTHO_DEG:
        out.append(VectorRule(BBox(bx0, by0 - 0.3, bx1, by1 + 0.3), horizontal=True))
    elif abs(angle - 90) < _ORTHO_DEG:
        out.append(VectorRule(BBox(bx0 - 0.3, by0, bx1 + 0.3, by1), horizontal=False))


def _extract_rules(page) -> list[VectorRule]:
    rules, _ = _extract_rule_sets(page)
    return rules


def _extract_rule_sets(page, drawings=None) -> tuple[list[VectorRule], list[VectorRule]]:
    """返回拓扑 rule 与冻结分类用的高置信 rule。"""
    rules: list[VectorRule] = []
    classification_rules: list[VectorRule] = []
    for d in drawings if drawings is not None else page.get_drawings():
        for it in d["items"]:
            t = it[0]
            if t == "l":
                _seg_to_rules((it[1].x, it[1].y), (it[2].x, it[2].y), rules)
                _seg_to_rules(
                    (it[1].x, it[1].y),
                    (it[2].x, it[2].y),
                    classification_rules,
                )
            elif t == "re":
                r = it[1]
                rect_rule = _filled_rect_rule(r)
                if rect_rule is not None:
                    rules.append(rect_rule)
                classification_rule = _classification_rect_rule(r)
                if classification_rule is not None:
                    classification_rules.append(classification_rule)
            elif t == "c":
                pts = it[1]
                if pts and len(pts) == 4 and _bezier_flat(pts):
                    _seg_to_rules((pts[0].x, pts[0].y), (pts[3].x, pts[3].y), rules)
                    _seg_to_rules(
                        (pts[0].x, pts[0].y),
                        (pts[3].x, pts[3].y),
                        classification_rules,
                    )
            elif t == "qu":
                pts = it[1]
                if pts and len(pts) == 3 and _bezier_flat([pts[0], pts[0], pts[1], pts[2]]):
                    _seg_to_rules((pts[0].x, pts[0].y), (pts[2].x, pts[2].y), rules)
                    _seg_to_rules(
                        (pts[0].x, pts[0].y),
                        (pts[2].x, pts[2].y),
                        classification_rules,
                    )
    return rules, classification_rules


def _filled_rect_rule(rect) -> VectorRule | None:
    """把细长填充矩形恢复成一条几何 rule。

    判定只使用 PDF point 几何，不依赖颜色、文档来源或单元格文字。短边必须
    足够薄，长边还要同时满足绝对长度和相对细长程度；因此大面积底色和近似
    方形的交点块不会进入表格边界集合。
    """
    width, height = float(rect.width), float(rect.height)
    short, long = min(width, height), max(width, height)
    if (
        short <= 0.0
        or short > _MAX_RULE_THICKNESS
        or long < _MIN_RULE_LEN
        or long < short * _MIN_RULE_ASPECT
    ):
        return None
    if width >= height:
        return VectorRule(
            BBox(rect.x0, rect.y0 - 0.3, rect.x1, rect.y0 + 0.3),
            horizontal=True,
            thickness=height,
        )
    return VectorRule(
        BBox(rect.x0 - 0.3, rect.y0, rect.x1 + 0.3, rect.y1),
        horizontal=False,
        thickness=width,
    )


def _classification_rect_rule(rect) -> VectorRule | None:
    """复现旧版的严格矩形判定，仅用于冻结 table/snapshot 分类。"""
    width, height = float(rect.width), float(rect.height)
    if width <= 0.0 or height <= 0.0:
        return None
    if width >= height and width / height >= _CLASSIFICATION_RECT_ASPECT:
        return VectorRule(
            BBox(rect.x0, rect.y0 - 0.3, rect.x1, rect.y0 + 0.3),
            horizontal=True,
            thickness=height,
        )
    if height > width and height / width >= _CLASSIFICATION_RECT_ASPECT:
        return VectorRule(
            BBox(rect.x0 - 0.3, rect.y0, rect.x1 + 0.3, rect.y1),
            horizontal=False,
            thickness=width,
        )
    return None


def _bezier_flat(pts) -> bool:
    (p0, p1, p2, p3) = pts

    def dist(a, b, pp):
        dx, dy = b[0] - a[0], b[1] - a[1]
        L2 = dx * dx + dy * dy
        if L2 == 0:
            return math.hypot(pp[0] - a[0], pp[1] - a[1])
        t = ((pp[0] - a[0]) * dx + (pp[1] - a[1]) * dy) / L2
        t = max(0.0, min(1.0, t))
        px, py = a[0] + t * dx, a[1] + t * dy
        return math.hypot(pp[0] - px, pp[1] - py)

    return all(dist(p0, p3, point) <= 1.0 for point in (p1, p2))


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def extract_native_page(page: fitz.Page) -> NativePage:
    # rawdict 保留逐字符 bbox。很多规格书用一长串前导空格把页眉、页码或
    # 表格文字推到目标位置；若直接采用 span bbox，这些不可见空格会让一行
    # 看起来横跨半页，进而无法绑定到 PP 的 header/footer anchor。
    d = page.get_text("rawdict")
    npg = NativePage(
        page_number=page.number + 1,
        width=page.rect.width,
        height=page.rect.height,
    )
    all_lines: list[NativeLine] = []
    for bi, block in enumerate(d["blocks"]):
        btype = block.get("type", 0)
        bb = block.get("bbox")
        nblock = NativeBlock(
            block_id=bi,
            kind="image" if btype == 1 else "text",
            bbox=BBox(bb[0], bb[1], bb[2], bb[3]) if bb else BBox(0, 0, 0, 0),
        )
        if btype == 1:
            npg.images.append(NativeImage(bbox=nblock.bbox, xref=0, embedded=True))
        block_lines: list[NativeLine] = []
        for li, line in enumerate(block.get("lines", [])):
            line_bbox = line["bbox"]
            ln = NativeLine(
                block_id=bi,
                line_id=li,
                bbox=BBox(line_bbox[0], line_bbox[1], line_bbox[2], line_bbox[3]),
            )
            for si, span in enumerate(line["spans"]):
                chars = span.get("chars", [])
                text = "".join(ch.get("c", "") for ch in chars)
                if not chars:
                    text = span.get("text", "")
                sb = _visible_span_bbox(span, chars)
                ln.spans.append(
                    NativeSpan(
                        text=text,
                        bbox=sb,
                        font=span.get("font", ""),
                        size=span.get("size", 0.0),
                        flags=span.get("flags", 0),
                        block_id=bi,
                        line_id=li,
                        span_id=si,
                        characters=[
                            NativeChar(
                                text=char.get("c", ""),
                                bbox=BBox(*char["bbox"]),
                            )
                            for char in chars
                            if char.get("bbox") is not None
                        ],
                    )
                )
            if ln.spans:
                ln.bbox = _merge_span_bbox(ln.spans)
            all_lines.append(ln)
            block_lines.append(ln)
        nblock.lines = block_lines
        npg.blocks.append(nblock)

    npg.lines = all_lines
    drawings = page.get_drawings()
    npg.drawings = [
        NativeDrawing(
            bbox=BBox(
                float(drawing["rect"].x0),
                float(drawing["rect"].y0),
                float(drawing["rect"].x1),
                float(drawing["rect"].y1),
            ),
            item_count=len(drawing.get("items", [])),
            filled=drawing.get("fill") is not None,
        )
        for drawing in drawings
        if drawing.get("rect") is not None
    ]
    npg.rules, npg.classification_rules = _extract_rule_sets(page, drawings)
    return npg


def _merge_span_bbox(spans: list[NativeSpan]) -> BBox:
    visible = [s for s in spans if s.text.strip()] or spans
    x0 = min(s.bbox.x0 for s in visible)
    y0 = min(s.bbox.y0 for s in visible)
    x1 = max(s.bbox.x1 for s in visible)
    y1 = max(s.bbox.y1 for s in visible)
    return BBox(x0, y0, x1, y1)


def _visible_span_bbox(span: dict, chars: list[dict]) -> BBox:
    """返回 span 中可见字符的紧 bbox；全空白 span 才退回原 bbox。"""
    visible = [ch for ch in chars if ch.get("c", "").strip()]
    if visible:
        return BBox(
            min(ch["bbox"][0] for ch in visible),
            min(ch["bbox"][1] for ch in visible),
            max(ch["bbox"][2] for ch in visible),
            max(ch["bbox"][3] for ch in visible),
        )
    sb = span.get("bbox", (0.0, 0.0, 0.0, 0.0))
    return BBox(sb[0], sb[1], sb[2], sb[3])


def _join_spans(spans: list[NativeSpan]) -> str:
    """按几何间距拼接 span，同时去掉只用于定位的首尾空白。"""
    parts: list[str] = []
    previous: NativeSpan | None = None
    for span in sorted((s for s in spans if s.text.strip()), key=lambda s: s.bbox.x0):
        text = span.text.strip()
        if previous is not None and parts:
            gap = span.bbox.x0 - previous.bbox.x1
            font_size = max(previous.size, span.size, 1.0)
            left = parts[-1][-1] if parts[-1] else ""
            right = text[0] if text else ""
            explicit_space = previous.text[-1:].isspace() or span.text[:1].isspace()
            ascii_word_boundary = (
                left.isascii() and right.isascii() and left.isalnum() and right.isalnum()
            )
            if (
                explicit_space
                or (ascii_word_boundary and gap > 0.12 * font_size)
                or (gap > 0.55 * font_size and (left.isascii() or right.isascii()))
            ) and not parts[-1].endswith(" "):
                parts.append(" ")
        parts.append(text)
        previous = span
    return "".join(parts).strip()
