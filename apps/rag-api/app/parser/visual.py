"""Conservative native-vector recovery and snapshot boundary refinement."""

from __future__ import annotations

from .anchor import Anchor
from .models import BBox, NativeDrawing
from .native_pdf import NativeLine, NativePage
from .table import Table

_TOUCH_TOLERANCE = 2.5
_SNAPSHOT_PADDING_X = 4.0
_SNAPSHOT_PADDING_Y = 3.0
_SNAPSHOT_LABEL_Y_REACH = 4.0
_MIN_SEED_PATHS = 4
_MIN_SEED_SIDE = 20.0
_MIN_TEXT_LINES = 4
_NEARBY_COMPONENT_GAP_RATIO = 0.04
_NEARBY_HORIZONTAL_OVERLAP = 0.2
_LINE_HORIZONTAL_REACH_RATIO = 0.3
_EXCLUSIVE_OVERLAP = 0.35
_NEIGHBOR_BAND_OVERLAP = 0.25


def refine_snapshot_anchors(native: NativePage, anchors: list[Anchor]) -> None:
    """Expand failed-table and visual anchors to include clipped peer labels.

    Layout models often detect the central drawing but omit labels that end
    exactly at, or slightly outside, the predicted box.  Expansion is driven
    by same-band native text and overlapping PDF vector/image geometry.  It
    does not inspect diagram titles or cell vocabulary.
    """
    original_boxes = [anchor.bbox for anchor in anchors]
    for anchor_index, anchor in enumerate(anchors):
        if not _is_snapshot_anchor(anchor):
            continue
        context_boxes = [
            other.bbox
            for index, other in enumerate(anchors)
            if index != anchor_index and other.kind in ("TEXT", "HEADING")
        ]
        geometry = [anchor.bbox]
        geometry.extend(
            drawing.bbox
            for drawing in native.drawings
            if _boxes_overlap(drawing.bbox, anchor.bbox, margin=2.0)
        )
        geometry.extend(
            image.bbox
            for image in native.images
            if _boxes_overlap(image.bbox, anchor.bbox, margin=2.0)
        )
        vertical_band = _union_bbox(geometry)
        peer_lines = [
            line
            for line in native.lines
            if line.text.strip()
            and _vertical_gap(line.bbox, vertical_band) <= _SNAPSHOT_LABEL_Y_REACH
            and (
                _vertical_gap(line.bbox, vertical_band) <= 0.1
                or not _line_claimed_by_context(line, context_boxes)
            )
            and _horizontal_gap(line.bbox, vertical_band)
            <= native.width * _LINE_HORIZONTAL_REACH_RATIO
        ]
        candidate = _padded_bbox(
            _union_bbox([*geometry, *(line.bbox for line in peer_lines)]),
            native.width,
            native.height,
        )
        anchor.bbox = _clip_at_neighbor_midlines(
            candidate,
            vertical_band,
            [
                bbox
                for index, bbox in enumerate(original_boxes)
                if index != anchor_index
                and anchors[index].kind in ("TABLE", "VISUAL")
            ],
            vertical_context=context_boxes,
        )


def add_recovered_visual_anchors(native: NativePage, anchors: list[Anchor]) -> list[Anchor]:
    """Recover unmodeled vector diagrams from repeated drawing structure.

    A seed needs several connected vector paths, meaningful two-dimensional
    extent, and multiple native text labels.  Nearby disconnected vector
    components are joined only when they share a horizontal band and a small
    vertical gap; this captures waveforms made of separate rows without using
    document-specific captions.
    """
    drawing_components = _drawing_components(
        [drawing for drawing in native.drawings if not drawing.filled]
    )
    exclusive = [anchor.bbox for anchor in anchors if anchor.kind in ("TABLE", "VISUAL", "NOISE")]
    recovered: list[BBox] = []
    max_gap = native.height * _NEARBY_COMPONENT_GAP_RATIO

    for seed_index, seed in enumerate(drawing_components):
        seed_bbox = _drawing_union(seed)
        path_count = sum(max(1, drawing.item_count) for drawing in seed)
        if (
            path_count < _MIN_SEED_PATHS
            or seed_bbox.w < _MIN_SEED_SIDE
            or seed_bbox.h < _MIN_SEED_SIDE
            or seed_bbox.cy < native.height * 0.09
            or seed_bbox.cy > native.height * 0.92
        ):
            continue

        selected = {seed_index}
        candidate = seed_bbox
        changed = True
        while changed:
            changed = False
            for index, component in enumerate(drawing_components):
                if index in selected:
                    continue
                component_bbox = _drawing_union(component)
                if _components_are_near(candidate, component_bbox, max_gap):
                    selected.add(index)
                    candidate = _union_bbox([candidate, component_bbox])
                    changed = True

        if any(_substantial_overlap(candidate, known) for known in exclusive):
            continue

        peer_lines = _peer_lines(native, candidate)
        if len(peer_lines) < _MIN_TEXT_LINES:
            continue
        final_bbox = _padded_bbox(
            _union_bbox([candidate, *(line.bbox for line in peer_lines)]),
            native.width,
            native.height,
        )
        if any(_substantial_overlap(final_bbox, known) for known in [*exclusive, *recovered]):
            continue

        anchors.append(
            Anchor(
                anchor_id=max((anchor.anchor_id for anchor in anchors), default=-1) + 1,
                bbox=final_bbox,
                pp_label="image",
                pp_order=None,
                kind="VISUAL",
                source="native_visual",
            )
        )
        recovered.append(final_bbox)
        exclusive.append(final_bbox)
    return anchors


def _is_snapshot_anchor(anchor: Anchor) -> bool:
    if anchor.kind == "VISUAL":
        return True
    return anchor.kind == "TABLE" and not (isinstance(anchor.table, Table) and anchor.table.valid)


def _drawing_components(drawings: list[NativeDrawing]) -> list[list[NativeDrawing]]:
    if not drawings:
        return []
    parent = list(range(len(drawings)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, drawing in enumerate(drawings):
        for right in range(left + 1, len(drawings)):
            if _boxes_overlap(
                drawing.bbox,
                drawings[right].bbox,
                margin=_TOUCH_TOLERANCE,
            ):
                union(left, right)

    groups: dict[int, list[NativeDrawing]] = {}
    for index, drawing in enumerate(drawings):
        groups.setdefault(find(index), []).append(drawing)
    return list(groups.values())


def _components_are_near(left: BBox, right: BBox, max_gap: float) -> bool:
    vertical_gap = max(0.0, max(left.y0, right.y0) - min(left.y1, right.y1))
    if vertical_gap > max_gap:
        return False
    overlap = max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))
    return overlap / max(min(left.w, right.w), 1e-6) >= _NEARBY_HORIZONTAL_OVERLAP


def _peer_lines(native: NativePage, bbox: BBox) -> list[NativeLine]:
    return [
        line
        for line in native.lines
        if line.text.strip()
        and bbox.y0 - 4.0 <= line.bbox.cy <= bbox.y1 + 4.0
        and _horizontal_gap(line.bbox, bbox) <= native.width * _LINE_HORIZONTAL_REACH_RATIO
    ]


def _drawing_union(drawings: list[NativeDrawing]) -> BBox:
    return _union_bbox([drawing.bbox for drawing in drawings])


def _union_bbox(boxes: list[BBox]) -> BBox:
    return BBox(
        min(box.x0 for box in boxes),
        min(box.y0 for box in boxes),
        max(box.x1 for box in boxes),
        max(box.y1 for box in boxes),
    )


def _boxes_overlap(left: BBox, right: BBox, margin: float = 0.0) -> bool:
    return (
        left.x0 <= right.x1 + margin
        and right.x0 <= left.x1 + margin
        and left.y0 <= right.y1 + margin
        and right.y0 <= left.y1 + margin
    )


def _horizontal_gap(left: BBox, right: BBox) -> float:
    return max(0.0, max(left.x0, right.x0) - min(left.x1, right.x1))


def _vertical_gap(left: BBox, right: BBox) -> float:
    return max(0.0, max(left.y0, right.y0) - min(left.y1, right.y1))


def _line_claimed_by_context(line: NativeLine, context_boxes: list[BBox]) -> bool:
    if line.bbox.area <= 0.0:
        return False
    for bbox in context_boxes:
        x0, y0 = max(line.bbox.x0, bbox.x0), max(line.bbox.y0, bbox.y0)
        x1, y1 = min(line.bbox.x1, bbox.x1), min(line.bbox.y1, bbox.y1)
        intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if intersection / line.bbox.area >= 0.45:
            return True
    return False


def _substantial_overlap(left: BBox, right: BBox) -> bool:
    x0, y0 = max(left.x0, right.x0), max(left.y0, right.y0)
    x1, y1 = min(left.x1, right.x1), min(left.y1, right.y1)
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return intersection / max(min(left.area, right.area), 1e-6) >= _EXCLUSIVE_OVERLAP


def _padded_bbox(bbox: BBox, page_width: float, page_height: float) -> BBox:
    return BBox(
        max(0.0, bbox.x0 - _SNAPSHOT_PADDING_X),
        max(0.0, bbox.y0 - _SNAPSHOT_PADDING_Y),
        min(page_width, bbox.x1 + _SNAPSHOT_PADDING_X),
        min(page_height, bbox.y1 + _SNAPSHOT_PADDING_Y),
    )


def _clip_at_neighbor_midlines(
    candidate: BBox,
    original: BBox,
    neighbors: list[BBox],
    *,
    vertical_context: list[BBox] | None = None,
) -> BBox:
    """Keep expansion inside the local visual region's Voronoi-like band.

    Parallel charts and package views commonly occupy the same horizontal or
    vertical band.  Their midpoint is a neutral boundary that prevents one
    snapshot from absorbing another while leaving isolated diagrams free to
    grow toward their labels.
    """
    x0, y0, x1, y1 = candidate.x0, candidate.y0, candidate.x1, candidate.y1
    for neighbor in neighbors:
        vertical_overlap = _axis_overlap_ratio(
            original.y0,
            original.y1,
            neighbor.y0,
            neighbor.y1,
        )
        if vertical_overlap >= _NEIGHBOR_BAND_OVERLAP:
            if neighbor.x1 <= original.x0:
                x0 = max(x0, (neighbor.x1 + original.x0) / 2.0)
            elif neighbor.x0 >= original.x1:
                x1 = min(x1, (original.x1 + neighbor.x0) / 2.0)

        horizontal_overlap = _axis_overlap_ratio(
            original.x0,
            original.x1,
            neighbor.x0,
            neighbor.x1,
        )
        if horizontal_overlap >= _NEIGHBOR_BAND_OVERLAP:
            if neighbor.y1 <= original.y0:
                y0 = max(y0, (neighbor.y1 + original.y0) / 2.0)
            elif neighbor.y0 >= original.y1:
                y1 = min(y1, (original.y1 + neighbor.y0) / 2.0)
    for context in vertical_context or []:
        horizontal_overlap = _axis_overlap_ratio(
            original.x0,
            original.x1,
            context.x0,
            context.x1,
        )
        if horizontal_overlap < _NEIGHBOR_BAND_OVERLAP:
            continue
        if context.y1 <= original.y0:
            y0 = max(y0, (context.y1 + original.y0) / 2.0)
        elif context.y0 >= original.y1:
            y1 = min(y1, (original.y1 + context.y0) / 2.0)
    return BBox(x0, y0, x1, y1)


def _axis_overlap_ratio(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    overlap = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    return overlap / max(min(left_end - left_start, right_end - right_start), 1e-6)
