"""Small shared models for the deterministic PDF parser.

Only geometry and layout-proposal facts live here.  Parser versions must not
depend on an older implementation merely to reuse these value objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Hint = Literal["TEXT", "TABLE", "VISUAL", "NOISE", "UNKNOWN"]

_TEXT_LABELS = {
    "text",
    "content",
    "paragraph_title",
    "algorithm",
    "display_formula",
    "figure_title",
    "vision_footnote",
}
_TABLE_LABELS = {"table"}
_VISUAL_LABELS = {"image", "chart"}
_NOISE_LABELS = {"header", "header_image", "footer", "number"}


def normalize_hint(label: str | None) -> Hint:
    value = (label or "").strip().lower()
    if value in _TEXT_LABELS:
        return "TEXT"
    if value in _TABLE_LABELS:
        return "TABLE"
    if value in _VISUAL_LABELS:
        return "VISUAL"
    if value in _NOISE_LABELS:
        return "NOISE"
    return "UNKNOWN"


@dataclass(frozen=True)
class BBox:
    """PDF point coordinates with a top-left origin."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def h(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def area(self) -> float:
        return self.w * self.h

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    @classmethod
    def from_pdf(cls, coordinates: list[float]) -> BBox:
        x0, y0, x1, y1 = coordinates
        return cls(x0, y0, x1, y1)


@dataclass(frozen=True)
class LayoutProposal:
    bbox: BBox
    label: str
    hint: Hint = "UNKNOWN"
    score: float = 0.0
    order: int | None = None


@dataclass(frozen=True)
class VectorRule:
    bbox: BBox
    horizontal: bool
    thickness: float = 0.5


@dataclass(frozen=True)
class NativeImage:
    bbox: BBox
    xref: int = 0
    embedded: bool = False


@dataclass(frozen=True)
class NativeDrawing:
    """One native PDF vector path and its visible bounds."""

    bbox: BBox
    item_count: int = 0
    filled: bool = False
