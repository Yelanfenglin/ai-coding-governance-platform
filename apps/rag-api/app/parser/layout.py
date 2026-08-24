"""Load PP-DocLayoutV3 proposals without adding semantic guesses."""

from __future__ import annotations

import json
from pathlib import Path

from .models import BBox, LayoutProposal, normalize_hint


def load_all_proposals(json_path: str | Path) -> dict[int, list[LayoutProposal]]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    pages: dict[int, list[LayoutProposal]] = {}
    for key, page in data.items():
        proposals: list[LayoutProposal] = []
        for region in page.get("regions", []):
            coordinates = region.get("bbox_pdf")
            if not coordinates or len(coordinates) != 4:
                continue
            label = region.get("label", "")
            proposals.append(
                LayoutProposal(
                    bbox=BBox.from_pdf([float(value) for value in coordinates]),
                    label=label,
                    hint=normalize_hint(label),
                    score=float(region.get("score", 0.0)),
                    order=region.get("order"),
                )
            )
        pages[int(key)] = proposals
    return pages
