"""PP-DocLayoutV3 批量版面分块，支持任意 PDF、断点续跑与原子 checkpoint。"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import fitz
import numpy as np
from paddleocr import LayoutDetection

ROOT = Path(__file__).resolve().parents[1]


def run(
    pdf_path: Path,
    output_path: Path,
    *,
    summary_path: Path | None = None,
    model_name: str = "PP-DocLayoutV3",
    device: str = "cpu",
    dpi: int = 160,
    pages: list[int] | None = None,
    resume: bool = False,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = _load_existing(output_path) if resume else {}

    with fitz.open(str(pdf_path)) as document:
        page_range = pages if pages is not None else list(range(1, len(document) + 1))
        invalid = [page for page in page_range if page < 1 or page > len(document)]
        if invalid:
            raise ValueError(f"page out of range: {invalid}; PDF has {len(document)} pages")
        pending = [page for page in page_range if str(page) not in result]
        if pending:
            print(f"[init] loading {model_name} on {device}...", flush=True)
            started = time.time()
            model = LayoutDetection(model_name=model_name, device=device)
            print(f"[init] model ready {time.time() - started:.1f}s", flush=True)
        else:
            model = None

        batch_started = time.time()
        for position, page_number in enumerate(pending, 1):
            page = document[page_number - 1]
            page_width, page_height = page.rect.width, page.rect.height
            pixmap = page.get_pixmap(dpi=dpi, alpha=False)
            image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, pixmap.n
            )
            if pixmap.n == 4:
                image = image[:, :, :3]
            page_started = time.time()
            predictions = list(model.predict(image)) if model is not None else []
            elapsed = time.time() - page_started
            raw = predictions[0] if predictions else {}
            data = _result_json(raw)
            scale_x, scale_y = page_width / pixmap.width, page_height / pixmap.height
            entries = []
            for box in data.get("boxes", []):
                if not isinstance(box, dict):
                    continue
                coordinates = box.get("coordinate")
                bbox_pdf = bbox_norm = None
                if coordinates and len(coordinates) == 4:
                    bbox_pdf = [
                        round(value, 1)
                        for value in (
                            coordinates[0] * scale_x,
                            coordinates[1] * scale_y,
                            coordinates[2] * scale_x,
                            coordinates[3] * scale_y,
                        )
                    ]
                    bbox_norm = [
                        round(coordinates[0] / pixmap.width, 4),
                        round(coordinates[1] / pixmap.height, 4),
                        round(coordinates[2] / pixmap.width, 4),
                        round(coordinates[3] / pixmap.height, 4),
                    ]
                entries.append(
                    {
                        "label": box.get("label"),
                        "score": round(float(box.get("score", 0.0)), 4),
                        "order": box.get("order"),
                        "bbox_pdf": bbox_pdf,
                        "bbox_norm": bbox_norm,
                    }
                )
            result[str(page_number)] = {
                "time_s": round(elapsed, 2),
                "n": len(entries),
                "regions": entries,
            }
            _atomic_json(output_path, _sorted_pages(result))
            print(
                f"[page {page_number}/{len(document)}] {len(entries)} regions, {elapsed:.2f}s "
                f"({position}/{len(pending)})",
                flush=True,
            )

    result = _sorted_pages(result)
    if summary_path is not None:
        _write_summary(summary_path, pdf_path.name, result)
    print(
        f"[done] {len(pending)} pages in {time.time() - batch_started:.1f}s -> {output_path}",
        flush=True,
    )
    return result


def _result_json(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    value = getattr(raw, "json", None)
    value = value() if callable(value) else value
    if isinstance(value, str):
        value = json.loads(value)
    return value if isinstance(value, dict) else {}


def _load_existing(path: Path) -> dict:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _sorted_pages(data: dict) -> dict:
    return {key: data[key] for key in sorted(data, key=lambda value: int(value))}


def _atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _write_summary(path: Path, pdf_name: str, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# PP-DocLayoutV3 版面分块摘要", "", f"文档: {pdf_name}  已处理 {len(result)} 页", ""]
    for key, page in result.items():
        counts = Counter(region.get("label") for region in page.get("regions", []))
        labels = ", ".join(f"{label}={count}" for label, count in counts.items())
        lines.append(f"p{int(key):>3} ({page.get('n', 0):>3} 区域): {labels}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PP-DocLayoutV3 for a PDF")
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "paddle_doclayout_v3")
    parser.add_argument("--name", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--model-name", default="PP-DocLayoutV3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--pages", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    name = args.name or args.pdf.stem
    output = args.output or args.out_dir / f"{name}_doclayout_v3_regions.json"
    summary = args.summary or args.out_dir / f"{name}_doclayout_v3_summary.md"
    pages = [int(value) for value in args.pages.split(",") if value.strip()] if args.pages else None
    run(
        args.pdf,
        output,
        summary_path=summary,
        model_name=args.model_name,
        device=args.device,
        dpi=args.dpi,
        pages=pages,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
