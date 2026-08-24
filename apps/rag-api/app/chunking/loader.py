"""Load a parsed document from storage/parsed/{doc} into a ParsedDocument."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ParsedDocument:
    doc_id: str
    product: str
    version: str
    pdf_name: str
    pages: list[dict]  # pages.json entries

    @property
    def total_pages(self) -> int:
        return len(self.pages)


_PRODUCT_RE = re.compile(r"^(?P<product>[A-Za-z0-9]{2,})\s*[_-]?\s*SPEC")
_VERSION_RE = re.compile(r"V?v?(?P<version>\d+(?:\.\d+)+)(?P<suffix>[A-Za-z]*)\s*$", re.I)


def _doc_meta_from_pdf_name(pdf_name: str) -> tuple[str, str, str]:
    base = Path(pdf_name).stem
    product = base
    version = ""
    m = _PRODUCT_RE.search(base)
    if m:
        product = m.group("product")
    v = _VERSION_RE.search(base.upper())
    if v:
        version = v.group("version")
    doc_id = f"{product}_spec_v{version}".lower() if version else f"{product}_spec".lower()
    return product, version, doc_id


def load_parsed_document(doc_dir: Path) -> ParsedDocument:
    """Load pages.json + summary.json from a storage/parsed/<product>/ directory."""
    doc_dir = Path(doc_dir)
    pages = __import__("json").loads((doc_dir / "pages.json").read_text(encoding="utf-8"))
    summary = __import__("json").loads((doc_dir / "summary.json").read_text(encoding="utf-8"))
    pdf_name = summary.get("pdf", "")
    product, version, doc_id = _doc_meta_from_pdf_name(pdf_name)
    return ParsedDocument(
        doc_id=doc_id,
        product=product,
        version=version,
        pdf_name=pdf_name,
        pages=pages,
    )
