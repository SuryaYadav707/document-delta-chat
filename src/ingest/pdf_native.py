"""Native (born-digital) PDF adapter — PyMuPDF. No OCR: text spans + boxes read
directly (confidence 1.0). Each fitz line -> `Line`, then shared
regionize.build_regions. Vector geometry is only counted (per-entity diff is out
of scope)."""
from __future__ import annotations

import fitz  # PyMuPDF

from src.canonical.model import BBox, Document, Page, Token
from src.ingest.base import FormatAdapter, register
from src.ingest.regionize import Line, build_regions, union
from src.ingest.resolver import ResolvedDoc


@register("native_pdf")
class NativePdfAdapter(FormatAdapter):
    def parse(self, resolved: ResolvedDoc) -> Document:
        pages: list[Page] = []
        with fitz.open(stream=resolved.raw_bytes, filetype="pdf") as doc:
            for pindex in range(doc.page_count):
                pages.append(self._parse_page(resolved.pid, doc[pindex], pindex))
        return Document(
            pid=resolved.pid, doc_family=resolved.doc_family, rev_label=resolved.rev_label,
            source_format="native_pdf", page_count=len(pages),
            metadata={"path": resolved.path}, pages=pages,
        )

    def _parse_page(self, pid, page, pindex) -> Page:
        pw, ph = page.rect.width, page.rect.height
        lines = self._extract_lines(page)
        regions = build_regions(pid, pindex, lines, pw, ph, provenance="native")
        return Page(
            page_index=pindex, width=pw, height=ph, rotation=page.rotation,
            regions=regions, raw_text=page.get_text(),
            metadata={"vector_path_count": len(page.get_drawings())},
        )

    @staticmethod
    def _extract_lines(page) -> list[Line]:
        out = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = " ".join(s["text"].strip() for s in spans)
                tokens = [Token(s["text"].strip(), BBox(*s["bbox"]), 1.0) for s in spans]
                out.append(Line(text, union([s["bbox"] for s in spans]), tokens))
        return out
