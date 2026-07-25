"""PIDResolver — resolves an opaque PID (a local file path here; swappable for an
object store later) to bytes + metadata, and detects the source format from
content (text-layer coverage) so the registry picks the right adapter."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from src.canonical.model import SourceFormat

# below this many extracted chars per page, a PDF page is treated as scanned
_NATIVE_CHARS_PER_PAGE = 100
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_DWG_EXTS = {".dwg", ".dxf"}


@dataclass
class ResolvedDoc:
    pid: str
    raw_bytes: bytes
    source_format: SourceFormat
    page_count: int
    rev_label: str
    doc_family: str
    path: str
    # per-page routing: indexes of pages with no reliable text layer (need OCR)
    scanned_pages: list[int]


class PIDResolver:
    """resolve(pid) -> ResolvedDoc. PID == local path for the take-home."""

    def resolve(self, pid: str) -> ResolvedDoc:
        path = Path(pid)
        if not path.exists():
            raise FileNotFoundError(f"cannot resolve PID to bytes: {pid}")
        raw = path.read_bytes()
        ext = path.suffix.lower()

        if ext in _DWG_EXTS:
            return ResolvedDoc(pid, raw, "dwg", 1, path.stem, path.stem, str(path), [])
        if ext in _IMAGE_EXTS:
            # a standalone image is a single scanned page
            return ResolvedDoc(pid, raw, "scanned_pdf", 1, path.stem, path.stem, str(path), [0])

        # PDF: decide native vs scanned by per-page text coverage
        page_count, scanned_pages = self._pdf_text_coverage(raw)
        fmt: SourceFormat = "scanned_pdf" if len(scanned_pages) == page_count else "native_pdf"
        family, rev = self._family_and_rev(path.stem)
        return ResolvedDoc(pid, raw, fmt, page_count, rev, family, str(path), scanned_pages)

    @staticmethod
    def _pdf_text_coverage(raw: bytes) -> tuple[int, list[int]]:
        """Return (page_count, [indexes of pages lacking a usable text layer])."""
        scanned: list[int] = []
        with fitz.open(stream=raw, filetype="pdf") as doc:
            n = doc.page_count
            for i in range(n):
                if len(doc[i].get_text().strip()) < _NATIVE_CHARS_PER_PAGE:
                    scanned.append(i)
        return n, scanned

    @staticmethod
    def _family_and_rev(stem: str) -> tuple[str, str]:
        """Best-effort: strip a trailing rev token (rev-b, _v2, (1)) for doc_family."""
        m = re.search(r"[-_ ]?(rev[-_ ]?[a-z0-9]+|v\d+|\(\d+\))$", stem, re.I)
        if m:
            return stem[: m.start()].strip(" -_"), m.group(1)
        return stem, stem

    @staticmethod
    def detect_format(raw_bytes: bytes, hint: str | None = None) -> SourceFormat:
        """Standalone format sniff (used by tests)."""
        if hint:
            ext = Path(hint).suffix.lower()
            if ext in _DWG_EXTS:
                return "dwg"
            if ext in _IMAGE_EXTS:
                return "scanned_pdf"
        _, scanned = PIDResolver._pdf_text_coverage(raw_bytes)
        with fitz.open(stream=raw_bytes, filetype="pdf") as doc:
            n = doc.page_count
        return "scanned_pdf" if len(scanned) == n else "native_pdf"
