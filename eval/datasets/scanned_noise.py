"""Scanned-noise eval case — quantifies OCR-induced false positives.

Compares the native Export doc against a SCANNED twin of the same page. The ideal
delta is EMPTY, so every reported change is an OCR false positive. Measures how
much noise OCR injects into the delta.

OCR is slow on CPU, so the scanned canonical is generated once and cached. Run:
    python -m eval.datasets.scanned_noise      # prepare the cache (OCR)
then `make eval` picks it up automatically. Without the cache the case is skipped.
"""
from __future__ import annotations

import pickle
from pathlib import Path

from src.ingest.pdf_native import NativePdfAdapter
from src.ingest.resolver import PIDResolver

EXPORT = "data/samples/export-gas-compressor.pdf"
SCANNED = "data/samples/export-scanned.pdf"
CACHE = Path("eval/datasets/scanned-noise/export_scanned_canonical.pkl")


def build_case():
    """Return (base_native, rev_scanned) or None if the OCR cache isn't prepared."""
    if not CACHE.exists():
        return None
    base = NativePdfAdapter().parse(PIDResolver().resolve(EXPORT))
    rev = pickle.loads(CACHE.read_bytes())
    return base, rev


def prepare() -> None:
    """OCR the scanned twin once and cache its canonical doc."""
    from src.ingest.pdf_scanned import ScannedPdfAdapter
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    ad = ScannedPdfAdapter()
    ad.cfg.dpi = 200
    ad.cfg.vision_verify = False  # deterministic (no LLM) for a stable eval baseline
    doc = ad.parse(PIDResolver().resolve(SCANNED))
    CACHE.write_bytes(pickle.dumps(doc))
    print(f"cached scanned canonical -> {CACHE}")


if __name__ == "__main__":
    prepare()
