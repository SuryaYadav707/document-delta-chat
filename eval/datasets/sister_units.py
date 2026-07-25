"""Sister-units eval case — real-world, hand-labeled (title-block scope).

Lift (26-KA-901) vs Export (26-KA-902) are sister compressors, not a literal
revision, but their title blocks align 1:1 and differ meaningfully. We label the
title-block field changes exhaustively (independent ground truth) and score the
delta restricted to kind=title_block. Realistic; measures real matching + value
extraction (a missed field = honest recall loss).
"""
from __future__ import annotations

from src.ingest.pdf_native import NativePdfAdapter
from src.ingest.resolver import PIDResolver

LIFT = "data/samples/lift-gas-compressor.pdf"
EXPORT = "data/samples/export-gas-compressor.pdf"

SCOPE = {"title_block"}

# title-block fields that genuinely differ Lift -> Export (independent labels)
GOLD = [
    {"change_type": "modified", "kind": "title_block", "key": "field:TAG NUMBER"},
    {"change_type": "modified", "kind": "title_block", "key": "field:SERVICE"},
    {"change_type": "modified", "kind": "title_block", "key": "field:DUTY"},
    {"change_type": "modified", "kind": "title_block", "key": "field:FLOW RATE"},
    {"change_type": "modified", "kind": "title_block", "key": "field:DISCHARGE / SUCTION OP. PRESS. (MAX)"},
    {"change_type": "modified", "kind": "title_block", "key": "field:DISCHARGE / SUCTION OP. TEMP."},
    {"change_type": "modified", "kind": "title_block", "key": "field:VESSEL TRIM"},
]


def build_case():
    res = PIDResolver()
    base = NativePdfAdapter().parse(res.resolve(LIFT))
    rev = NativePdfAdapter().parse(res.resolve(EXPORT))
    return base, rev, GOLD, SCOPE
