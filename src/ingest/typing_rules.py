"""Shared region-typing rules — used by BOTH native and scanned adapters.

Keeping the semantic classification in one place means the scanned OCR path and
the native path produce the same Region.kind + attrs for the same content, so the
delta engine never sees format-specific shapes.
"""
from __future__ import annotations

import re

# instrument / equipment tag, e.g. 26-PDI-9054, 26-KA-901, 26-CX-9122
TAG_RE = re.compile(r"\b\d{2}-[A-Z]{2,4}-\d{3,5}\b")
# pipe/line spec, e.g. 8"-PV-26-9035-FC11S-08
LINE_SPEC_RE = re.compile(r'\d+(?:\s*\d+/\d+)?"?-[A-Z]{2}-\d{2}-\d{3,4}-[A-Z0-9]+')
# setpoints, e.g. HH : 160, LL: 120, H:0.3
SETPOINT_RE = re.compile(r"\b(HH|LL|H|L)\s*:\s*([0-9]+(?:\.[0-9]+)?)")
# set pressure, e.g. SP = 257 bar (g)
SP_RE = re.compile(r"SP\s*=?\s*([0-9]+(?:\.[0-9]+)?)\s*bar", re.I)
# note item start, e.g. "1. ", "16.  "
NOTE_START_RE = re.compile(r"^\s*(\d{1,2})\.\s")
# connection callout
CALLOUT_RE = re.compile(
    r"\b(TO|FROM)\b.*\b(FLARE|COMPRESSOR|DRAIN|SCRUBBER|COOLER|HEADER|VENT|YARD)\b", re.I
)

# title-block field labels (order matters: longest/most-specific first)
FIELD_LABELS = [
    "TAG NUMBER", "SERVICE", "DUTY", "FLOW RATE",
    "DISCHARGE / SUCTION OP. PRESS. (MAX)", "DISCHARGE / SUCTION OP. TEMP.",
    "DISCHARGE / SUCTION DESIGN PRESS. (MAX)", "DISCHARGE / SUCTION DESIGN TEMP.",
    "MATERIAL", "QUANTITY", "TYPE", "VESSEL TRIM", "VENDOR",
]


def parse_setpoints(text: str) -> list[dict]:
    """Extract numeric setpoints so delta can diff them as values, not strings."""
    out = [{"limit": m.group(1).upper(), "value": float(m.group(2))} for m in SETPOINT_RE.finditer(text)]
    out += [{"limit": "SP", "value": float(m.group(1)), "unit": "barg"} for m in SP_RE.finditer(text)]
    return out


def classify_line(text: str) -> tuple[str, dict]:
    """Map a single line of text to (Region.kind, attrs). Note merging is done by
    the adapter (multi-line); this handles single-line kinds."""
    t = text.strip()
    if not t:
        return "text_block", {}

    for label in FIELD_LABELS:
        if t.upper().startswith(label):
            return "title_block", {"field_label": label, "value": t[len(label):].strip()}

    setpoints = parse_setpoints(t)
    if setpoints:
        return "dimension", {"setpoints": setpoints, "raw": t}

    tags = TAG_RE.findall(t)
    line_spec = LINE_SPEC_RE.search(t)
    if line_spec:
        return "callout", {"line_spec": line_spec.group(0), "tag_ids": tags}
    if tags:
        return "tag", {"tag_ids": tags}
    if CALLOUT_RE.search(t):
        return "callout", {}
    return "text_block", {}
