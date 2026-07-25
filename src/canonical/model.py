"""Canonical representation — the format-agnostic intermediate model.

Every FormatAdapter (native PDF, scanned PDF, DWG) normalizes its input into a
`Document`. The delta engine and chat layer operate ONLY on this model and never
learn which source format produced it. Everything anchors to (page_index, bbox)
so citations, markup, and delta locations work identically across formats.

A 4th format plugs in by writing one adapter that emits `Document` — nothing
downstream changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

RegionKind = Literal[
    "title_block", "notes_list", "note_item", "tag", "table", "table_cell",
    "dimension", "geometry", "callout", "legend", "text_block", "unknown",
]
SourceFormat = Literal["native_pdf", "scanned_pdf", "dwg"]
Provenance = Literal["native", "ocr", "ocr+vision", "dxf"]


@dataclass
class BBox:
    """Axis-aligned box, origin top-left. Absolute (points) + normalized 0..1."""
    x0: float; y0: float; x1: float; y1: float
    x0n: float = 0.0; y0n: float = 0.0; x1n: float = 0.0; y1n: float = 0.0


@dataclass
class Token:
    """Word-level unit. confidence == 1.0 for born-digital text."""
    text: str
    bbox: BBox
    confidence: float = 1.0


@dataclass
class Geometry:
    """Vector info — populated by native PDF drawings + DXF; empty for scans."""
    entity_type: Literal["line", "polyline", "circle", "arc", "block_ref", "dim"]
    points: Optional[list[tuple[float, float]]] = None
    layer: Optional[str] = None
    block_name: Optional[str] = None
    dim_value: Optional[float] = None


@dataclass
class Region:
    """A semantic cluster — the unit of both diffing and chunking."""
    region_id: str
    kind: RegionKind
    bbox: BBox
    text: str = ""
    tokens: list[Token] = field(default_factory=list)
    confidence: float = 1.0
    provenance: Provenance = "native"
    attrs: dict = field(default_factory=dict)   # note_number, tag_id, setpoints, value/unit, row/col, field_label
    geometry: Optional[Geometry] = None
    neighbors: list[str] = field(default_factory=list)   # spatially-adjacent region_ids


@dataclass
class Page:
    page_index: int
    width: float
    height: float
    rotation: int = 0
    regions: list[Region] = field(default_factory=list)
    raw_text: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class Document:
    """One PID / one revision, normalized."""
    pid: str
    doc_family: str
    rev_label: str
    source_format: SourceFormat
    page_count: int
    metadata: dict = field(default_factory=dict)
    pages: list[Page] = field(default_factory=list)
