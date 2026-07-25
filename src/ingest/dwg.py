"""DWG/DXF adapter — interface stub (intentionally not built out).

The assignment scopes 2 of 3 formats; native PDF + scanned PDF are the two implemented
here. This registers the vector/DWG seam so the adapter registry is complete and the
abstraction is proven to extend, but parsing is NOT implemented:
  - binary DWG needs external conversion (ODA File Converter) -> DXF (documented in README).
  - the DXF -> canonical path (TEXT/MTEXT/DIMENSION/LINE/INSERT via ezdxf) is designed
    (see the field notes below), not coded.
parse() raises NotImplementedError pointing at that path — an honest cut, not a hidden gap.

DXF path sketch (if built out):
  TEXT / MTEXT    -> Region(kind=text_block|tag) w/ insertion-point BBox
  DIMENSION       -> Region(kind=dimension) w/ dim_value + location
  LINE/LWPOLYLINE -> Region(kind=geometry, Geometry(points, layer))
  INSERT          -> Region(kind=geometry, Geometry(block_ref, block_name))
"""
from __future__ import annotations

from src.canonical.model import Document
from src.ingest.base import FormatAdapter, register
from src.ingest.resolver import ResolvedDoc


@register("dwg")
class DwgAdapter(FormatAdapter):
    def parse(self, resolved: ResolvedDoc) -> Document:
        raise NotImplementedError(
            "DWG/DXF ingestion is a deliberate scope cut (2 of 3 formats built: native + "
            "scanned PDF). The seam is registered; the DXF->canonical path via ezdxf is "
            "designed but not implemented. Binary DWG additionally needs ODA File Converter "
            "-> DXF first. See README 'What I deliberately cut'.")
