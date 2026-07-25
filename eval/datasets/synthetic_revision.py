"""Synthetic-revision eval case — deterministic, controllable ground truth.

Clones the Export P&ID canonical doc and applies a FIXED edit spec. The same
function emits both the revised doc AND the gold change list, so labels can never
drift from the data. This is the primary regression case (exact GT, no OCR/LLM).
"""
from __future__ import annotations

import copy

from src.canonical.model import BBox, Geometry, Region
from src.delta.engine import region_key
from src.ingest.pdf_native import NativePdfAdapter
from src.ingest.resolver import PIDResolver

EXPORT = "data/samples/export-gas-compressor.pdf"


def _geo(rid, name, cx):
    return Region(rid, "geometry", BBox(0, 0, 1, 1, cx, 0.5, cx + 0.01, 0.51),
                  geometry=Geometry("block_ref", block_name=name))


def build_case():
    """Return (base_doc, rev_doc, gold[list of {change_type,kind,key}])."""
    src = NativePdfAdapter().parse(PIDResolver().resolve(EXPORT))
    base = copy.deepcopy(src)
    rev = copy.deepcopy(src)
    pb, pr = base.pages[0], rev.pages[0]
    gold: list[dict] = []

    # 1) modify a dimension setpoint value
    dim = sorted([r for r in pr.regions if r.kind == "dimension" and r.attrs.get("setpoints")],
                 key=lambda r: r.region_id)[0]
    gold.append({"change_type": "modified", "kind": "dimension", "key": region_key(dim)})
    dim.attrs["setpoints"][0]["value"] += 99

    # 2) modify a title-block field value
    duty = [r for r in pr.regions if r.attrs.get("field_label") == "DUTY"][0]
    gold.append({"change_type": "modified", "kind": "title_block", "key": region_key(duty)})
    duty.attrs["value"] = "9999 NOTE 29"

    # 3) remove a callout
    callout = sorted([r for r in pr.regions if r.kind == "callout"], key=lambda r: r.region_id)[0]
    gold.append({"change_type": "removed", "kind": "callout", "key": region_key(callout)})
    pr.regions = [r for r in pr.regions if r is not callout]

    # 4) remove a numbered note (8)
    note8 = [r for r in pr.regions if r.kind == "note_item" and r.attrs.get("note_number") == 8]
    if note8:
        gold.append({"change_type": "removed", "kind": "note_item", "key": "note:8"})
        pr.regions = [r for r in pr.regions if r is not note8[0]]

    # 5) add a new note
    pr.regions.append(Region("rev:p0:rNEW", "note_item", BBox(0, 0, 1, 1, .5, .9, .6, .92),
                             text="99. NEW SAFETY NOTE", attrs={"note_number": 99, "text": "99. NEW SAFETY NOTE"}))
    gold.append({"change_type": "added", "kind": "note_item", "key": "note:99"})

    # 5b) RENUMBER a note (keep its text) -> must NOT show as add/remove: the note
    #     is matched by content, so it produces no change and no gold entry. A
    #     regression (number-only anchoring) would emit removed+added = 2 false positives.
    renumber = [r for r in pr.regions if r.kind == "note_item" and r.attrs.get("note_number") == 13]
    if renumber:
        renumber[0].attrs["note_number"] = 97

    # 6) rename a tag (remove old id, add new)
    for r in pr.regions:
        ids = r.attrs.get("tag_ids")
        if ids and "26-CX-9021" in ids:
            r.attrs["tag_ids"] = ["26-CX-9099" if t == "26-CX-9021" else t for t in ids]
    gold.append({"change_type": "removed", "kind": "tag", "key": "tag:26-CX-9021"})
    gold.append({"change_type": "added", "kind": "tag", "key": "tag:26-CX-9099"})

    # 7) geometry: VALVE layout shift (low-conf), FILTER removed, PUMP added
    pb.regions.append(_geo("b:p0:v", "VALVE", 0.20)); pr.regions.append(_geo("r:p0:v", "VALVE", 0.40))
    gold.append({"change_type": "moved", "kind": "geometry", "key": "geom:VALVE"})
    pb.regions.append(_geo("b:p0:f", "FILTER", 0.30))
    gold.append({"change_type": "removed", "kind": "geometry", "key": "geom:FILTER"})
    pr.regions.append(_geo("r:p0:p", "PUMP", 0.60))
    gold.append({"change_type": "added", "kind": "geometry", "key": "geom:PUMP"})

    return base, rev, gold
