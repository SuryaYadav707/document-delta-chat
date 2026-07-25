"""Shared region assembly for both adapters: a page's `Line`s -> typed `Region`s
(notes, title-block field/value rows, and single-line kinds via `typing_rules`).
Native (fitz spans) and scanned (OCR clusters) feed the same assembler, so the
delta engine never sees format-specific shapes."""
from __future__ import annotations

import re

from src.canonical.model import BBox, Provenance, Region, Token

_NUM_RE = re.compile(r"^\s*(\d{1,2})\.\s*(.*)$")


class Line:
    __slots__ = ("text", "bbox", "tokens")

    def __init__(self, text: str, bbox: tuple, tokens: list[Token]):
        self.text = text
        self.bbox = bbox            # (x0, y0, x1, y1)
        self.tokens = tokens


def union(boxes) -> tuple:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def rows_overlap(a, b) -> bool:
    h = max(min(a[3] - a[1], b[3] - b[1]), 1.0)
    return (min(a[3], b[3]) - max(a[1], b[1])) > 0.5 * h


def mk_bbox(raw, pw, ph) -> BBox:
    x0, y0, x1, y1 = raw
    return BBox(x0, y0, x1, y1, x0 / pw, y0 / ph, x1 / pw, y1 / ph)


def build_regions(pid: str, pindex: int, lines: list[Line], pw: float, ph: float,
                  provenance: Provenance = "native") -> list[Region]:
    from src.ingest.typing_rules import classify_line

    used = [False] * len(lines)
    reading_order = sorted(range(len(lines)), key=lambda i: (lines[i].bbox[1], lines[i].bbox[0]))
    regions: list[Region] = []
    next_id = 0

    # thresholds are PAGE-WIDTH-RELATIVE so the same logic works in points (native,
    # pw~1191) and in pixels (scanned/OCR, pw~3309) without retuning.
    gap_max = 0.035 * pw            # note number -> body head max x-gap
    col_tol = 0.035 * pw            # same-column tolerance for the next-note stop
    cont_x_tol = max(0.005 * pw, 2.0)   # continuation left-edge alignment tolerance
    tb_label_gap = 0.075 * pw       # title-block label width before the value column
    tb_value_max = 0.23 * pw        # right edge of the title-block value column

    def emit(kind, boxes, text_parts, tokens, attrs):
        nonlocal next_id
        confidence = round(min((t.confidence for t in tokens), default=1.0), 3)
        regions.append(Region(
            region_id=f"{pid}:p{pindex}:r{next_id}", kind=kind,
            bbox=mk_bbox(union(boxes), pw, ph), text=" ".join(text_parts).strip(),
            tokens=tokens, confidence=confidence, provenance=provenance, attrs=attrs,
        ))
        next_id += 1

    note_anchors = [(i, lines[i].bbox[0], lines[i].bbox[1])
                    for i in reading_order if _NUM_RE.match(lines[i].text.strip())]

    def next_note_y(x, y):
        below = [ay for (_, ax, ay) in note_anchors if abs(ax - x) <= col_tol and ay > y + 1]
        return min(below) if below else float("inf")

    # pass 1: numbered notes = "N." + body-to-the-right + wrapped continuation lines
    for i in reading_order:
        if used[i]:
            continue
        match = _NUM_RE.match(lines[i].text.strip())
        if not match:
            continue
        used[i] = True
        note_num, inline_body = int(match.group(1)), match.group(2).strip()
        anchor_box = lines[i].bbox
        text_parts, boxes, tokens = [lines[i].text.strip()], [anchor_box], list(lines[i].tokens)
        body_x = anchor_box[0]
        if not inline_body:  # body sits on the same row, a small gap to the right
            same_row = [j for j in reading_order if not used[j] and rows_overlap(anchor_box, lines[j].bbox)
                        and 0 <= lines[j].bbox[0] - anchor_box[2] <= gap_max]
            if same_row:
                j = min(same_row, key=lambda j: lines[j].bbox[0])
                used[j] = True; body_x = lines[j].bbox[0]
                text_parts.append(lines[j].text); boxes.append(lines[j].bbox); tokens += lines[j].tokens
        stop_y = next_note_y(anchor_box[0], anchor_box[1])
        while True:  # pull wrapped continuation lines until the next numbered note
            cur_box = union(boxes)
            line_height = max(cur_box[3] - cur_box[1], 4.0)
            below = [j for j in reading_order if not used[j]
                     and not _NUM_RE.match(lines[j].text.strip())
                     and abs(lines[j].bbox[0] - body_x) <= cont_x_tol
                     and 0 <= lines[j].bbox[1] - cur_box[3] <= 1.6 * line_height
                     and lines[j].bbox[1] < stop_y - 1]
            if not below:
                break
            j = min(below, key=lambda j: lines[j].bbox[1])
            used[j] = True; text_parts.append(lines[j].text); boxes.append(lines[j].bbox); tokens += lines[j].tokens
        emit("note_item", boxes, text_parts, tokens, {"note_number": note_num, "text": " ".join(text_parts).strip()})

    # pass 2a: title-block labels first, pairing each with its value cell(s) to the right
    for i in reading_order:
        if used[i]:
            continue
        kind, attrs = classify_line(lines[i].text)
        if kind != "title_block":
            continue
        used[i] = True
        label_box = lines[i].bbox
        boxes, text_parts, tokens = [label_box], [lines[i].text], list(lines[i].tokens)
        value_ids = sorted(
            [j for j in reading_order if not used[j] and rows_overlap(label_box, lines[j].bbox)
             and label_box[0] + tb_label_gap <= lines[j].bbox[0] <= tb_value_max],
            key=lambda j: lines[j].bbox[0],
        )
        if value_ids:
            attrs["value"] = " ".join(lines[j].text.strip() for j in value_ids)
            for j in value_ids:
                used[j] = True; boxes.append(lines[j].bbox); tokens += lines[j].tokens
        emit(kind, boxes, text_parts, tokens, attrs)

    # pass 2b: everything else, one region per line
    for i in reading_order:
        if used[i]:
            continue
        used[i] = True
        kind, attrs = classify_line(lines[i].text)
        emit(kind, [lines[i].bbox], [lines[i].text], list(lines[i].tokens), attrs)

    _attach_neighbors(regions)
    _attach_instrument(regions)
    return regions


_INST_NUM = re.compile(r"9\d{3}")   # P&ID instrument numbers (9xxx)


def _attach_instrument(regions) -> None:
    """Give each setpoint (dimension) its instrument number, read from the nearest
    neighbor that carries one (e.g. 'HH:245' next to 'PI'/'9062' -> instrument 9062).
    Lets the delta anchor setpoints by (instrument, limit) instead of mispairing them."""
    by_id = {r.region_id: r for r in regions}
    for region in regions:
        if region.kind != "dimension":
            continue
        for text in [by_id[n].text for n in region.neighbors if n in by_id]:
            match = _INST_NUM.search(text)
            if match:
                region.attrs["instrument"] = match.group(0)
                break


def _attach_neighbors(regions, k: int = 6, max_dist: float = 0.08) -> None:
    """Record each region's k nearest regions (by normalized centroid), so the index
    can build a 'parent-window' of context for thin chunks (a lone tag/setpoint)."""
    def centroid(r):
        b = r.bbox
        return (b.x0n + b.x1n) / 2, (b.y0n + b.y1n) / 2

    centroids = [centroid(r) for r in regions]
    for i, region in enumerate(regions):
        cx, cy = centroids[i]
        nearest = sorted(((cx - ox) ** 2 + (cy - oy) ** 2, j)
                         for j, (ox, oy) in enumerate(centroids) if j != i)
        region.neighbors = [regions[j].region_id for dist2, j in nearest[:k] if dist2 <= max_dist ** 2]
