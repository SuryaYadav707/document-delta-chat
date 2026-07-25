"""Delta engine — classify add / remove / modify / MOVED + typed diff + confidence.

Structural output is fully deterministic and reproducible (`use_llm=False`
produces the complete delta). The LLM, when enabled, only ANNOTATES ambiguous
modifies — it never decides matching or the change set.

Change types:
  - added / removed : unmatched regions (+ tag / geometry set-diff)
  - modified        : matched content changed (text or numeric value)
  - moved           : matched element changed SHEET/PAGE (engineering-meaningful)

"Moved" semantics (domain-scoped — see PLAN.md):
  On a P&ID, an element's intra-sheet XY position is NOT engineering-meaningful
  (drafters reflow sheets between revisions). So:
    * text kinds (note/title/setpoint/callout): XY moves are IGNORED — only
      content changes are reported.
    * a change of SHEET/PAGE is meaningful -> reported as "moved".
    * GEOMETRY symbols: an intra-sheet centroid shift is emitted as a LOW-
      confidence "layout" signal (evidence.reason="layout_shift"), because we
      cannot tell cosmetic re-draw from a real relocation without connectivity.
  True topological move ("valve moved upstream of the pump") needs a connectivity
  graph, which is an explicit scope cut.

Typed diffs: title_block/dimension -> numeric old->new->delta; note_item/callout
-> text change; tag -> id set-diff; geometry -> symbol add/remove/move by block.
confidence = min extraction confidence of the involved regions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from src.canonical.model import BBox, Document, Region
from src.delta.align import align, text_sim

ChangeType = Literal["added", "removed", "modified", "moved"]
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_NOTE_TAIL = re.compile(r"\s*NOTE\s*\d+\s*$", re.I)


@dataclass
class ValueChange:
    old: Optional[str]
    new: Optional[str]
    numeric_delta: Optional[float] = None
    unit: Optional[str] = None


@dataclass
class DeltaItem:
    id: str
    change_type: ChangeType
    kind: str
    page: int
    bbox: BBox
    description: str
    confidence: float
    a_ref: Optional[str] = None
    b_ref: Optional[str] = None
    value_change: Optional[ValueChange] = None
    evidence: dict = field(default_factory=dict)


@dataclass
class Delta:
    comparison_id: str
    summary: dict
    items: list[DeltaItem] = field(default_factory=list)


def _num(s: str) -> Optional[float]:
    m = _NUMBER.search(s or "")
    return float(m.group()) if m else None


def _clean_value(s: str) -> str:
    return _NOTE_TAIL.sub("", " ".join((s or "").split())).strip()


def region_key(r: Region) -> str:
    """Stable identity of a region's target — used to match predicted vs gold in eval."""
    a = r.attrs
    if r.kind == "note_item":
        return f"note:{a.get('note_number')}"
    if r.kind == "title_block":
        return f"field:{a.get('field_label')}"
    if r.kind == "dimension":
        limits = ",".join(sorted(s["limit"] for s in a.get("setpoints", [])))
        return f"dim:{a.get('instrument', '')}:{limits}"
    if r.kind == "geometry" and r.geometry:
        return f"geom:{r.geometry.block_name or r.geometry.entity_type}"
    if r.kind == "callout":
        return f"callout:{(r.text or '')[:24]}"
    return f"{r.kind}:{(r.text or '')[:16]}"


def _page_of(region: Region) -> int:
    try:
        return int(region.region_id.rsplit(":", 2)[1].lstrip("p"))
    except Exception:
        return 0


def _centroid(b: BBox) -> tuple[float, float]:
    return (b.x0n + b.x1n) / 2, (b.y0n + b.y1n) / 2


def _shift(a: Region, b: Region) -> float:
    ax, ay = _centroid(a.bbox); bx, by = _centroid(b.bbox)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def compute_delta(doc_a: Document, doc_b: Document, cfg, comparison_id: str = "cmp",
                  use_llm: bool = True) -> Delta:
    al = align(doc_a, doc_b, cfg)
    items: list[DeltaItem] = []
    n = 0

    def add(ct, kind, region, desc, conf, a_ref=None, b_ref=None, vc=None, ev=None):
        nonlocal n
        ev = dict(ev or {})
        ev.setdefault("key", region_key(region))   # stable target id for eval matching
        items.append(DeltaItem(
            id=f"{comparison_id}:d{n}", change_type=ct, kind=kind,
            page=_page_of(region), bbox=region.bbox, description=desc,
            confidence=round(conf, 3), a_ref=a_ref, b_ref=b_ref,
            value_change=vc, evidence=ev))
        n += 1

    # --- matched -> modified / moved(sheet change) / unchanged ---
    # Intra-sheet XY moves of text are NOT reported (position is not meaningful on
    # a P&ID). A change of sheet/page IS meaningful -> "moved".
    for ra, rb in al.matched:
        change = _classify_modify(ra, rb, cfg)
        pa, pb = _page_of(ra), _page_of(rb)
        if change is not None:
            desc, vc = change
            ev = {"a_text": ra.text, "b_text": rb.text}
            if pa != pb:
                ev["sheet_change"] = [pa, pb]
            add("modified", ra.kind, rb, desc, min(ra.confidence, rb.confidence),
                a_ref=ra.region_id, b_ref=rb.region_id, vc=vc, ev=ev)
        elif pa != pb:
            add("moved", ra.kind, rb, f"relocated: sheet {pa} -> {pb}",
                min(ra.confidence, rb.confidence), a_ref=ra.region_id, b_ref=rb.region_id,
                ev={"reason": "sheet_change", "from_sheet": pa, "to_sheet": pb})

    # --- unmatched -> removed / added ---
    # A "DELETED" note is a kept-for-numbering placeholder, not content. If it is
    # only on one side (the other renumbered/merged it), that is not a meaningful
    # add/remove — both revisions consider it deleted. Suppress to avoid noise.
    for r in al.only_a:
        if _is_deleted_placeholder(r):
            continue
        add("removed", r.kind, r, _summ(r), r.confidence, a_ref=r.region_id)
    for r in al.only_b:
        if _is_deleted_placeholder(r):
            continue
        add("added", r.kind, r, _summ(r), r.confidence, b_ref=r.region_id)

    # --- tag + geometry set-diffs ---
    _tag_delta(doc_a, doc_b, add)
    _geometry_delta(doc_a, doc_b, cfg, add)

    if use_llm and cfg.use_llm_annotation:
        _annotate_llm(items, cfg)

    return Delta(comparison_id=comparison_id, summary=_summarize(items), items=items)


def _classify_modify(ra: Region, rb: Region, cfg):
    """Return (description, ValueChange|None) if content changed, else None."""
    if ra.kind == "title_block":
        va, vb = _clean_value(ra.attrs.get("value", "")), _clean_value(rb.attrs.get("value", ""))
        if va == vb:
            return None
        na, nb = _num(va), _num(vb)
        numeric = na is not None and nb is not None and not re.search(r"[A-Za-z]", va + vb)
        vc = ValueChange(va, vb, round(nb - na, 3) if numeric else None)
        return f"{ra.attrs.get('field_label', 'field')}: {va!r} -> {vb!r}", vc

    if ra.kind == "dimension":
        sa = {s["limit"]: s["value"] for s in ra.attrs.get("setpoints", [])}
        sb = {s["limit"]: s["value"] for s in rb.attrs.get("setpoints", [])}
        diffs = [(k, sa[k], sb[k]) for k in sa if k in sb and sa[k] != sb[k]]
        if not diffs:
            return None
        k, oa, ob = diffs[0]
        return f"setpoint {k}: {oa} -> {ob}", ValueChange(str(oa), str(ob), round(ob - oa, 3), "barg")

    if text_sim(ra.text, rb.text) >= (1 - cfg.modify_min_text_delta):
        return None
    return f"text changed: {ra.text[:60]!r} -> {rb.text[:60]!r}", None


def _is_deleted_placeholder(r: Region) -> bool:
    """A note whose body is just 'DELETED' — a renumbering placeholder, not content."""
    if r.kind != "note_item":
        return False
    body = re.sub(r"^\s*\d+\.\s*", "", r.text or "").strip().rstrip(".").upper()
    return body == "DELETED"


def _summ(r: Region) -> str:
    if r.kind == "note_item":
        return f"note {r.attrs.get('note_number')}: {r.text[:70]}"
    if r.kind == "title_block":
        return f"{r.attrs.get('field_label')} = {r.attrs.get('value','')[:40]}"
    return f"{r.kind}: {r.text[:70]}"


def _all_tag_ids(doc: Document) -> dict:
    out = {}
    for p in doc.pages:
        for r in p.regions:
            for t in r.attrs.get("tag_ids", []):
                out.setdefault(t, r)
    return out


def _tag_delta(doc_a, doc_b, add) -> None:
    a, b = _all_tag_ids(doc_a), _all_tag_ids(doc_b)
    for t in sorted(set(a) - set(b)):
        add("removed", "tag", a[t], f"tag removed: {t}", a[t].confidence,
            a_ref=a[t].region_id, ev={"tag_id": t, "key": f"tag:{t}"})
    for t in sorted(set(b) - set(a)):
        add("added", "tag", b[t], f"tag added: {t}", b[t].confidence,
            b_ref=b[t].region_id, ev={"tag_id": t, "key": f"tag:{t}"})


def _geometry_regions(doc: Document) -> list[Region]:
    return [r for p in doc.pages for r in p.regions if r.kind == "geometry" and r.geometry]


def _geo_key(r: Region) -> str:
    g = r.geometry
    return (g.block_name or g.entity_type or "geom") if g else "geom"


def _geometry_delta(doc_a, doc_b, cfg, add) -> None:
    """Symbol-level geometry diff (added / removed / moved).

    Matches geometry regions by block/entity key + nearest position, then flags
    a relocation as 'moved'. Populated by the DXF adapter (block references);
    native-PDF raster paths are not symbol-segmented (documented cut), so this is
    a no-op for native input.
    """
    ga, gb = _geometry_regions(doc_a), _geometry_regions(doc_b)
    if not ga and not gb:
        return
    from collections import defaultdict
    buckets_b = defaultdict(list)
    for r in gb:
        buckets_b[_geo_key(r)].append(r)
    used_b = set()
    for ra in ga:
        cands = [r for r in buckets_b[_geo_key(ra)] if id(r) not in used_b]
        if not cands:
            add("removed", "geometry", ra, f"symbol removed: {_geo_key(ra)}", ra.confidence,
                a_ref=ra.region_id, ev={"block": _geo_key(ra)})
            continue
        rb = min(cands, key=lambda r: _shift(ra, r))
        used_b.add(id(rb))
        pa, pb = _page_of(ra), _page_of(rb)
        s = _shift(ra, rb)
        if pa != pb:  # sheet change -> meaningful move (full confidence)
            add("moved", "geometry", rb, f"symbol relocated: {_geo_key(ra)} (sheet {pa} -> {pb})",
                min(ra.confidence, rb.confidence), a_ref=ra.region_id, b_ref=rb.region_id,
                ev={"reason": "sheet_change", "block": _geo_key(ra), "from_sheet": pa, "to_sheet": pb})
        elif s > cfg.move_min_shift:  # intra-sheet shift -> LOW-conf layout signal only
            add("moved", "geometry", rb,
                f"symbol layout shift: {_geo_key(ra)} (Δpos={s:.3f}; cosmetic vs topological "
                f"undetermined — needs connectivity)",
                cfg.geometry_layout_conf, a_ref=ra.region_id, b_ref=rb.region_id,
                ev={"reason": "layout_shift", "block": _geo_key(ra), "shift": round(s, 3)})
    for r in gb:
        if id(r) not in used_b:
            add("added", "geometry", r, f"symbol added: {_geo_key(r)}", r.confidence,
                b_ref=r.region_id, ev={"block": _geo_key(r)})


_TEXT_KINDS = {"note_item", "callout"}   # modifies whose significance is ambiguous


def _annotate_llm(items, cfg) -> None:
    """Flag significance + write a crisp description for TEXT modifies (note/callout)
    in one batched, temp-0 call. Skips exact numeric changes (no LLM needed) and
    never changes the structural set — only fills evidence + description. No-op
    without an API key; best-effort (annotation must never break the delta)."""
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        return
    targets = [it for it in items if it.change_type == "modified" and it.kind in _TEXT_KINDS]
    targets = targets[:cfg.annotate_max]
    if not targets:
        return
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from pydantic import BaseModel

        from src.chat.llm import build_chat_model
        from src.config import get_settings
        from src.observability.tracing import record_llm

        class _Ann(BaseModel):
            id: str
            significant: bool   # meaningful engineering change vs trivial reword/typo
            summary: str        # one-line description of what changed

        class _Anns(BaseModel):
            items: list[_Ann]

        payload = "\n".join(
            f"{it.id} | OLD: {it.evidence.get('a_text', '')[:180]} | NEW: {it.evidence.get('b_text', '')[:180]}"
            for it in targets)
        sys = ("For each change (OLD -> NEW) in an engineering document, decide if it is a "
               "MEANINGFUL change (a requirement, scope, value, or reference) or trivial "
               "(reword, punctuation, OCR typo). Return per id: significant (bool) and a "
               "one-line summary of what changed.")
        model = build_chat_model().with_structured_output(_Anns, include_raw=True)
        out = model.invoke([SystemMessage(sys), HumanMessage(payload)])
        parsed, raw = out.get("parsed"), out.get("raw")
        usage = getattr(raw, "usage_metadata", None) or {}
        name = get_settings().llm.chat_model
        record_llm(name, usage.get("input_tokens", 0), usage.get("output_tokens", 0),
                   (usage.get("input_tokens", 0) * 0.15 + usage.get("output_tokens", 0) * 0.60) / 1e6)
        if not parsed:
            return
        by_id = {a.id: a for a in parsed.items}
        for it in targets:
            ann = by_id.get(it.id)
            if ann:
                it.evidence["significant"] = ann.significant
                it.evidence["llm_summary"] = ann.summary
                it.description = ann.summary
    except Exception:
        return


def _summarize(items) -> dict:
    by_type, by_kind = {}, {}
    for it in items:
        by_type[it.change_type] = by_type.get(it.change_type, 0) + 1
        by_kind[it.kind] = by_kind.get(it.kind, 0) + 1
    return {"total": len(items), "by_change_type": by_type, "by_kind": by_kind}
