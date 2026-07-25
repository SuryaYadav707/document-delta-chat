"""Content alignment between two revisions — the hard part, kept deterministic.

Strategy per Region.kind:
  - note_item   -> identifier-anchored by note_number
  - title_block -> identifier-anchored by field_label
  - dimension / callout -> fuzzy: Hungarian on cost = w_text*(1-text_sim) +
    w_spatial*(1-spatial_sim), accepted below match_threshold
  - tag         -> handled as a set-diff in the engine (tags are scattered, not
    1:1 regions), NOT matched here
  - text_block / legend / geometry -> intentionally NOT diffed (low signal, would
    flood the delta with noise); documented as a scope cut

No LLM here — the set of matched / unmatched items is reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from src.canonical.model import Document, Region

# title_block anchors on an exact key; notes + dimensions have custom matchers
# (renumbering / instrument association); tags are set-diffed in the engine.
ANCHORED = {"title_block": "field_label"}
FUZZY_KINDS = {"callout"}


@dataclass
class Alignment:
    matched: list[tuple[Region, Region]] = field(default_factory=list)
    only_a: list[Region] = field(default_factory=list)   # -> removed
    only_b: list[Region] = field(default_factory=list)   # -> added


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def text_sim(a: str, b: str) -> float:
    return fuzz.token_set_ratio(_norm(a), _norm(b)) / 100.0


def spatial_sim(a: Region, b: Region) -> float:
    ax, ay = (a.bbox.x0n + a.bbox.x1n) / 2, (a.bbox.y0n + a.bbox.y1n) / 2
    bx, by = (b.bbox.x0n + b.bbox.x1n) / 2, (b.bbox.y0n + b.bbox.y1n) / 2
    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    return max(0.0, 1.0 - dist)


def _regions_by_kind(doc: Document, kind: str) -> list[Region]:
    return [r for p in doc.pages for r in p.regions if r.kind == kind]


def _match_anchored(regs_a, regs_b, key: str, align: Alignment) -> None:
    by_a = {r.attrs.get(key): r for r in regs_a if r.attrs.get(key) is not None}
    by_b = {r.attrs.get(key): r for r in regs_b if r.attrs.get(key) is not None}
    for k, ra in by_a.items():
        if k in by_b:
            align.matched.append((ra, by_b[k]))
        else:
            align.only_a.append(ra)
    for k, rb in by_b.items():
        if k not in by_a:
            align.only_b.append(rb)


def _cost(a, b, wt, ws) -> float:
    return wt * (1 - text_sim(a.text, b.text)) + ws * (1 - spatial_sim(a, b))


def _match_fuzzy(regs_a, regs_b, cfg, align: Alignment) -> None:
    if not regs_a or not regs_b:
        align.only_a.extend(regs_a); align.only_b.extend(regs_b)
        return
    wt, ws, thr = cfg.text_sim_weight, cfg.spatial_sim_weight, cfg.match_threshold
    matched_a, matched_b = set(), set()

    if len(regs_a) * len(regs_b) > cfg.large_bucket ** 2:
        # greedy nearest — avoids O(n^3) Hungarian blowup on very large/arbitrary docs
        pairs = sorted(((_cost(a, b, wt, ws), i, j)
                        for i, a in enumerate(regs_a) for j, b in enumerate(regs_b)),
                       key=lambda x: x[0])
        for c, i, j in pairs:
            if c > thr:
                break
            if i not in matched_a and j not in matched_b:
                align.matched.append((regs_a[i], regs_b[j]))
                matched_a.add(i); matched_b.add(j)
    else:
        cost = [[_cost(a, b, wt, ws) for b in regs_b] for a in regs_a]
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if cost[i][j] <= thr:
                align.matched.append((regs_a[i], regs_b[j]))
                matched_a.add(i); matched_b.add(j)

    align.only_a.extend(regs_a[i] for i in range(len(regs_a)) if i not in matched_a)
    align.only_b.extend(regs_b[j] for j in range(len(regs_b)) if j not in matched_b)


def _match_notes(regs_a, regs_b, cfg, align: Alignment) -> None:
    """Anchor notes by note_number ONLY when the text also agrees; the rest re-match
    by content. Handles inserted/renumbered notes (number N no longer = same note)."""
    by_a = {r.attrs.get("note_number"): r for r in regs_a if r.attrs.get("note_number") is not None}
    by_b = {r.attrs.get("note_number"): r for r in regs_b if r.attrs.get("note_number") is not None}
    used_a, used_b = set(), set()
    for num, ra in by_a.items():
        rb = by_b.get(num)
        if rb is not None and text_sim(ra.text, rb.text) >= cfg.note_anchor_sim:
            align.matched.append((ra, rb)); used_a.add(num); used_b.add(num)
    _match_fuzzy([r for n, r in by_a.items() if n not in used_a],
                 [r for n, r in by_b.items() if n not in used_b], cfg, align)


def _match_dimensions(regs_a, regs_b, cfg, align: Alignment) -> None:
    """Anchor setpoints by (instrument, limit) so 'HH on 9062' matches 'HH on 9062'
    across revisions instead of mispairing by raw text/position; rest -> fuzzy."""
    def keyed(regs):
        out = {}
        for r in regs:
            inst = r.attrs.get("instrument")
            if inst:
                for s in r.attrs.get("setpoints", []):
                    out.setdefault((inst, s["limit"]), r)
        return out

    ka, kb = keyed(regs_a), keyed(regs_b)
    used_a, used_b = set(), set()
    for key, ra in ka.items():
        rb = kb.get(key)
        if rb is not None and id(ra) not in used_a and id(rb) not in used_b:
            align.matched.append((ra, rb)); used_a.add(id(ra)); used_b.add(id(rb))
    _match_fuzzy([r for r in regs_a if id(r) not in used_a],
                 [r for r in regs_b if id(r) not in used_b], cfg, align)


def align(doc_a: Document, doc_b: Document, cfg) -> Alignment:
    al = Alignment()
    for kind, key in ANCHORED.items():
        _match_anchored(_regions_by_kind(doc_a, kind), _regions_by_kind(doc_b, kind), key, al)
    _match_notes(_regions_by_kind(doc_a, "note_item"), _regions_by_kind(doc_b, "note_item"), cfg, al)
    _match_dimensions(_regions_by_kind(doc_a, "dimension"), _regions_by_kind(doc_b, "dimension"), cfg, al)
    for kind in FUZZY_KINDS:
        _match_fuzzy(_regions_by_kind(doc_a, kind), _regions_by_kind(doc_b, kind), cfg, al)
    return al
