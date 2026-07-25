"""Shared Chroma index. Semantic chunks (one per Region / DeltaItem, plus doc +
delta overview chunks) with metadata for comparison-scoped retrieval. The store is
a module singleton shared by indexer and retriever."""
from __future__ import annotations

import functools
import os

from src.canonical.model import Document, Region
from src.config import get_settings
from src.delta.engine import Delta, region_key

# region kinds worth embedding as document chunks (skip pure geometry: no text)
_DOC_KINDS = {"note_item", "title_block", "tag", "dimension", "callout", "table", "table_cell", "text_block"}
# thin kinds get a "parent-window": neighbor text folded in for retrieval context
_THIN_KINDS = {"tag", "dimension", "callout"}
_NEIGHBOR_BUDGET = 180   # max chars of neighbor context appended to a thin chunk


@functools.lru_cache(maxsize=1)
def get_store():
    from langchain_chroma import Chroma
    from src.chat.llm import build_embeddings
    cfg = get_settings().chat
    return Chroma(collection_name="corpus", embedding_function=build_embeddings(),
                  persist_directory=cfg.chroma_dir)


def _bbox_str(b) -> str:
    return f"{b.x0n:.4f},{b.y0n:.4f},{b.x1n:.4f},{b.y1n:.4f}"


def _chunk_body(r: Region) -> str:
    # title-block value lives in attrs -> fold it into the text ("DUTY: 1835")
    if r.kind == "title_block":
        return f"{r.attrs.get('field_label', '')}: {r.attrs.get('value', '')}".strip(": ")
    return (r.text or "").strip()


def _doc_chunk(doc: Document, r: Region, comparison_id: str, bodies: dict | None = None,
               role: str = "document"):
    body = _chunk_body(r)
    if r.kind not in _DOC_KINDS or len(body) < 3:
        return None
    if r.kind == "text_block" and len(body) < 15:
        return None   # drop tiny label fragments ("GAS LIFT", "PROVISION FOR") — retrieval noise

    embed_text = body
    if bodies and r.kind in _THIN_KINDS and r.neighbors:
        context_parts, used_len = [], 0
        for neighbor_id in r.neighbors:
            neighbor = bodies.get(neighbor_id, "")
            if not neighbor or neighbor == body:
                continue
            if used_len + len(neighbor) > _NEIGHBOR_BUDGET:
                break
            context_parts.append(neighbor); used_len += len(neighbor)
        if context_parts:
            embed_text = f"{body}  || nearby: {' ; '.join(context_parts)}"   # parent-window

    page = 0
    meta = {
        "pid": doc.pid, "doc_family": doc.doc_family, "rev": doc.rev_label,
        "source_type": "document", "comparison_id": comparison_id,
        "page": page, "region_id": r.region_id, "kind": r.kind, "role": role,
        "key": region_key(r), "bbox": _bbox_str(r.bbox), "confidence": r.confidence,
    }
    if r.attrs.get("tag_ids"):
        meta["tag_id"] = ",".join(r.attrs["tag_ids"])
    header = f"[{role.upper()} doc={doc.doc_family} {r.kind}] "
    return r.region_id, header + embed_text, meta


def _delta_chunk(it, comparison_id: str):
    text = f"[change {it.change_type} {it.kind}] {it.description}"
    meta = {
        "source_type": "delta_report", "comparison_id": comparison_id,
        "item_id": it.id, "change_type": it.change_type, "kind": it.kind,
        "key": it.evidence.get("key", ""), "page": it.page, "bbox": _bbox_str(it.bbox),
        "a_ref": it.a_ref or "", "b_ref": it.b_ref or "", "confidence": it.confidence,
    }
    return it.id, text, meta


class Indexer:
    def __init__(self, cfg=None):
        self.cfg = cfg or get_settings()
        self.store = get_store()

    def _delete_where(self, where: dict) -> None:
        """Remove existing chunks matching a filter (avoids orphans on re-index)."""
        try:
            ids = self.store.get(where=where).get("ids", [])
            if ids:
                self.store.delete(ids=ids)
        except Exception:
            pass

    def _add(self, triples) -> int:
        triples = [t for t in triples if t]
        if not triples:
            return 0
        self.store.add_texts(texts=[t[1] for t in triples], metadatas=[t[2] for t in triples],
                             ids=[t[0] for t in triples])
        return len(triples)

    def upsert_document(self, doc: Document, comparison_id: str, role: str = "document") -> int:
        self._delete_where({"pid": doc.pid})   # clear prior chunks for this doc (incl. filtered-out)
        regions = [r for p in doc.pages for r in p.regions]
        bodies = {r.region_id: _chunk_body(r) for r in regions}   # for parent-window lookup
        n = self._add(_doc_chunk(doc, r, comparison_id, bodies, role) for r in regions)
        self._add([_doc_overview_chunk(doc, comparison_id, role)])  # "what is this doc" identity
        return n

    def upsert_delta(self, delta: Delta) -> int:
        self._delete_where({"$and": [{"comparison_id": delta.comparison_id},
                                     {"source_type": "delta_report"}]})
        n = self._add(_delta_chunk(it, delta.comparison_id) for it in delta.items)
        self._add([_overview_chunk(delta)])   # a retrievable summary for "summarize/overview" queries
        return n


def _doc_overview_chunk(doc: Document, comparison_id: str, role: str = "document"):
    """A retrievable 'what is this document' chunk. Field-agnostic: lists whatever
    title-block fields were extracted (no hardcoded field names), so it works for
    any document type. Optionally an LLM summary (config chat.overview_llm)."""
    fields = {r.attrs["field_label"]: r.attrs.get("value", "")
              for p in doc.pages for r in p.regions
              if r.kind == "title_block" and r.attrs.get("field_label")}
    n_notes = sum(1 for p in doc.pages for r in p.regions if r.kind == "note_item")
    n_tags = len({t for p in doc.pages for r in p.regions for t in r.attrs.get("tag_ids", [])})

    if get_settings().chat.overview_llm and os.environ.get("OPENAI_API_KEY"):
        body = _llm_overview(doc)
    else:
        field_str = "; ".join(f"{k}: {v}" for k, v in list(fields.items())[:12] if v)
        body = f"Fields — {field_str}. {n_notes} notes, {n_tags} tagged items."

    text = f"[{role.upper()} overview of {doc.doc_family}] {body}"
    meta = {"pid": doc.pid, "doc_family": doc.doc_family, "rev": doc.rev_label, "role": role,
            "source_type": "document", "comparison_id": comparison_id, "page": 0,
            "region_id": f"{doc.pid}:overview", "kind": "overview", "key": "doc_overview",
            "bbox": "0,0,0,0", "confidence": 1.0}
    return f"{doc.pid}:overview", text, meta


def _llm_overview(doc: Document) -> str:
    """One-sentence summary from the document's raw text (any doc type)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from src.chat.llm import build_chat_model
    raw = " ".join(p.raw_text for p in doc.pages)[:2500]
    try:
        resp = build_chat_model().invoke([
            SystemMessage("Summarize what this engineering document is, in one sentence."),
            HumanMessage(raw)])
        return (resp.content or "").strip()
    except Exception:
        return f"{doc.doc_family} document."


def _overview_chunk(delta: Delta):
    s = delta.summary
    top = "; ".join(it.description[:70] for it in delta.items[:15])
    text = (f"[comparison overview] Summary of the delta between the base and revised "
            f"documents: {s['total']} total changes "
            f"({', '.join(f'{k} {v}' for k, v in s['by_change_type'].items())}). "
            f"By kind: {', '.join(f'{k} {v}' for k, v in s['by_kind'].items())}. "
            f"Notable changes: {top}")
    meta = {"source_type": "delta_report", "comparison_id": delta.comparison_id,
            "item_id": f"{delta.comparison_id}:overview", "change_type": "overview",
            "kind": "overview", "key": "overview", "page": 0, "bbox": "0,0,0,0",
            "a_ref": "", "b_ref": "", "confidence": 1.0}
    return f"{delta.comparison_id}:overview", text, meta
