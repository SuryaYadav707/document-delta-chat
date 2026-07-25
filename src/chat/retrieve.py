"""Comparison-scoped retrieval. Hybrid: an exact metadata-key fetch (note/tag/
field) first, then vector similarity — both restricted to the active comparison
(delta entries for change queries, the referenced document(s) for content ones)."""
from __future__ import annotations

from dataclasses import dataclass

from src.chat.index import get_store
from src.chat.router import RouteDecision


@dataclass
class Chunk:
    id: str
    text: str
    meta: dict
    score: float = 0.0


def _scope_filter(comparison_id: str, pid_filter: list[str], intent: str) -> dict:
    # change query -> answer from the DELTA entries (not document text, which may
    # merely contain the word "DELETED"). content query -> the document itself.
    if intent == "changed":
        return {"$and": [{"source_type": "delta_report"}, {"comparison_id": comparison_id}]}
    return {"$and": [{"source_type": "document"}, {"pid": {"$in": pid_filter}}]}


class Retriever:
    def __init__(self, cfg=None):
        from src.config import get_settings
        self.cfg = cfg or get_settings()
        self.store = get_store()

    _AGG_CAP = 80   # max delta items to enumerate for a "list all / summarize" query

    def retrieve(self, query: str, comparison_id: str, route: RouteDecision) -> list[Chunk]:
        k = self.cfg.chat.top_k

        # aggregate query ("list all added", "summarize what changed") -> pull the FULL
        # matching delta set, not top-k, so nothing is truncated.
        if route.aggregate and route.intent == "changed":
            conds = [{"source_type": "delta_report"}, {"comparison_id": comparison_id}]
            if route.change_types:
                conds.append({"change_type": {"$in": route.change_types}})
            got = self.store.get(where={"$and": conds})
            chunks = [Chunk(cid, text, meta, 0.0) for cid, text, meta
                      in zip(got.get("ids", []), got.get("documents", []), got.get("metadatas", []))
                      if meta.get("kind") != "overview"][: self._AGG_CAP]
            if chunks:
                return chunks

        where = _scope_filter(comparison_id, route.pid_filter, route.intent)
        chunks: list[Chunk] = []
        seen: set[str] = set()

        # hybrid: exact metadata-key fetch first (note:N / tag:X / field:LABEL) —
        # guarantees the precisely-referenced chunk is in context, which semantic
        # similarity alone misses for numbers/ids.
        if route.keys:
            kwhere = {"$and": [where, {"key": {"$in": route.keys}}]}
            try:
                got = self.store.get(where=kwhere)
                for chunk_id, text, meta in zip(got.get("ids", []), got.get("documents", []),
                                                got.get("metadatas", [])):
                    if chunk_id not in seen:
                        seen.add(chunk_id); chunks.append(Chunk(chunk_id, text, meta, 0.0))
            except Exception:
                pass

        for hit, score in self.store.similarity_search_with_score(query, k=k, filter=where):
            chunk_id = hit.metadata.get("region_id") or hit.metadata.get("item_id", "")
            if chunk_id not in seen:
                seen.add(chunk_id)
                chunks.append(Chunk(chunk_id, hit.page_content, hit.metadata, round(float(score), 3)))
        return chunks
