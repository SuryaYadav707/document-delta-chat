"""ChatService — grounded chat, scoped to one comparison.

ask(comparison_id, query) -> Answer:
  load comparison (pids) -> route -> retrieve (comparison filter) -> answer
  -> validate citations -> grounded answer | refusal
Wrapped in one Langfuse trace with route/retrieve/answer spans.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.chat import answer as answer_mod
from src.chat.retrieve import Retriever
from src.chat.router import route
from src.config import get_settings
from src.observability.logging import get_logger
from src.observability.tracing import set_meta, span, trace

_log = get_logger("chat")


class ChatService:
    def __init__(self, cfg=None):
        self.cfg = cfg or get_settings()
        self.retriever = Retriever(self.cfg)

    def _meta(self, comparison_id: str) -> dict:
        p = Path(self.cfg.paths.artifacts_dir) / comparison_id / "meta.json"
        if not p.exists():
            raise FileNotFoundError(f"unknown comparison (run `make run` first): {comparison_id}")
        return json.loads(p.read_text())

    def ask(self, comparison_id: str, query: str):
        meta = self._meta(comparison_id)
        with trace("chat") as tr:
            set_meta(comparison_id=comparison_id, query=query)
            _log.info("chat.ask", comparison_id=comparison_id, query=query)

            with span("route") as sp:
                rd = route(query, meta["pid_a"], meta["pid_b"], self.cfg,
                           meta.get("doc_family_a", ""), meta.get("doc_family_b", ""))
                sp.attrs.update(intent=rd.intent, pids=len(rd.pid_filter),
                                boost_delta=rd.boost_delta, keys="|".join(rd.keys))

            with span("retrieve") as sp:
                chunks = self.retriever.retrieve(query, comparison_id, rd)
                sp.attrs.update(
                    n_chunks=len(chunks),
                    sources="|".join(sorted({c.meta.get("source_type", "") for c in chunks})),
                    chunks=[{"id": c.id, "source": c.meta.get("source_type"),
                             "key": c.meta.get("key"), "score": c.score,
                             "text": c.text[:100]} for c in chunks],
                )

            with span("answer") as sp:
                ans = answer_mod.answer(query, chunks, self.cfg)
                sp.attrs.update(grounded=ans.grounded, citations=len(ans.citations))

            ans.trace_id = tr.id
            _log.info("chat.done", grounded=ans.grounded, citations=len(ans.citations))
            return ans

    def ask_stream(self, comparison_id: str, query: str):
        """Streaming variant of ask(). Same route/retrieve/trace tree; yields
        ("token", str) chunks live, then a final ("final", Answer). Consume the
        whole generator in ONE thread so the trace contextvars stay intact."""
        meta = self._meta(comparison_id)
        with trace("chat") as tr:
            set_meta(comparison_id=comparison_id, query=query)
            _log.info("chat.ask.stream", comparison_id=comparison_id, query=query)

            with span("route") as sp:
                rd = route(query, meta["pid_a"], meta["pid_b"], self.cfg,
                           meta.get("doc_family_a", ""), meta.get("doc_family_b", ""))
                sp.attrs.update(intent=rd.intent, pids=len(rd.pid_filter),
                                boost_delta=rd.boost_delta, keys="|".join(rd.keys))

            with span("retrieve") as sp:
                chunks = self.retriever.retrieve(query, comparison_id, rd)
                sp.attrs.update(
                    n_chunks=len(chunks),
                    sources="|".join(sorted({c.meta.get("source_type", "") for c in chunks})),
                    chunks=[{"id": c.id, "source": c.meta.get("source_type"),
                             "key": c.meta.get("key"), "score": c.score,
                             "text": c.text[:100]} for c in chunks],
                )

            final = None
            with span("answer") as sp:
                for kind, data in answer_mod.answer_stream(query, chunks, self.cfg):
                    if kind == "token":
                        yield "token", data
                    else:
                        final = data
                sp.attrs.update(grounded=final.grounded, citations=len(final.citations))

            final.trace_id = tr.id
            _log.info("chat.done", grounded=final.grounded, citations=len(final.citations))
            yield "final", final
