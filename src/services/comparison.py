"""ComparisonService — the unit-of-work orchestrator (offline pipeline).

create(pid_a, pid_b) -> comparison_id:
  resolve A,B -> ingest (canonical, cached by content hash) -> compute_delta
  -> render report (json+md/html) -> persist artifacts/<comparison_id>/
  -> (index doc + delta chunks — wired when the chat layer lands)

Ingest is cached by SHA-256 of the raw bytes so a document (especially a slow
scanned/OCR one) is parsed once and reused across comparisons.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import src.ingest  # noqa: F401  (registers all format adapters)
from src.canonical.model import Document
from src.config import get_settings
from src.delta import report
from src.delta.engine import compute_delta
from src.ingest.base import get_adapter
from src.ingest.resolver import PIDResolver
from src.observability.logging import get_logger
from src.observability.tracing import set_meta, span, trace

_log = get_logger("comparison")


@dataclass
class Comparison:
    comparison_id: str
    pid_a: str
    pid_b: str
    summary: dict
    artifacts_dir: str


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", Path(s).stem.lower()).strip("-")


class ComparisonService:
    def __init__(self, cfg=None):
        self.cfg = cfg or get_settings()
        self.resolver = PIDResolver()
        self.cache_dir = Path(self.cfg.paths.artifacts_dir) / "_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _ingest_cached(self, pid: str) -> Document:
        resolved = self.resolver.resolve(pid)
        sha = hashlib.sha256(resolved.raw_bytes).hexdigest()[:16]
        cache = self.cache_dir / f"{sha}.pkl"
        if cache.exists():
            return pickle.loads(cache.read_bytes())
        doc = get_adapter(resolved.source_format).parse(resolved)
        cache.write_bytes(pickle.dumps(doc))
        return doc

    def create(self, pid_a: str, pid_b: str, use_llm: bool = True) -> Comparison:
        comparison_id = f"{_slug(pid_a)}__vs__{_slug(pid_b)}"
        with trace("compare", correlation_id=comparison_id):
            set_meta(pid_a=pid_a, pid_b=pid_b, use_llm=use_llm)
            _log.info("compare.start", pid_a=pid_a, pid_b=pid_b)

            with span("ingest.a", pid=pid_a) as sp:
                doc_a = self._ingest_cached(pid_a)
                sp.attrs.update(format=doc_a.source_format,
                                regions=sum(len(p.regions) for p in doc_a.pages))
            with span("ingest.b", pid=pid_b) as sp:
                doc_b = self._ingest_cached(pid_b)
                sp.attrs.update(format=doc_b.source_format,
                                regions=sum(len(p.regions) for p in doc_b.pages))

            with span("delta") as sp:
                delta = compute_delta(doc_a, doc_b, self.cfg.delta,
                                      comparison_id=comparison_id, use_llm=use_llm)
                sp.attrs.update(delta.summary)
            set_meta(delta_summary=delta.summary)

            with span("report"):
                out = Path(self.cfg.paths.artifacts_dir) / comparison_id
                out.mkdir(parents=True, exist_ok=True)
                (out / "report.json").write_text(report.render_json(delta))
                (out / "report.md").write_text(report.render_markdown(delta))
                (out / "report.html").write_text(report.render_html(delta))
                (out / "meta.json").write_text(json.dumps({
                    "comparison_id": comparison_id, "pid_a": pid_a, "pid_b": pid_b,
                    "doc_family_a": doc_a.doc_family, "doc_family_b": doc_b.doc_family}))

            with span("index") as sp:
                from src.chat.index import Indexer
                idx = Indexer(self.cfg)
                nd = (idx.upsert_document(doc_a, comparison_id, role="base")
                      + idx.upsert_document(doc_b, comparison_id, role="revised"))
                ndl = idx.upsert_delta(delta)
                sp.attrs.update(doc_chunks=nd, delta_chunks=ndl)

            _log.info("compare.done", changes=delta.summary["total"])
            return Comparison(comparison_id, pid_a, pid_b, delta.summary, str(out))

    def get(self, comparison_id: str) -> Comparison:
        out = Path(self.cfg.paths.artifacts_dir) / comparison_id
        if not (out / "report.json").exists():
            raise FileNotFoundError(f"unknown comparison: {comparison_id}")
        import json
        summary = json.loads((out / "report.json").read_text())["summary"]
        return Comparison(comparison_id, "", "", summary, str(out))
