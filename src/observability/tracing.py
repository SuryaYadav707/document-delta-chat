"""Lightweight homegrown tracer (+ optional Langfuse bridge).

Each request writes a per-run JSON trace file to traces/ with nested timed spans
and LLM token/cost — zero infra. If LANGFUSE_* keys are set, the trace tree is
also mirrored to Langfuse and langchain_callbacks() attaches its handler;
otherwise it degrades to the file (best-effort, never breaks the pipeline).
correlation_id == trace_id, so structured logs join to traces."""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

_current_trace: ContextVar["Trace | None"] = ContextVar("_current_trace", default=None)
_span_stack: ContextVar[list] = ContextVar("_span_stack", default=[])


@dataclass
class Span:
    name: str
    start: float
    end: float | None = None
    status: str = "ok"
    attrs: dict = field(default_factory=dict)
    children: list = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return round(((self.end or time.time()) - self.start) * 1000, 1)

    def to_dict(self) -> dict:
        return {"name": self.name, "duration_ms": self.duration_ms, "status": self.status,
                "attrs": self.attrs, "children": [c.to_dict() for c in self.children]}


@dataclass
class Trace:
    id: str
    kind: str
    start: float
    correlation_id: str
    meta: dict = field(default_factory=dict)
    spans: list = field(default_factory=list)          # root spans
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    @property
    def comparison_id(self) -> str:
        # chat traces carry it in meta; compare traces use it as the correlation/trace id.
        return self.meta.get("comparison_id") or self.correlation_id

    def to_dict(self) -> dict:
        return {"trace_id": self.id, "kind": self.kind, "comparison_id": self.comparison_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.start)),
                "correlation_id": self.correlation_id,
                "duration_ms": round((time.time() - self.start) * 1000, 1), "meta": self.meta,
                "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                "cost_usd": round(self.cost_usd, 6), "spans": [s.to_dict() for s in self.spans]}


def _langfuse_enabled() -> bool:
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    sk = os.environ.get("LANGFUSE_SECRET_KEY", "")
    return bool(pk and sk and "..." not in pk and "..." not in sk)


@contextmanager
def trace(kind: str, correlation_id: str | None = None):
    """Root trace context. Writes traces/<id>.json on exit; mirrors to Langfuse if on."""
    from src.config import get_settings
    trace_id = correlation_id or uuid.uuid4().hex[:16]
    tr = Trace(id=trace_id, kind=kind, start=time.time(), correlation_id=trace_id)
    token = _current_trace.set(tr)
    stack = _span_stack.set([])
    try:
        yield tr
    finally:
        _current_trace.reset(token)
        _span_stack.reset(stack)
        try:
            out_dir = Path(get_settings().observability.traces_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            # filename: <time>__<kind>__<comparison>__<shortid> — sorts chronologically
            # and shows which comparison the trace belongs to at a glance.
            ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(tr.start))
            comp = re.sub(r"[^\w.-]", "_", tr.comparison_id or "trace")[:80]
            (out_dir / f"{ts}__{tr.kind}__{comp}__{tr.id[:6]}.json").write_text(
                json.dumps(tr.to_dict(), indent=2))
        except Exception:
            pass
        if _langfuse_enabled():
            _mirror_to_langfuse(tr)


@contextmanager
def span(name: str, **attrs):
    """Timed span; nests under the current span/trace. status=error on exception."""
    tr = _current_trace.get()
    sp = Span(name=name, start=time.time(), attrs=dict(attrs))
    stack = _span_stack.get()
    (stack[-1].children if stack else (tr.spans if tr else [])).append(sp)
    _span_stack.set(stack + [sp])
    try:
        yield sp
    except Exception as e:
        sp.status = "error"; sp.attrs["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        sp.end = time.time()
        _span_stack.set(stack)


def record_llm(model: str, tokens_in: int = 0, tokens_out: int = 0, cost_usd: float = 0.0) -> None:
    """Attach LLM telemetry to the current span + accumulate on the trace."""
    tr = _current_trace.get()
    if tr:
        tr.tokens_in += tokens_in; tr.tokens_out += tokens_out; tr.cost_usd += cost_usd
    stack = _span_stack.get()
    if stack:
        attrs = stack[-1].attrs
        attrs["model"] = model
        attrs["tokens_in"] = attrs.get("tokens_in", 0) + tokens_in
        attrs["tokens_out"] = attrs.get("tokens_out", 0) + tokens_out
        attrs["cost_usd"] = round(attrs.get("cost_usd", 0.0) + cost_usd, 6)


def set_meta(**kv) -> None:
    tr = _current_trace.get()
    if tr:
        tr.meta.update(kv)


def langchain_callbacks() -> list:
    """LangChain callbacks so LLM calls report tokens/cost to Langfuse (empty if off)."""
    if not _langfuse_enabled():
        return []
    try:
        from langfuse.callback import CallbackHandler
        return [CallbackHandler()]
    except Exception:
        return []


def _mirror_to_langfuse(tr: Trace) -> None:
    try:
        from langfuse import Langfuse
        client = Langfuse()
        lf_trace = client.trace(id=tr.id, name=tr.kind, metadata=tr.meta)

        def walk(sp: Span, parent):
            node = parent.span(name=sp.name, metadata=sp.attrs,
                               start_time=None, end_time=None)
            for c in sp.children:
                walk(c, node)
        for s in tr.spans:
            walk(s, lf_trace)
        client.flush()
    except Exception:
        pass  # observability must never break the pipeline
