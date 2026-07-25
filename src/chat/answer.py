"""Grounded answer + citation validation. The model answers from the numbered
context and cites [n]; cites are validated against retrieved chunks (out-of-range
dropped), and an unsupported / empty-citation answer is flagged grounded=False."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.chat.retrieve import Chunk
from src.observability.tracing import record_llm

# rough per-1M-token USD (input, output) for cost telemetry
_PRICE = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.5, 10.0)}

_SYS = (
    "You answer questions about two revisions of an engineering document (a P&ID) and "
    "the delta between them, using the numbered context provided.\n"
    "\nANSWER EXACTLY WHAT IS ASKED (this is important):\n"
    "- Specific question (a value, a field, whether X changed, one note/tag): answer in 1-2 "
    "sentences and cite the source [n]. Do NOT list other changes that merely happen to be in "
    "the context but were not asked about.\n"
    "- Broad / summary / list question ('what changed', 'all added'): open with ONE plain "
    "sentence giving the count + gist (e.g. 'The revised sheet adds 11 items — mostly new "
    "instrumentation:'), then GROUP the items by kind (tags, setpoints, notes, callouts, "
    "title-block) with a count per group.\n"
    "\nSTYLE (write like a helpful engineer, not a data dump):\n"
    "- Describe each item by WHAT it is (tag id, field name, setpoint limit + value); put the "
    "[n] citation right after the claim (e.g. '... 1835 kW [3]').\n"
    "- Attribute values to the correct document when it matters (items are tagged "
    "[BASE ...] / [REVISED ...]).\n"
    "\nGROUNDING:\n"
    "- Use ONLY the context. 'added'/'removed'/'modified' refer to '[change ...]' delta entries; "
    "a note whose text is just 'DELETED' is not itself a removal.\n"
    "- If the context genuinely lacks the answer, reply exactly: 'Not supported by the documents.'\n"
    "Be precise with tags, numbers, and setpoints."
)


@dataclass
class Citation:
    source_type: str
    ref: str
    chunk_id: str
    key: str = ""             # semantic chunk key (field:X / note:N / tag:X) — for eval
    pid: str = ""             # source document path (for the viewer)
    page: int = 0
    bbox: list = field(default_factory=list)   # [x0n,y0n,x1n,y1n] normalized
    n: int = 0                # 1-based context index the [n] marker points at (UI mapping)


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    grounded: bool = True
    trace_id: str = ""


def _label(c: Chunk) -> str:
    if c.meta.get("source_type") == "delta_report":
        return f"delta:{c.meta.get('item_id')}"
    return c.meta.get("region_id", "")


def _context(chunks: list[Chunk]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        src = "delta" if c.meta.get("source_type") == "delta_report" else \
            f"{c.meta.get('rev')} p{c.meta.get('page')}"
        lines.append(f"[{i}] ({src}) {c.text}")
    return "\n".join(lines)


def _cost(model: str, usage: dict) -> float:
    pin, pout = _PRICE.get(model, _PRICE["gpt-4o-mini"])
    return (usage.get("input_tokens", 0) * pin + usage.get("output_tokens", 0) * pout) / 1e6


def answer(query: str, chunks: list[Chunk], cfg=None) -> Answer:
    from langchain_core.messages import HumanMessage, SystemMessage
    from src.chat.llm import build_chat_model
    from src.config import get_settings
    from src.observability.tracing import langchain_callbacks

    settings = get_settings()
    if not chunks:
        return Answer("Not supported by the documents.", [], grounded=False)

    model = build_chat_model()
    msgs = [SystemMessage(_SYS),
            HumanMessage(f"Question: {query}\n\nContext:\n{_context(chunks)}")]
    resp = model.invoke(msgs, config={"callbacks": langchain_callbacks()})
    text = (resp.content or "").strip()

    usage = getattr(resp, "usage_metadata", None) or {}
    mname = settings.llm.chat_model
    record_llm(mname, usage.get("input_tokens", 0), usage.get("output_tokens", 0),
               _cost(mname, usage))

    citations, grounded = _validate_citations(text, chunks)
    return Answer(text, citations, grounded)


def answer_stream(query: str, chunks: list[Chunk], cfg=None):
    """Streaming variant of answer(). Yields ("token", str) as the model emits text,
    then one final ("final", Answer) whose citations are validated against the full
    text — citations can only resolve once every [n] marker has been seen."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from src.chat.llm import build_chat_model
    from src.config import get_settings
    from src.observability.tracing import langchain_callbacks

    settings = get_settings()
    if not chunks:
        yield "final", Answer("Not supported by the documents.", [], grounded=False)
        return

    model = build_chat_model()
    msgs = [SystemMessage(_SYS),
            HumanMessage(f"Question: {query}\n\nContext:\n{_context(chunks)}")]
    cb = {"callbacks": langchain_callbacks()}
    # stream_usage is a ChatOpenAI kwarg; skip it for other providers to avoid errors.
    extra = {"stream_usage": True} if settings.llm.provider == "openai" else {}

    parts: list[str] = []
    usage: dict = {}
    try:
        for chunk in model.stream(msgs, config=cb, **extra):
            piece = chunk.content if isinstance(chunk.content, str) else ""
            if piece:
                parts.append(piece)
                yield "token", piece
            um = getattr(chunk, "usage_metadata", None)
            if um:
                usage = um
    except Exception as e:  # surface streaming failures as a graceful refusal
        text = "".join(parts).strip()
        if not text:
            yield "final", Answer(f"Streaming failed: {e}", [], grounded=False)
            return
    text = "".join(parts).strip()

    mname = settings.llm.chat_model
    record_llm(mname, usage.get("input_tokens", 0), usage.get("output_tokens", 0),
               _cost(mname, usage))
    citations, grounded = _validate_citations(text, chunks)
    yield "final", Answer(text, citations, grounded)


def _validate_citations(text: str, chunks: list[Chunk]):
    """Keep only in-range [n] markers; drop hallucinated/duplicate refs.
    Returns (citations, grounded). Each Citation carries the n it was cited by."""
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    citations, seen = [], set()
    for n in sorted(cited):
        if 1 <= n <= len(chunks):
            c = chunks[n - 1]
            ref = _label(c)
            if ref in seen:
                continue
            seen.add(ref)
            cit = _citation(c, ref)
            cit.n = n
            citations.append(cit)
    # overview chunks are synthetic summaries with no page region -> keep as a citation
    # only when nothing concrete was cited (drop the redundant/dead-end ref otherwise).
    concrete = [c for c in citations if c.key not in ("overview", "doc_overview")]
    if concrete:
        citations = concrete
    grounded = "not supported by the documents" not in text.lower() and bool(citations)
    return citations, grounded


def _citation(chunk: Chunk, ref: str) -> Citation:
    meta = chunk.meta
    source_type = meta.get("source_type", "document")
    if source_type == "delta_report":
        pid = (meta.get("b_ref") or meta.get("a_ref") or "").rsplit(":", 2)[0]
    else:
        pid = meta.get("pid", "")
    try:
        bbox = [float(x) for x in meta.get("bbox", "").split(",")] if meta.get("bbox") else []
    except ValueError:
        bbox = []
    return Citation(source_type, ref, chunk.id, key=meta.get("key", ""),
                    pid=pid, page=int(meta.get("page", 0)), bbox=bbox)
