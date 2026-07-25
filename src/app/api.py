"""FastAPI app — REST + serves the minimal citation-viewer UI.

Endpoints:
  GET  /                         -> single-page UI
  GET  /comparisons              -> existing comparisons (from artifacts/*/meta.json)
  POST /upload (multipart file)  -> save under data/uploads/ -> {pid, name}
  POST /compare {pid_a, pid_b}   -> build a comparison -> {comparison_id, summary}
  GET  /report/{cid}             -> delta report JSON
  POST /chat {comparison_id, query} -> {text, grounded, citations[], trace_id}
  POST /chat/stream {comparison_id, query} -> SSE: token* then final|error
  GET  /page?pid=&page=          -> rendered page PNG (for the citation viewer)
  GET  /metrics                  -> observability summary (traces on disk)

Security: /page only renders PIDs referenced by an existing comparison's meta.json
(no arbitrary filesystem reads / path traversal).
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
import uuid
from dataclasses import asdict
from pathlib import Path

import fitz
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from src.config import get_settings
from src.observability.logging import configure

configure(get_settings().observability.log_level)
app = FastAPI(title="Document Delta & Grounded Chat")

_UI = Path(__file__).parent / "ui" / "index.html"


def _artifacts() -> Path:
    return Path(get_settings().paths.artifacts_dir)


def _known_pids() -> set[str]:
    pids = set()
    for m in _artifacts().glob("*/meta.json"):
        try:
            d = json.loads(m.read_text())
            pids.update([d.get("pid_a"), d.get("pid_b")])
        except Exception:
            continue
    return {p for p in pids if p}


@app.get("/", response_class=HTMLResponse)
def index():
    return _UI.read_text()


@app.get("/comparisons")
def comparisons():
    out = []
    for m in _artifacts().glob("*/meta.json"):
        d = json.loads(m.read_text())
        rep = m.parent / "report.json"
        d["summary"] = json.loads(rep.read_text())["summary"] if rep.exists() else {}
        out.append(d)
    return out


_UPLOAD_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dxf"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Accept a document upload -> save under data/uploads/ -> return its PID (path)."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _UPLOAD_EXTS:
        raise HTTPException(400, f"unsupported type: {ext or '?'} (allowed: {sorted(_UPLOAD_EXTS)})")
    updir = Path("data/uploads")
    updir.mkdir(parents=True, exist_ok=True)
    safe = Path(file.filename).name                      # strip any path components
    dest = updir / f"{uuid.uuid4().hex[:8]}_{safe}"      # uuid prefix: no collision/overwrite
    dest.write_bytes(await file.read())
    return {"pid": str(dest), "name": safe}


@app.post("/compare")
def compare(payload: dict):
    a, b = payload.get("pid_a"), payload.get("pid_b")
    if not a or not b:
        raise HTTPException(400, "pid_a and pid_b required")
    from src.services.comparison import ComparisonService
    cmp = ComparisonService().create(a, b, use_llm=payload.get("use_llm", False))
    return {"comparison_id": cmp.comparison_id, "summary": cmp.summary}


@app.get("/report/{cid}")
def report(cid: str):
    p = _artifacts() / cid / "report.json"
    if not p.exists():
        raise HTTPException(404, "no such comparison")
    return JSONResponse(json.loads(p.read_text()))


@app.post("/chat")
def chat(payload: dict):
    cid, q = payload.get("comparison_id"), payload.get("query")
    if not cid or not q:
        raise HTTPException(400, "comparison_id and query required")
    from src.services.chat_service import ChatService
    try:
        ans = ChatService().ask(cid, q)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return {"text": ans.text, "grounded": ans.grounded, "trace_id": ans.trace_id,
            "citations": [asdict(c) for c in ans.citations]}


@app.post("/chat/stream")
async def chat_stream(payload: dict):
    """SSE token stream for a grounded answer. Emits `token` events as the model
    writes, then one `final` event with validated citations (or an `error` event).

    The blocking ChatService generator runs start-to-finish in a dedicated thread
    so its trace contextvars stay bound; tokens cross to the async request via a
    queue, keeping the event loop unblocked."""
    cid, q = payload.get("comparison_id"), payload.get("query")
    if not cid or not q:
        raise HTTPException(400, "comparison_id and query required")
    from src.services.chat_service import ChatService

    events: queue.Queue = queue.Queue()
    _DONE = object()

    def produce():
        try:
            for kind, data in ChatService().ask_stream(cid, q):
                if kind == "token":
                    events.put({"type": "token", "text": data})
                else:  # final Answer
                    events.put({"type": "final", "grounded": data.grounded,
                                "trace_id": data.trace_id,
                                "citations": [asdict(c) for c in data.citations]})
        except FileNotFoundError as e:
            events.put({"type": "error", "message": str(e)})
        except Exception as e:  # surface any failure to the client, then close
            events.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            events.put(_DONE)

    threading.Thread(target=produce, daemon=True).start()

    async def sse():
        loop = asyncio.get_event_loop()
        while True:
            item = await loop.run_in_executor(None, events.get)
            if item is _DONE:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/page")
def page(pid: str, page: int = 0):
    if pid not in _known_pids():
        raise HTTPException(403, "pid not part of any comparison")
    try:
        doc = fitz.open(pid)
        pix = doc[page].get_pixmap(dpi=140)
        return Response(pix.tobytes("png"), media_type="image/png")
    except Exception as e:
        raise HTTPException(500, f"render failed: {e}")


@app.get("/metrics")
def metrics():
    tdir = Path(get_settings().observability.traces_dir)
    traces = sorted(tdir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    latest = json.loads(traces[0].read_text()) if traces else {}
    return {"trace_count": len(traces),
            "latest": {k: latest.get(k) for k in ("kind", "duration_ms", "tokens_in",
                                                  "tokens_out", "cost_usd", "meta")}}
