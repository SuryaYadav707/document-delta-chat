"""Delta markup overlay (BONUS) — the redline artifact the tool replaces.

Draws the delta back onto the sheets: a BASE page with removed items, and a
REVISED page with added/modified/moved items, each boxed and colour-coded. Reads
the persisted report.json + meta.json for a comparison, so it runs standalone.
"""
from __future__ import annotations

import json
from pathlib import Path

import fitz

from src.config import get_settings

# RGB (0..1) by change type
_COLOR = {"added": (0.0, 0.6, 0.1), "removed": (0.85, 0.1, 0.1),
          "modified": (0.9, 0.6, 0.0), "moved": (0.1, 0.3, 0.9)}
_A_SIDE = {"removed"}                      # items located on the base doc
_B_SIDE = {"added", "modified", "moved"}   # items located on the revised doc


def _draw_page(out: fitz.Document, pid: str, items: list[dict], title: str) -> None:
    with fitz.open(pid) as src:
        out.insert_pdf(src, from_page=0, to_page=0)
    page = out[-1]
    w, h = page.rect.width, page.rect.height
    counts: dict[str, int] = {}
    for it in items:
        b = it["bbox"]
        rect = fitz.Rect(b["x0n"] * w, b["y0n"] * h, b["x1n"] * w, b["y1n"] * h)
        if rect.width < 1 or rect.height < 1:
            continue   # skip degenerate/synthetic boxes
        page.draw_rect(rect, color=_COLOR.get(it["change_type"], (0, 0, 0)), width=1.5)
        counts[it["change_type"]] = counts.get(it["change_type"], 0) + 1
    banner = f"{title}  —  " + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
    page.insert_text((20, 24), banner, fontsize=11, color=(0, 0, 0))


def render_markup(comparison_id: str, cfg=None) -> str:
    """Write an annotated PDF (base page + revised page) -> return its path."""
    cfg = cfg or get_settings()
    art = Path(cfg.paths.artifacts_dir) / comparison_id
    meta = json.loads((art / "meta.json").read_text())
    items = json.loads((art / "report.json").read_text())["items"]

    out = fitz.open()
    _draw_page(out, meta["pid_a"], [it for it in items if it["change_type"] in _A_SIDE],
               "BASE — removed (red)")
    _draw_page(out, meta["pid_b"], [it for it in items if it["change_type"] in _B_SIDE],
               "REVISED — added (green) / modified (amber) / moved (blue)")
    path = str(art / "markup.pdf")
    out.save(path)
    out.close()
    return path
