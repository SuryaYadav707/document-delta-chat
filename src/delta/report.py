"""Delta report — human-readable + machine-parseable.

All three are written to disk per comparison under artifacts/<comparison_id>/:
  - JSON: full Delta (summary + items) — machine + eval input; served at /report/{cid}
  - Markdown: counts table, grouped by change_type x kind, per-item location +
    confidence + citation anchor — the human-readable form (open the file directly)
  - HTML: same content as a standalone color-coded table (openable in a browser)

The report is also a first-class retrievable source: chat indexes one chunk per
DeltaItem (source_type=delta_report, comparison_id) — see src/chat/index.py.
"""
from __future__ import annotations

import html
import json
from dataclasses import asdict

from src.delta.engine import Delta, DeltaItem

_ORDER = ["modified", "moved", "added", "removed"]
_MARK = {"modified": "~", "moved": "→", "added": "+", "removed": "-"}


def to_dict(delta: Delta) -> dict:
    return {"comparison_id": delta.comparison_id, "summary": delta.summary,
            "items": [asdict(i) for i in delta.items]}


def render_json(delta: Delta) -> str:
    return json.dumps(to_dict(delta), indent=2)


def _cite(it: DeltaItem) -> str:
    ref = it.b_ref or it.a_ref or ""
    return f"[cite: {ref}]" if ref else ""


def _loc(it: DeltaItem) -> str:
    b = it.bbox
    return f"p{it.page} ({b.x0n:.2f},{b.y0n:.2f})-({b.x1n:.2f},{b.y1n:.2f})"


def render_markdown(delta: Delta) -> str:
    s = delta.summary
    out = [f"# Delta report — `{delta.comparison_id}`", ""]
    out.append(f"**{s['total']} changes** — "
               + ", ".join(f"{k}: {v}" for k, v in sorted(s["by_change_type"].items())))
    out.append("")
    out.append("| kind | count |")
    out.append("|---|---|")
    for k, v in sorted(s["by_kind"].items(), key=lambda x: -x[1]):
        out.append(f"| {k} | {v} |")
    out.append("")
    for ct in _ORDER:
        items = [i for i in delta.items if i.change_type == ct]
        if not items:
            continue
        out.append(f"## {ct} ({len(items)})")
        for it in sorted(items, key=lambda i: (i.kind, -i.confidence)):
            vc = ""
            if it.value_change and it.value_change.numeric_delta is not None:
                vc = f" **Δ={it.value_change.numeric_delta}**"
            out.append(f"- `{_MARK[ct]}` **[{it.kind}]** {it.description}{vc} "
                       f"— _conf {it.confidence}_ · {_loc(it)} {_cite(it)}")
        out.append("")
    return "\n".join(out)


def render_html(delta: Delta) -> str:
    s = delta.summary
    rows = []
    for ct in _ORDER:
        for it in sorted([i for i in delta.items if i.change_type == ct],
                         key=lambda i: (i.kind, -i.confidence)):
            vc = (f"Δ={it.value_change.numeric_delta}"
                  if it.value_change and it.value_change.numeric_delta is not None else "")
            rows.append(
                f"<tr class='{ct}'><td>{ct}</td><td>{it.kind}</td>"
                f"<td>{html.escape(it.description)}</td><td>{vc}</td>"
                f"<td>{it.confidence}</td><td>{_loc(it)}</td>"
                f"<td>{html.escape(it.b_ref or it.a_ref or '')}</td></tr>")
    counts = ", ".join(f"{k}: {v}" for k, v in sorted(s["by_change_type"].items()))
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Delta report — {html.escape(delta.comparison_id)}</title>
<style>body{{font:14px system-ui;margin:24px}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left}}
tr.modified{{background:#fff7e6}}tr.added{{background:#e6ffed}}tr.removed{{background:#ffeef0}}</style>
</head><body>
<h1>Delta report — {html.escape(delta.comparison_id)}</h1>
<p><b>{s['total']} changes</b> — {counts}</p>
<table><thead><tr><th>change</th><th>kind</th><th>description</th><th>Δ</th>
<th>conf</th><th>location</th><th>cite</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></body></html>"""
