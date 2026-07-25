# Demo — Document Delta & Grounded Chat (2–4 min)

A walkthrough of one **delta**, one **grounded chat** exchange (with a clickable
citation), the **eval scorecard**, and **observability**. Everything below is
copy-pasteable. Base = Lift compressor (26-KA-901), Revised = Export (26-KA-902).

**Prereqs:** `source .venv/bin/activate` and an `OPENAI_API_KEY` in `.env`
(`cp .env.example .env`, then paste your key). The delta + eval-delta run without
a key; chat + Ragas need it.

---

## 1. Delta — ingest → align → typed report (~15 s, deterministic)

```bash
make run A=data/samples/lift-gas-compressor.pdf B=data/samples/export-gas-compressor.pdf
```
Prints a summary (~89 changes). Open the human report:
```bash
sed -n '1,40p' artifacts/lift-gas-compressor__vs__export-gas-compressor/report.md
```
**Point out** — it's a *typed* delta, not a text diff:
- `DUTY: '776' -> '1835'  Δ=1059`, `FLOW RATE 19057 -> 62809` — numeric field changes
- `setpoint HH: 245 -> 214` — parsed setpoints (instrument-anchored)
- tags renumbered (`-26-KA-901 / +26-KA-902`), DELETED-note placeholders suppressed
- every item has a **location (bbox) + confidence**. `report.json` is the machine form.

## 2. Markup overlay (bonus) — the redline artifact it replaces

```bash
make markup CMP=lift-gas-compressor__vs__export-gas-compressor
```
Opens `artifacts/.../markup.pdf` — **page 1 (base)** with removed items boxed red,
**page 2 (revised)** with added (green) / modified (amber) / moved (blue).

## 3. Grounded chat — cited answers + citation viewer

```bash
make chat            # http://localhost:8001
```
In the UI (the comparison from step 1 is already in the **Active Comparison**
dropdown — it persists on disk):

1. **Ask:** `did the compressor duty change?`
   → *"Yes, the duty changed from 776 to 1835 kW [1]."*
   → **click the [1] badge** → the cited page renders on the right with the **DUTY
   region highlighted**. That's the grounding — every claim traces to a location.
2. **Ask:** `summarize what changed`
   → a human summary: one-line gist + changes **grouped by kind with counts**
   (pulls the *full* delta set, not a truncated top-k).
3. **Ask:** `what is the warranty period of the compressor?`
   → *"Not supported by the documents."* — it **refuses** instead of hallucinating.

*(Optional — OCR path: upload `data/samples/export-scanned.pdf` as Revised and
Compare. The trace's ingest span shows `format=scanned_pdf` + PaddleOCR time; chat
then answers off the OCR'd text.)*

## 4. Eval scorecard — "is it actually good?"

```bash
make eval
```
Prints:
- **Delta**: synthetic-revision **F1=1.00** (exact GT), sister-units **F1=0.93**
  (real pair, title-block scope), scanned-noise **FP count** (OCR-induced noise).
- **Chat (deterministic)**: citation P/R/F1, retrieval **recall@k=1.0**, refusal
  **accuracy=1.0** (0 hallucination, 0 false-refusal), chunk **utilization**.
- **Chat (Ragas, judge = gpt-4o ≠ the answer model)**: faithfulness, relevancy,
  context precision/recall, factual correctness.

Regression guard:
```bash
make eval-baseline       # freeze a reference run once -> results/baseline.json
make eval eval-compare   # re-run (-> run-<timestamp>.json), diff baseline vs newest; exit 1 on F1 drop
```

## 5. Observability — every request is traceable

```bash
cat $(ls -t traces/*.json | head -1) | python -m json.tool | head -40
```
One JSON trace per request: nested spans (`ingest → delta → report`, or
`route → retrieve → answer`) with **per-stage timings**, the **retrieved chunks**,
**tokens + cost**, and a `correlation_id` that matches the structured JSON logs.
Also served live:
```bash
curl -s localhost:8001/metrics | python -m json.tool
```
*(Set `LANGFUSE_*` keys + `make up` to mirror the same traces into the Langfuse UI —
optional; the file traces satisfy the requirement with zero infra.)*

---

## One-line talking points
- **Delta ≠ text diff** — content is aligned by identity (note#, field, tag,
  instrument) then classified + typed; the LLM is *out* of the structural path
  (reproducible with `--no-llm`), *in* only for OCR-verify and change-description.
- **Grounded chat** — comparison-scoped retrieval, citations validated against the
  retrieved set, refuses when unsupported; answers trace to a highlightable region.
- **Measured, not asserted** — one command prints P/R/F1 + RAG metrics; a second
  catches regressions.
