# Eval Scorecard

Snapshot of a full `make eval` run (delta + chat + Ragas). Regenerate any time with
`make eval`; this file is the human-readable summary of `eval/results/run-*.json`.

Judge model for the LLM-graded metrics is **`gpt-4o`** — deliberately *not* the
`gpt-4o-mini` answer model, so the judge never grades its own output.

---

## Delta quality (deterministic, `use_llm=False`)

Predicted `DeltaItem`s matched to gold by `(kind, change_type, location, value)`.

| Pair | Precision | Recall | **F1** | Notes |
|---|---:|---:|---:|---|
| **synthetic-revision** | 1.00 | 1.00 | **1.00** | authored edit list = exact ground truth (10/10) |
| **sister-units** | 0.875 | 1.00 | **0.933** | real pair, title-block scope; 1 false positive |
| **scanned-noise** | — | — | — | OCR-twin, ideal delta = 0 → **19 false positives** |

**synthetic-revision — by change type** (all perfect, 10 items):

| Change type | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| added | 3 | 0 | 0 | 1.00 |
| removed | 4 | 0 | 0 | 1.00 |
| modified | 2 | 0 | 0 | 1.00 |
| moved | 1 | 0 | 0 | 1.00 |

Covered kinds: `tag`, `note_item`, `title_block`, `dimension`, `callout`, `geometry` — every
type the engine emits is exercised.

**scanned-noise — false positives by kind** (this is the *honesty* metric — every count is
OCR error, not a real change):

| Kind | FPs |
|---|---:|
| title_block | 7 |
| note_item | 5 |
| callout | 4 |
| dimension | 2 |
| tag | 1 |
| **total** | **19** |

> The scanned pair is a rasterized twin of a document compared against *itself* — the ideal
> delta is empty, so all 19 detections are OCR-induced noise. Reporting the number honestly
> quantifies how much the OCR path costs in precision.

---

## Grounded chat

**Citation accuracy** — do cited `[n]` markers point at chunks that support the claim, vs gold
citation targets (16 answerable questions):

| Metric | Value |
|---|---:|
| precision | 0.938 |
| recall | 0.938 |
| **F1** | **0.938** |
| TP / FP / FN | 15 / 1 / 1 |

**Retrieval** — is the gold-supporting chunk in the retrieved set?

| Metric | Value |
|---|---:|
| **recall@k** | **1.00** (16/16) |

**Refusal** — does it refuse the unanswerable and answer the answerable? (5 unanswerable in set)

| Metric | Value |
|---|---:|
| accuracy | 0.952 |
| refuse precision | 0.833 |
| refuse recall | 1.00 |
| refuse F1 | 0.909 |
| refuse TP / FP / FN | 5 / 1 / 0 |

> Recall 1.0 = it never answered an unsupported question (0 hallucinated answers). Precision
> 0.833 = one answerable question was refused too conservatively — the single honest miss.

**Utilization** — how much of retrieved context is actually cited:

| Metric | Value |
|---|---:|
| avg retrieved / question | 8.19 |
| avg cited / question | 0.76 |
| utilization | 0.093 |

> Low by design: answers cite the *minimum* supporting chunk (often 1), not everything pulled.
> A diagnostic, not a target.

---

## Ragas (LLM-judged, judge = `gpt-4o`)

| Metric | Value |
|---|---:|
| faithfulness | 0.906 |
| answer relevancy | 0.850 |
| context precision (w/ reference) | 0.850 |
| context recall | 0.969 |
| factual correctness (F1) | 0.459 |

> **faithfulness 0.906 / context recall 0.969** — answers stay grounded in retrieved context and
> the context contains the needed facts. **factual correctness 0.459 is deflated**, not a real
> quality signal: it F1-matches short gold answers against verbose grounded replies, so extra
> (correct) detail is scored as "wrong". Flagged, not trusted blindly.

---

## Regression guard

```bash
make eval-baseline    # freeze a good run -> results/baseline.json
make eval             # every change -> results/run-<timestamp>.json
make eval-compare     # diff baseline vs newest; exits 1 on any F1 drop
```

`eval-compare` anchors on the **deterministic delta F1** (chat metrics are LLM-judged and not a
stable regression anchor), so a threshold change that lowers delta quality is caught before it
ships.
