# Eval datasets

Three labeled cases (ground truth lives here):

1. **synthetic-revision/** — Export PDF with a documented edit list (setpoint
   change, note add/delete, tag rename, duty change) re-exported as A′.
   `changes.json` = gold delta (deterministic, controllable). Primary regression case.
2. **sister-units/** — Lift vs Export P&IDs, hand-labeled salient deltas in
   `changes.json` (renumbered tags, setpoint diffs, balance-line-cooler on Lift only,
   duty/flow diffs). Realistic, noisier; labeled conservatively.
3. **scanned-noise/** — a rasterized+degraded twin of a native PDF. Gold delta ≈ empty;
   anything detected = OCR false positives → quantifies OCR-induced noise.

`qa.jsonl` — ~15–25 questions across pairs with gold answers + gold citation
targets (pid/page/region or delta item), including unanswerable questions to test refusal.
