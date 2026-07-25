# Sample data provenance

Honesty note: the two provided PDFs are **sister units**, not literal revisions
of one drawing. Documented as-is; a synthetic true-revision pair is added for
controllable eval ground truth.

| Sample | Source | Nature | Use |
|---|---|---|---|
| `lift-gas-compressor.pdf` (26-KA-901) | provided | native PDF, real P&ID | demo comparison pair (base A) |
| `export-gas-compressor.pdf` (26-KA-902) | provided | native PDF, real P&ID | demo comparison pair (revised B) |
| `export-rev-a-prime.pdf` | **synthesized** from Export via a documented edit list | native PDF, true revision | eval ground truth (deterministic) |
| `export-scanned.pdf` | **synthesized**: rasterize + degrade Export | scanned PDF | 2nd format demo + OCR-noise eval |
| `sample.dxf` | small hand-made / open DXF | vector CAD | DWG adapter real-stub proof |

Sister pair differs meaningfully (renumbered instrument tags, changed setpoints,
balance-line-cooler subsystem present on Lift only, duty/flow differences) — a
realistic delta. The synthetic pair's edit list is committed alongside it as the
gold change set.

> No secrets or PII in any sample. Synthesis steps reproducible from scripts in
> the build phase.
