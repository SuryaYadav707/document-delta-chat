"""Chat Q&A eval set — base=Lift (26-KA-901), revised=Export (26-KA-902).

Every gold_answer is verified against the extracted documents/delta (not against
what the chatbot happens to say). Deliberately balanced + honest:
  - all query types: change / content-base / content-revised / structured / compare
  - 5 must-refuse (unanswerable) so the refusal classifier has both classes
  - hard cases included on purpose (vessel-trim that once false-refused, a DELETED
    note, a noisy setpoint) — NOT cherry-picked to pass.

gold_citation_keys use the semantic chunk `key` (field:X / note:N / tag:X /
dim:instrument:limit), stable across runs (unlike region_ids).
"""
from __future__ import annotations

BASE = "data/samples/lift-gas-compressor.pdf"       # 26-KA-901 (lift)
REVISED = "data/samples/export-gas-compressor.pdf"   # 26-KA-902 (export)

QA = [
    # --- change queries (delta) ---
    {"id": "chg-duty", "category": "change", "answerable": True,
     "question": "Did the compressor duty change, and to what value?",
     "gold_answer": "Yes — duty changed from 776 kW (base) to 1835 kW (revised).",
     "gold_citation_keys": ["field:DUTY"]},
    {"id": "chg-flow", "category": "change", "answerable": True,
     "question": "Did the flow rate change?",
     "gold_answer": "Yes — flow rate changed from 19057 kg/h to 62809 kg/h.",
     "gold_citation_keys": ["field:FLOW RATE"]},
    {"id": "chg-oppress", "category": "change", "answerable": True,
     "question": "What changed in the discharge/suction operating pressure?",
     "gold_answer": "Discharge operating pressure changed from 229 to 199 barg; suction stayed 108.5.",
     "gold_citation_keys": ["field:DISCHARGE / SUCTION OP. PRESS. (MAX)"]},
    {"id": "chg-tag", "category": "change", "answerable": True,
     "question": "Was the equipment tag number changed?",
     "gold_answer": "Yes — tag number changed from 26-KA-901 to 26-KA-902.",
     "gold_citation_keys": ["field:TAG NUMBER"]},
    {"id": "chg-vesseltrim", "category": "change", "answerable": True,
     "question": "Did the vessel trim change?",
     "gold_answer": "Yes — vessel trim changed from TT-26-9712-AS20-00 to TT-26-9711-AS20-00.",
     "gold_citation_keys": ["field:VESSEL TRIM"]},

    # --- content, base document ---
    {"id": "base-duty", "category": "content-base", "answerable": True,
     "question": "What is the duty of the base compressor?",
     "gold_answer": "776 kW.",
     "gold_citation_keys": ["field:DUTY"]},
    {"id": "base-service", "category": "content-base", "answerable": True,
     "question": "What is the service description of the base document?",
     "gold_answer": "3RD STAGE HP GAS LIFT COMPRESSOR.",
     "gold_citation_keys": ["field:SERVICE"]},
    {"id": "base-material", "category": "content-base", "answerable": True,
     "question": "What is the material of the base compressor?",
     "gold_answer": "LTCS (1.7218).",
     "gold_citation_keys": ["field:MATERIAL"]},

    # --- content, revised document ---
    {"id": "rev-tag", "category": "content-revised", "answerable": True,
     "question": "What is the tag number of the revised compressor?",
     "gold_answer": "26-KA-902.",
     "gold_citation_keys": ["field:TAG NUMBER"]},
    {"id": "rev-vesseltrim", "category": "content-revised", "answerable": True,
     "question": "What is the vessel trim of the revised document?",
     "gold_answer": "TT-26-9711-AS20-00.",
     "gold_citation_keys": ["field:VESSEL TRIM"]},

    # --- structured lookups ---
    {"id": "note1-base", "category": "structured", "answerable": True,
     "question": "What is note 1 about in the base document?",
     "gold_answer": "26-PDI-9054 HH initiates a pressurized compressor stop.",
     "gold_citation_keys": ["note:1"]},
    {"id": "note9-base", "category": "structured", "answerable": True,
     "question": "What does note 9 say in the base document?",
     "gold_answer": "Manual drain prior to each start-up; push button with permissive for start-up sequence.",
     "gold_citation_keys": ["note:9"]},
    {"id": "note6-deleted", "category": "structured", "answerable": True,
     "question": "What is note 6 about in the base document?",
     "gold_answer": "Note 6 is marked DELETED (no content).",
     "gold_citation_keys": ["note:6"]},
    {"id": "setpoint-9062", "category": "structured", "answerable": True,
     "question": "What is the HH (high-high) shutdown setpoint on instrument 9062 in the base document?",
     "gold_answer": "245.",
     "gold_citation_keys": ["dim:9062:HH"]},

    # --- compare across both ---
    {"id": "cmp-service", "category": "compare", "answerable": True,
     "question": "Compare the service description of both documents.",
     "gold_answer": "Base = 3RD STAGE HP GAS LIFT COMPRESSOR; revised = 3RD STAGE HP GAS EXPORT COMPRESSOR.",
     "gold_citation_keys": ["field:SERVICE"]},
    {"id": "cmp-optemp", "category": "compare", "answerable": True,
     "question": "What is the operating temperature in each document?",
     "gold_answer": "Base = 122-135 / 50 °C; revised = 77-109 / 50 °C.",
     "gold_citation_keys": ["field:DISCHARGE / SUCTION OP. TEMP."]},

    # --- must-refuse (unanswerable — not in the documents) ---
    {"id": "ref-warranty", "category": "refuse", "answerable": False,
     "question": "What is the warranty period of the compressor?",
     "gold_answer": "Not supported by the documents.", "gold_citation_keys": []},
    {"id": "ref-pm", "category": "refuse", "answerable": False,
     "question": "Who is the project manager for this drawing?",
     "gold_answer": "Not supported by the documents.", "gold_citation_keys": []},
    {"id": "ref-cost", "category": "refuse", "answerable": False,
     "question": "What is the total purchase cost of the compressor?",
     "gold_answer": "Not supported by the documents.", "gold_citation_keys": []},
    {"id": "ref-ambient", "category": "refuse", "answerable": False,
     "question": "What is the ambient air temperature at the installation site?",
     "gold_answer": "Not supported by the documents.", "gold_citation_keys": []},
    {"id": "ref-paint", "category": "refuse", "answerable": False,
     "question": "What is the exterior paint colour / coating specification?",
     "gold_answer": "Not supported by the documents.", "gold_citation_keys": []},
]


def build_case():
    """Return (base_pid, revised_pid, qa_list)."""
    return BASE, REVISED, QA
