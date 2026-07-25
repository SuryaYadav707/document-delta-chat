"""Eval harness entrypoint — `make eval`.

Runs each delta case, scores predicted vs gold, prints a scorecard, and writes
eval/results/<run>.json for regression comparison. Deterministic: uses the
delta engine with use_llm=False so runs are byte-comparable across changes.

Chat metrics (correctness/groundedness) are added once the chat layer exists.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from src.config import get_settings
from src.delta.engine import compute_delta
from eval import metrics
from eval.datasets import scanned_noise, sister_units, synthetic_revision


def _run_delta(base, rev, cid):
    return compute_delta(base, rev, get_settings().delta, comparison_id=cid, use_llm=False)


def run_chat() -> dict:
    """Run the chat Q&A set: route->retrieve->answer per question, then score
    citation P/R/F1, retrieval recall@k, refusal accuracy, utilization, and
    judged answer accuracy."""
    import os

    from src.chat.answer import answer as answer_fn
    from src.chat.retrieve import Retriever
    from src.chat.router import route
    from src.ingest.resolver import PIDResolver
    from src.services.comparison import ComparisonService
    from eval import ragas_eval
    from eval.datasets import chat_qa

    if not os.environ.get("OPENAI_API_KEY"):
        return {"skipped": "no OPENAI_API_KEY"}

    base, rev, QA = chat_qa.build_case()
    cfg = get_settings()
    cmp = ComparisonService().create(base, rev, use_llm=False)   # ingest + delta + index
    cid = cmp.comparison_id
    fam_a = PIDResolver().resolve(base).doc_family
    fam_b = PIDResolver().resolve(rev).doc_family
    retriever = Retriever(cfg)

    cited, gold, retrieved, n_ret, n_cit, refused, answerable, ragas_samples = [], [], [], [], [], [], [], []
    for qa in QA:
        rd = route(qa["question"], base, rev, cfg, fam_a, fam_b)
        chunks = retriever.retrieve(qa["question"], cid, rd)
        ans = answer_fn(qa["question"], chunks, cfg)
        cited.append([c.key for c in ans.citations])
        gold.append(qa["gold_citation_keys"])
        retrieved.append([c.meta.get("key") for c in chunks])
        n_ret.append(len(chunks)); n_cit.append(len(ans.citations))
        refused.append(not ans.grounded); answerable.append(qa["answerable"])
        if qa["answerable"]:
            ragas_samples.append({"question": qa["question"], "answer": ans.text,
                                  "contexts": [c.text for c in chunks], "gold": qa["gold_answer"]})

    return {
        "citation": metrics.citation_prf(cited, gold),
        "retrieval": metrics.retrieval_recall(retrieved, gold),
        "refusal": metrics.refusal_classification(refused, answerable),
        "utilization": metrics.utilization(n_ret, n_cit),
        "ragas": ragas_eval.run_ragas(ragas_samples, cfg),   # judge = gpt-4o (not the answer model)
    }


def run() -> dict:
    results = {}

    # 1) synthetic-revision — exact GT
    base, rev, gold = synthetic_revision.build_case()
    d = _run_delta(base, rev, "synthetic")
    results["synthetic-revision"] = metrics.score_delta(d.items, gold)

    # 2) sister-units — title-block scope
    base, rev, gold, scope = sister_units.build_case()
    d = _run_delta(base, rev, "sisters")
    results["sister-units"] = metrics.score_delta(d.items, gold, scope_kinds=scope)

    # 3) scanned-noise — ideal empty; measures OCR false positives (optional)
    case = scanned_noise.build_case()
    if case is None:
        results["scanned-noise"] = {"skipped": "run `python -m eval.datasets.scanned_noise` to prepare (OCR)"}
    else:
        base, rev = case
        d = _run_delta(base, rev, "scanned_noise")
        results["scanned-noise"] = metrics.false_positive_rate(d.items)

    # 4) chat Q&A (needs API key + indexing)
    try:
        results["chat"] = run_chat()
    except Exception as e:
        results["chat"] = {"error": str(e)[:150]}

    return results


def _fmt_chat(c: dict) -> list[str]:
    if "skipped" in c or "error" in c:
        return [f"chat                 {c.get('skipped') or c.get('error')}"]
    cit, ret, ref, u = c["citation"], c["retrieval"], c["refusal"], c["utilization"]
    lines = [
        "chat eval",
        "-" * 60,
        "  deterministic:",
        f"    citation  P={cit['precision']} R={cit['recall']} F1={cit['f1']}",
        f"    retrieval recall@k  {ret['recall_at_k']}  ({ret['hits']}/{ret['n']})",
        f"    refusal   acc={ref['accuracy']}  refuse-F1={ref['refuse_f1']}",
        f"    chunk use  avg_retrieved={u['avg_retrieved']} avg_cited={u['avg_cited']} utilization={u['utilization']}",
        "  ragas (judge=gpt-4o):",
    ]
    rg = c.get("ragas", {})
    if "skipped" in rg or "error" in rg:
        lines.append(f"    {rg.get('skipped') or rg.get('error')}")
    else:
        lines += [f"    {k} = {v}" for k, v in rg.items()]
    return lines


def _fmt(results: dict) -> str:
    lines = ["", "delta eval scorecard", "=" * 60]
    for case, r in results.items():
        if case == "chat":
            continue
        if "skipped" in r:
            lines.append(f"{case:<20} SKIPPED — {r['skipped']}")
        elif "false_positives" in r:
            lines.append(f"{case:<20} false_positives={r['false_positives']}  by_kind={r['by_kind']}")
        else:
            o = r["overall"]
            lines.append(f"{case:<20} P={o['precision']:.2f} R={o['recall']:.2f} "
                         f"F1={o['f1']:.2f}  (tp={o['tp']} fp={o['fp']} fn={o['fn']}, "
                         f"gold={r['n_gold']})")
    lines.append("=" * 60)
    if "chat" in results:
        lines += ["", *_fmt_chat(results["chat"]), "=" * 60]
    return "\n".join(lines)


def main() -> None:
    results = run()
    print(_fmt(results))
    out_dir = Path(get_settings().eval.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # run id: a caller label (e.g. `baseline`) or a timestamp so each run is named
    # for WHEN it ran — `eval-compare` diffs baseline vs the newest such run.
    run_id = sys.argv[1] if len(sys.argv) > 1 else time.strftime("run-%Y-%m-%d_%H-%M-%S")
    (out_dir / f"{run_id}.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_dir}/{run_id}.json")


if __name__ == "__main__":
    main()
