"""Eval metrics.

Delta: match predicted DeltaItems <-> gold entries by (change_type, kind, key)
-> Precision/Recall/F1 (overall + per change_type/kind) + false-positive rate.
Chat: citation P/R/F1, retrieval recall@k, refusal accuracy, chunk utilization.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


def predicted_pairs(items) -> list[tuple]:
    """(change_type, kind, key) triple for each predicted DeltaItem."""
    return [(i.change_type, i.kind, i.evidence.get("key", "")) for i in items]


def gold_pairs(gold: list[dict]) -> list[tuple]:
    return [(g["change_type"], g["kind"], g["key"]) for g in gold]


@dataclass
class Score:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn,
                "precision": round(self.precision, 3), "recall": round(self.recall, 3),
                "f1": round(self.f1, 3)}


def score_delta(items, gold: list[dict], scope_kinds: set[str] | None = None) -> dict:
    """Score predicted delta vs gold.

    scope_kinds: if set, only these kinds are scored (predicted items of other
    kinds are ignored, not counted as FP) — used when gold labels only a subset
    of kinds (e.g. sister-units labels title_block only).
    """
    pred = predicted_pairs(items)
    gold_t = gold_pairs(gold)
    if scope_kinds is not None:
        pred = [t for t in pred if t[1] in scope_kinds]
        gold_t = [t for t in gold_t if t[1] in scope_kinds]

    pred_counts, gold_counts = _count(pred), _count(gold_t)
    tp = sum(min(pred_counts[t], gold_counts[t]) for t in set(pred_counts) & set(gold_counts))
    overall = Score(tp, len(pred) - tp, len(gold_t) - tp)
    return {
        "overall": overall.as_dict(),
        "by_change_type": {k: v.as_dict() for k, v in _breakdown(pred, gold_t, 0).items()},
        "by_kind": {k: v.as_dict() for k, v in _breakdown(pred, gold_t, 1).items()},
        "n_predicted": len(pred), "n_gold": len(gold_t),
    }


def false_positive_rate(items, scope_kinds: set[str] | None = None) -> dict:
    """For an ideal-empty case (scanned twin of same doc): every predicted change
    is a false positive. Reports count + rate over a chosen scope."""
    pred = [i for i in items if scope_kinds is None or i.kind in scope_kinds]
    return {"false_positives": len(pred),
            "by_kind": dict(_count(i.kind for i in pred))}


def _count(items) -> dict:
    counts = defaultdict(int)
    for item in items:
        counts[item] += 1
    return counts


# --- chat metrics ---------------------------------------------------------

def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn}


def citation_prf(cited_keys, gold_keys) -> dict:
    """Precision/recall/F1 of the answer's cited keys vs gold citation keys."""
    tp = fp = fn = 0
    for cited, gold in zip(cited_keys, gold_keys):
        c, g = set(cited), set(gold)
        tp += len(c & g); fp += len(c - g); fn += len(g - c)
    return _prf(tp, fp, fn)


def retrieval_recall(retrieved_keys, gold_keys) -> dict:
    """Recall@k: fraction of answerable cases whose gold key appeared in retrieval."""
    hits = total = 0
    for retrieved, gold in zip(retrieved_keys, gold_keys):
        if not gold:
            continue
        total += 1
        hits += 1 if (set(gold) & set(retrieved)) else 0
    return {"recall_at_k": round(hits / total, 3) if total else 0.0, "hits": hits, "n": total}


def refusal_classification(refused, answerable) -> dict:
    """Answered-vs-refused decision scored against answerable-vs-unanswerable.
    Catches BOTH failure modes: FP = answered an unanswerable (hallucination),
    FN = refused an answerable (false refusal)."""
    tp = tn = fp = fn = 0
    for ref, ans in zip(refused, answerable):
        should_refuse = not ans
        if should_refuse and ref:
            tn += 1
        elif should_refuse and not ref:
            fp += 1   # answered something unanswerable
        elif not should_refuse and ref:
            fn += 1   # false refusal
        else:
            tp += 1
    acc = (tp + tn) / len(refused) if refused else 0.0
    # precision/recall on the "refuse" class
    ref_tp, ref_fp, ref_fn = tn, fn, fp
    return {"accuracy": round(acc, 3), **{f"refuse_{k}": v for k, v in _prf(ref_tp, ref_fp, ref_fn).items()}}


def utilization(n_retrieved, n_cited) -> dict:
    """How many retrieved chunks the answer actually used (cited)."""
    tot_r, tot_c = sum(n_retrieved), sum(n_cited)
    return {"avg_retrieved": round(tot_r / len(n_retrieved), 2) if n_retrieved else 0,
            "avg_cited": round(tot_c / len(n_cited), 2) if n_cited else 0,
            "utilization": round(tot_c / tot_r, 3) if tot_r else 0.0}


def _breakdown(pred, gold, field_idx: int) -> dict:
    """Per-value (change_type or kind) P/R/F1 breakdown."""
    out = {}
    for value in sorted({t[field_idx] for t in pred} | {t[field_idx] for t in gold}):
        p_sub = [t for t in pred if t[field_idx] == value]
        g_sub = [t for t in gold if t[field_idx] == value]
        pc, gc = _count(p_sub), _count(g_sub)
        tp = sum(min(pc[t], gc[t]) for t in set(pc) & set(gc))
        out[value] = Score(tp, len(p_sub) - tp, len(g_sub) - tp)
    return out
