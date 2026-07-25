"""Regression compare — `make eval-compare`.

Diffs baseline.json vs latest.json (or two named runs) and prints a table of
metric deltas, flagging any drop. Proves the harness can catch a regression — a
change that lowers F1 shows up red. `make eval-baseline` freezes the reference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.config import get_settings


def _newest_run(results_dir: Path, exclude=("baseline",)):
    """Most recently written result file, skipping the frozen baseline."""
    runs = [p for p in results_dir.glob("*.json") if p.stem not in exclude]
    runs.sort(key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def _latest_two(results_dir: Path):
    runs = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if len(runs) < 2:
        return None
    return runs[-2], runs[-1]


def _f1(case: dict):
    if isinstance(case, dict) and "overall" in case:
        return case["overall"]["f1"]
    if isinstance(case, dict) and "false_positives" in case:
        return -case["false_positives"]  # fewer FPs is better; show as higher
    return None


def main() -> None:
    results_dir = Path(get_settings().eval.results_dir)
    if len(sys.argv) > 2:                       # explicit: eval-compare <prev> <cur>
        prev, cur = results_dir / f"{sys.argv[1]}.json", results_dir / f"{sys.argv[2]}.json"
    else:                                       # default: frozen baseline vs newest run
        prev, cur = results_dir / "baseline.json", _newest_run(results_dir)
        if not prev.exists() or cur is None:
            pair = _latest_two(results_dir)     # fall back to two most recent by mtime
            if not pair:
                print("need a baseline.json + at least one run. "
                      "run `make eval-baseline` then `make eval`.")
                return
            prev, cur = pair

    # deterministic-F1 cases only; chat metrics are LLM-judged -> not a stable anchor
    prev_data, cur_data = json.loads(prev.read_text()), json.loads(cur.read_text())
    print(f"\nregression: {prev.name} -> {cur.name}")
    print("=" * 60)
    regressed = False
    for case in sorted(set(prev_data) | set(cur_data)):
        f1_prev, f1_cur = _f1(prev_data.get(case, {})), _f1(cur_data.get(case, {}))
        if f1_prev is None or f1_cur is None:
            print(f"{case:<20} (n/a)")
            continue
        delta = f1_cur - f1_prev
        flag = "  <== REGRESSION" if delta < -1e-9 else ("  (improved)" if delta > 1e-9 else "")
        regressed = regressed or delta < -1e-9
        print(f"{case:<20} {f1_prev:+.3f} -> {f1_cur:+.3f}  (Δ {delta:+.3f}){flag}")
    print("=" * 60)
    print("RESULT:", "REGRESSION DETECTED" if regressed else "no regressions")
    sys.exit(1 if regressed else 0)


if __name__ == "__main__":
    main()
