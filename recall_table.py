"""VAL accuracy vs fractured recall at thresholds that give >= 85%, 80% and 75% recall (VAL only).

Usage: python recall_table.py run1 [run2 ...]
Uses saved val probabilities; prints a table and writes analysis/val_recall_tradeoff.md.
"""
import json
import sys
from pathlib import Path

from clf import tune

ROOT = Path(__file__).resolve().parent
TARGETS = (0.85, 0.80, 0.75)


def main():
    L = ["| Run | Target recall | Threshold | Val accuracy | Val recall | Val precision | Val F1 | FN | FP |",
         "|---|---|---|---|---|---|---|---|---|"]
    for run in sys.argv[1:]:
        j = json.loads((ROOT / "runs" / "classify" / run / "val_threshold.json").read_text())
        names = sorted(j["val_probs"])
        y = [j["val_labels"][n] for n in names]
        p = [j["val_probs"][n] for n in names]
        for t in TARGETS:
            m = tune(y, p, t)
            L.append(f"| {run} | {t:.2f} | {m['threshold']:.4f} | {m['accuracy']:.3f} | {m['recall']:.3f} | "
                     f"{m['precision']:.3f} | {m['f1']:.3f} | {m['fn']} | {m['fp']} |")
    text = "\n".join(L)
    (ROOT / "analysis").mkdir(exist_ok=True)
    (ROOT / "analysis" / "val_recall_tradeoff.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
