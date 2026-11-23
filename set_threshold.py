"""Re-tune a classifier run's decision threshold on VAL for a different fractured-recall target.

Usage: python set_threshold.py fracatlas_yolov8m_cls_512 0.80
Used after no candidate reached 90% val accuracy at recall 0.85 and the user chose, from
analysis/val_recall_tradeoff.md, yolov8m-cls at 80% recall. Uses the saved VAL probabilities only
(no model inference, no test data). The original 0.85 result is kept as "val_tuned_085".
"""
import json
import sys
from pathlib import Path

from clf import tune

ROOT = Path(__file__).resolve().parent


def main():
    run, target = sys.argv[1], float(sys.argv[2])
    p = ROOT / "runs" / "classify" / run / "val_threshold.json"
    j = json.loads(p.read_text())
    names = list(j["val_probs"])
    y = [j["val_labels"][n] for n in names]
    q = [j["val_probs"][n] for n in names]
    j.setdefault("val_tuned_085", j["val_tuned"])
    j["val_tuned"] = tune(y, q, target)
    j["target_recall"] = target
    j["note"] = (f"User chose target recall {target:.2f} after 90% val accuracy was not reached at 0.85 "
                 "(see analysis/val_recall_tradeoff.md).")
    p.write_text(json.dumps(j, indent=2))
    t = j["val_tuned"]
    print(f"{run}: threshold {t['threshold']:.4f} -> val acc {t['accuracy']:.4f}, recall {t['recall']:.4f}, "
          f"precision {t['precision']:.4f}, F1 {t['f1']:.4f}")


if __name__ == "__main__":
    main()
