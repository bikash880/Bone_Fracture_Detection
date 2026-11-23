"""Average the VAL P(fractured) of two or more trained classifier runs and tune the ensemble threshold on VAL.

Usage: python ensemble.py fracatlas_yolov8s_cls_512 fracatlas_yolov8m_cls_512
Uses the per-image val probabilities saved by train_classify.py (no test images involved).
Writes runs/classify/ensemble__<a>__<b>/val_threshold.json in the same format as a single run.
"""
import json
import sys
from pathlib import Path

from clf import metrics, tune

ROOT = Path(__file__).resolve().parent
TARGET_RECALL = 0.85


def main():
    runs = sys.argv[1:]
    assert len(runs) >= 2, "need at least two runs"
    infos = [json.loads((ROOT / "runs" / "classify" / r / "val_threshold.json").read_text()) for r in runs]
    names = sorted(set.intersection(*(set(i["val_probs"]) for i in infos)))
    assert all(len(i["val_probs"]) == len(names) for i in infos), "runs were validated on different val sets"
    y = [infos[0]["val_labels"][n] for n in names]
    p = [sum(i["val_probs"][n] for i in infos) / len(infos) for n in names]

    argmax = metrics(y, p, 0.5)
    chosen = tune(y, p, TARGET_RECALL)
    name = "ensemble__" + "__".join(runs)
    out = ROOT / "runs" / "classify" / name
    out.mkdir(parents=True, exist_ok=True)
    result = dict(name=name, model="ensemble(" + " + ".join(i["model"] for i in infos) + ")",
                  imgsz="/".join(str(i["imgsz"]) for i in infos),
                  members=[m for i in infos for m in i["members"]],
                  target_recall=TARGET_RECALL, val_argmax=argmax, val_tuned=chosen,
                  val_probs=dict(zip(names, p)), val_labels=dict(zip(names, y)))
    (out / "val_threshold.json").write_text(json.dumps(result, indent=2))
    print(f"=== VAL results for {name} ===")
    for label, m in (("argmax (0.50)", argmax), (f"tuned ({chosen['threshold']:.4f})", chosen)):
        print(f"{label:<18} acc={m['accuracy']:.4f} recall={m['recall']:.4f} "
              f"precision={m['precision']:.4f} f1={m['f1']:.4f}  TP={m['tp']} FN={m['fn']} FP={m['fp']} TN={m['tn']}")


if __name__ == "__main__":
    main()
