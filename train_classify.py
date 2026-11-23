"""Train a YOLOv8-cls fractured / non_fractured classifier and tune its decision threshold on VAL only.

Usage: python train_classify.py --model yolov8s-cls.pt [--imgsz 512 --epochs 100 --patience 20 --batch 16]
- Builds datasets/cls_os/ with only train (fractured oversampled 3x) and val. No test folder, so test
  images cannot be used for training, early stopping or threshold tuning.
- Images listed in damaged_images.txt are excluded (they are already deleted; safety net).
- After training: P(fractured) on every val image -> highest threshold with val recall >= 0.85.
- Writes runs/classify/<name>/val_threshold.json incl. per-image val probabilities. Never touches test.
- Resumable: an unfinished run continues from its last.pt (YOLO(last.pt).train(resume=True)); a finished run
  skips training and only re-runs the (VAL-only) threshold tuning. A STOP file stops safely after an epoch.
- Exit codes: 3 = CUDA out-of-memory (caller halves the batch and retries), 4 = stopped via STOP file.
"""
import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

from clf import Classifier, metrics, split_files, tune
from resume_util import EXIT_OOM, EXIT_STOPPED, StopRequested, add_stop_callback, ckpt_epoch, ckpt_status, is_oom

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "datasets" / "cls"
OS_DIR = ROOT / "datasets" / "cls_os"
DAMAGED = ROOT / "damaged_images.txt"
OVERSAMPLE = 3
TARGET_RECALL = 0.85


def damaged_images() -> set[str]:
    if not DAMAGED.exists():
        raise SystemExit("Run scan_damaged.py first.")
    out = set()
    for line in DAMAGED.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 4 and parts[0] in ("train", "val", "test"):
            out.add(parts[2])
    return out


def build_oversampled(bad: set[str]):
    if OS_DIR.exists():
        shutil.rmtree(OS_DIR)
    counts = {}
    for split in ("train", "val"):
        for cname in ("fractured", "non_fractured"):
            dst = OS_DIR / split / cname
            dst.mkdir(parents=True)
            n = 0
            for p in sorted((SRC / split / cname).glob("*.jpg")):
                if p.name in bad:
                    continue
                reps = OVERSAMPLE if (split == "train" and cname == "fractured") else 1
                for k in range(reps):
                    shutil.copy2(p, dst / (p.name if k == 0 else f"{p.stem}_dup{k}{p.suffix}"))
                    n += 1
            counts[f"{split}/{cname}"] = n
    print("cls_os image counts:", counts)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-cls.pt")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=None, help="default 16 (on resume: keep the checkpoint's batch)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--skip-build", action="store_true", help="reuse existing datasets/cls_os")
    ap.add_argument("--tune-only", action="store_true", help="skip training; tune threshold from existing best.pt")
    a = ap.parse_args()
    name = a.name or f"fracatlas_{Path(a.model).stem.replace('-', '_')}_{a.imgsz}"
    run_dir = ROOT / "runs" / "classify" / name
    last = run_dir / "weights" / "last.pt"
    status = ckpt_status(last)

    counts = None
    if (not a.skip_build and not a.tune_only and status == "none") or not OS_DIR.exists():
        counts = build_oversampled(damaged_images())
    assert not (OS_DIR / "test").exists(), "cls_os must not contain a test split"

    if status == "finished" and not a.tune_only:
        print(f"{name}: training already finished -> only re-running VAL threshold tuning")
    elif not a.tune_only:
        if status == "resumable":
            print(f"{name}: resuming from {last} (epoch {ckpt_epoch(last)} done)")
            model = add_stop_callback(YOLO(str(last)))
            train_kw = dict(resume=True, **({"batch": a.batch} if a.batch else {}))
        else:
            model = add_stop_callback(YOLO(a.model))
            train_kw = dict(data=str(OS_DIR), epochs=a.epochs, patience=a.patience, imgsz=a.imgsz,
                            batch=a.batch or 16, workers=4, device=0, seed=42,
                            project=str(ROOT / "runs" / "classify"), name=name, exist_ok=True)
        try:
            model.train(**train_kw)
        except StopRequested:
            sys.exit(EXIT_STOPPED)
        except Exception as e:  # torch.OutOfMemoryError / torch.AcceleratorError / RuntimeError
            if is_oom(e):
                print(f"CUDA OUT OF MEMORY at batch {a.batch}: {e}")
                sys.exit(EXIT_OOM)
            raise
        del model  # free the trainer's GPU memory before threshold tuning
        gc.collect()
        torch.cuda.empty_cache()

    # ---- threshold tuning on VAL only ----
    weights = run_dir / "weights" / "best.pt"
    files, y = split_files(OS_DIR, "val")
    p = Classifier([{"weights": weights, "imgsz": a.imgsz}]).probs(files)
    argmax = metrics(y, p, 0.5)
    chosen = tune(y, p, TARGET_RECALL)

    args_yaml = run_dir / "args.yaml"  # batch actually used (may differ after OOM halving / resume)
    used_batch = next((int(l.split(":", 1)[1]) for l in args_yaml.read_text().splitlines()
                       if l.startswith("batch:")), a.batch) if args_yaml.exists() else a.batch
    result = dict(name=name, model=a.model, imgsz=a.imgsz, epochs=a.epochs, patience=a.patience, batch=used_batch,
                  members=[{"weights": str(weights), "imgsz": a.imgsz}],
                  target_recall=TARGET_RECALL, val_argmax=argmax, val_tuned=chosen, dataset_counts=counts,
                  val_probs={f.name: q for f, q in zip(files, p)},
                  val_labels={f.name: bool(t) for f, t in zip(files, y)})
    (run_dir / "val_threshold.json").write_text(json.dumps(result, indent=2))

    print(f"\n=== VAL results for {name} (positive = fractured) ===")
    for label, m in (("argmax (0.50)", argmax), (f"tuned ({chosen['threshold']:.4f})", chosen)):
        print(f"{label:<18} acc={m['accuracy']:.4f} recall={m['recall']:.4f} "
              f"precision={m['precision']:.4f} f1={m['f1']:.4f}  TP={m['tp']} FN={m['fn']} FP={m['fp']} TN={m['tn']}")
    print(f"saved: {run_dir / 'val_threshold.json'}")


if __name__ == "__main__":
    main()
