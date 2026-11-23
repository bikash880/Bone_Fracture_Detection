"""Train YOLOv8 fracture detection (localization). VAL only - the test set is never touched here.

Usage: python train_detect.py [--model yolov8n.pt] [--name detect_clean] [--batch 16]
       python train_detect.py --model yolov8s.pt --name detect_clean_s --batch 8 --patience 20
Resumable: if runs/detect/<name>/weights/last.pt exists and the run is not finished, training continues
from it (YOLO(last.pt).train(resume=True)); a finished run is skipped. Create a file named STOP in the
project folder to stop safely after the current epoch.
Test evaluation happens once, for the VAL-selected model, in evaluate_test.py.
Exit codes: 3 = CUDA out-of-memory (caller halves the batch and retries), 4 = stopped via STOP file.
"""
import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

from resume_util import EXIT_OOM, EXIT_STOPPED, StopRequested, add_stop_callback, ckpt_epoch, ckpt_status, is_oom

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "det" / "data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")  # COCO-pretrained weights
    ap.add_argument("--name", default="fracatlas_yolov8n")
    ap.add_argument("--batch", type=int, default=None, help="default 16 (on resume: keep the checkpoint's batch)")
    ap.add_argument("--patience", type=int, default=25)
    a = ap.parse_args()
    run = ROOT / "runs" / "detect" / a.name
    last = run / "weights" / "last.pt"

    status = ckpt_status(last)
    try:
        if status == "finished":
            print(f"{a.name}: already finished -> skipping training")
        elif status == "resumable":
            print(f"{a.name}: resuming from {last} (epoch {ckpt_epoch(last)} done)")
            extra = {"batch": a.batch} if a.batch else {}
            add_stop_callback(YOLO(str(last))).train(resume=True, **extra)
        else:
            add_stop_callback(YOLO(a.model)).train(
                data=str(DATA), epochs=100, patience=a.patience, imgsz=640, batch=a.batch or 16,
                workers=4, device=0, seed=42, project=str(ROOT / "runs" / "detect"), name=a.name, exist_ok=True)
    except StopRequested:
        sys.exit(EXIT_STOPPED)
    except Exception as e:  # torch.OutOfMemoryError / torch.AcceleratorError / RuntimeError
        if is_oom(e):
            print(f"CUDA OUT OF MEMORY at batch {a.batch}: {e}")
            sys.exit(EXIT_OOM)
        raise
    print(f"done: {run / 'weights' / 'best.pt'} (no test evaluation)")


if __name__ == "__main__":
    main()
