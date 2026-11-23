"""Choose between detection runs using VAL mAP50 only (on the current, cleaned val set).

Usage: python select_detect.py fracatlas_yolov8n detect_clean
Writes runs/detect/detect_selection.json. Never touches the test set.
"""
import json
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "det" / "data.yaml"


def main():
    runs = sys.argv[1:]
    res = {}
    for run in runs:
        m = YOLO(str(ROOT / "runs" / "detect" / run / "weights" / "best.pt")).val(
            data=str(DATA), split="val", imgsz=640, batch=16, device=0,
            project=str(ROOT / "runs" / "detect"), name=f"{run}_val_clean", exist_ok=True)
        res[run] = dict(map50=m.box.map50, map50_95=m.box.map, precision=m.box.mp, recall=m.box.mr)
    chosen = max(res, key=lambda r: res[r]["map50"])
    out = dict(criterion="val mAP50 on cleaned val set", val=res, chosen=chosen)
    (ROOT / "runs" / "detect" / "detect_selection.json").write_text(json.dumps(out, indent=2))
    print("\n=== Detection selection (VAL, cleaned) ===")
    for r, v in res.items():
        print(f"{r:<22} mAP50 {v['map50']:.4f}  mAP50-95 {v['map50_95']:.4f}  P {v['precision']:.4f}  R {v['recall']:.4f}"
              + ("   <- chosen" if r == chosen else ""))


if __name__ == "__main__":
    main()
