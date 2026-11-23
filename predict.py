"""Run the final detection and classification models on images and save annotated results.

Usage: python predict.py img1.jpg [img2.jpg ...]
Uses final_models/ and the VAL-tuned classifier threshold (single model or averaged ensemble).
"""
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

from clf import Classifier

ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "final_models"
OUT_DIR = ROOT / "outputs"


def main():
    imgs = [Path(p) for p in sys.argv[1:]]
    if not imgs:
        raise SystemExit("usage: python predict.py img1.jpg [img2.jpg ...]")
    cfg = json.loads((FINAL / "classifier_threshold.json").read_text())
    thr = cfg["fractured_threshold"]
    clf = Classifier(cfg["members"], base=FINAL)
    det = YOLO(str(FINAL / "detect_best.pt"))
    OUT_DIR.mkdir(exist_ok=True)

    for img_path, p_fr in zip(imgs, clf.probs(imgs)):
        d = det.predict(str(img_path), imgsz=640, conf=0.25, device=0, verbose=False)[0]
        label = "fractured" if p_fr >= thr else "non_fractured"
        canvas = d.plot()
        text = f"Classifier: {label} (P_frac={p_fr:.2f}, thr={thr:.2f}) | Boxes: {len(d.boxes)}"
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(canvas, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        out = OUT_DIR / f"pred_{img_path.stem}.jpg"
        cv2.imwrite(str(out), canvas)

        print(f"\nImage      : {img_path.name}  (true folder: {img_path.parent.name})")
        print(f"Classifier : {label}  P(fractured)={p_fr:.3f}  threshold={thr:.3f}")
        for b in d.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            print(f"Detection  : fractured conf={float(b.conf):.3f} box=({x1},{y1},{x2},{y2})")
        if len(d.boxes) == 0:
            print("Detection  : no boxes above conf 0.25")
        print(f"Saved      : {out}")


if __name__ == "__main__":
    main()
