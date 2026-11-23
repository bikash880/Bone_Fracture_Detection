"""FracAtlas web UI backend: serves the React app (ui/dist) and a prediction API using final_models/.

Run:  .venv\\Scripts\\python.exe ui_server.py      then open http://127.0.0.1:8000

API
  GET  /api/info                 final models, threshold, saved test metrics
  POST /api/predict  (file)      classifier verdict + detector boxes for one X-ray image
  GET  /api/examples             a few test images to try (inference only, nothing is tuned)
  GET  /api/examples/{name}      the image file
"""
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "final_models"
DIST = ROOT / "ui" / "dist"
TEST_DIR = ROOT / "datasets" / "cls" / "test"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
MIN_BOX_CONF = 0.05          # boxes above this are returned; the UI slider filters further (default 0.25)
MAX_UPLOAD = 25 * 1024 * 1024

cls_cfg = json.loads((FINAL / "classifier_threshold.json").read_text())
det_cfg = json.loads((FINAL / "detector_info.json").read_text())
THRESHOLD = float(cls_cfg["fractured_threshold"])
CLS_MEMBERS = [(YOLO(str(FINAL / m["weights"])), int(m["imgsz"])) for m in cls_cfg["members"]]
DETECTOR = YOLO(str(FINAL / det_cfg["weights"]))
DET_IMGSZ = int(det_cfg.get("imgsz", 640))
LOCK = threading.Lock()      # one GPU, one request at a time


def _saved_metrics():
    out = {}
    c = ROOT / "runs" / "classify" / cls_cfg["source_run"] / "test_results.json"
    d = ROOT / "runs" / "detect" / det_cfg["source_run"] / "test_results.json"
    if c.exists():
        t = json.loads(c.read_text())["test"]
        out["classification"] = {k: t[k] for k in ("accuracy", "recall", "precision", "f1", "tp", "fn", "fp", "tn")}
    if d.exists():
        t = json.loads(d.read_text())
        out["detection"] = {k: t[k] for k in ("map50", "map50_95", "precision", "recall")}
    return out


INFO = {
    "classifier": {"model": "YOLOv8m-cls", "imgsz": CLS_MEMBERS[0][1], "threshold": THRESHOLD,
                   "run": cls_cfg["source_run"]},
    "detector": {"model": "YOLOv8n", "imgsz": DET_IMGSZ, "run": det_cfg["source_run"], "default_conf": 0.25},
    "device": "GPU (CUDA)" if DEVICE == 0 else "CPU",
    "test_metrics": _saved_metrics(),
}

app = FastAPI(title="FracAtlas fracture detection")


@app.get("/api/info")
def info():
    return INFO


def _warmup():
    dummy = np.zeros((512, 512, 3), dtype=np.uint8)
    with LOCK:
        for m, sz in CLS_MEMBERS:
            m.predict(dummy, imgsz=sz, device=DEVICE, verbose=False)
        DETECTOR.predict(dummy, imgsz=DET_IMGSZ, device=DEVICE, verbose=False)


def analyse(img):
    t0 = time.perf_counter()
    with LOCK:
        probs = []
        for m, sz in CLS_MEMBERS:
            r = m.predict(img, imgsz=sz, device=DEVICE, verbose=False)[0]
            fi = [k for k, v in r.names.items() if v == "fractured"][0]
            probs.append(float(r.probs.data[fi]))
        d = DETECTOR.predict(img, imgsz=DET_IMGSZ, conf=MIN_BOX_CONF, device=DEVICE, verbose=False)[0]
    p = float(np.mean(probs))
    boxes = [{"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1), "conf": round(c, 4)}
             for (x1, y1, x2, y2), c in zip(d.boxes.xyxy.tolist(), d.boxes.conf.tolist())]
    boxes.sort(key=lambda b: -b["conf"])
    h, w = img.shape[:2]
    return {"fractured": p >= THRESHOLD, "probability": p, "threshold": THRESHOLD,
            "score_vs_threshold": p / THRESHOLD, "boxes": boxes, "width": w, "height": h,
            "ms": round((time.perf_counter() - t0) * 1000)}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Image is larger than 25 MB.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not read this file as an image (use JPG or PNG).")
    return {"filename": file.filename, **analyse(img)}


def _examples():
    out = []
    for cname, label in (("fractured", "fractured"), ("non_fractured", "normal")):
        fs = sorted((TEST_DIR / cname).glob("*.jpg"))
        for f in (fs[0], fs[len(fs) // 3], fs[2 * len(fs) // 3]) if len(fs) >= 3 else fs:
            out.append({"name": f.name, "true_label": label, "url": f"/api/examples/{f.name}"})
    return out


EXAMPLES = _examples() if TEST_DIR.exists() else []
EX_PATHS = {e["name"]: TEST_DIR / ("fractured" if e["true_label"] == "fractured" else "non_fractured") / e["name"]
            for e in EXAMPLES}


@app.get("/api/examples")
def examples():
    return EXAMPLES


@app.get("/api/examples/{name}")
def example_file(name: str):
    if name not in EX_PATHS:
        raise HTTPException(404, "Unknown example")
    return FileResponse(EX_PATHS[name], media_type="image/jpeg")


if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")
else:
    @app.get("/")
    def no_build():
        return {"message": "UI not built yet: run `npm install` and `npm run build` in the ui/ folder."}


if __name__ == "__main__":
    import uvicorn
    _warmup()
    print(f"Models loaded on {INFO['device']}. Open http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
