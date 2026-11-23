"""Shared classifier helpers: P(fractured) from one model or an averaged ensemble, metrics, threshold tuning.

A classifier config is {"members": [{"weights": path, "imgsz": int}, ...], "threshold": float}.
One member = single model; several members = average of their P(fractured).
"""
from pathlib import Path

import torch
from ultralytics import YOLO


class Classifier:
    def __init__(self, members, base: Path | None = None):
        self.members = []
        for m in members:
            w = Path(m["weights"])
            if base is not None and not w.is_absolute():
                w = base / w
            self.members.append((YOLO(str(w)), int(m["imgsz"])))

    def probs(self, paths, batch=8):
        paths = [str(p) for p in paths]
        if not paths:
            return []
        total = [0.0] * len(paths)
        for model, imgsz in self.members:
            total = [t + q for t, q in zip(total, self._member_probs(model, imgsz, paths, batch))]
        return [t / len(self.members) for t in total]

    CHUNK = 64  # paths per predict() call: one call with ~450 paths hits CUDA OOM in Ultralytics' warmup

    @classmethod
    def _member_probs(cls, model, imgsz, paths, batch):
        """P(fractured) for one model; on CUDA out-of-memory, halve the batch and retry (down to 1)."""
        while True:
            try:
                out, fi = [], None
                for i in range(0, len(paths), cls.CHUNK):
                    for r in model.predict(paths[i:i + cls.CHUNK], imgsz=imgsz, device=0, verbose=False,
                                           stream=True, batch=batch):
                        if fi is None:
                            fi = [k for k, v in r.names.items() if v == "fractured"][0]
                        out.append(float(r.probs.data[fi]))
                assert len(out) == len(paths)
                return out
            except Exception as e:  # torch.OutOfMemoryError / torch.AcceleratorError / RuntimeError
                if "out of memory" not in str(e).lower() or batch == 1:
                    raise
                torch.cuda.empty_cache()
                batch //= 2
                print(f"CUDA out of memory during prediction -> retrying with batch {batch}")


def chunked_predict(model, paths, chunk=64, **kw):
    """Yield model.predict results for many paths, CHUNK paths per call (avoids the warmup OOM)."""
    paths = [str(p) for p in paths]
    for i in range(0, len(paths), chunk):
        yield from model.predict(paths[i:i + chunk], stream=True, **kw)


def metrics(y_true, p, thr):
    tp = sum(1 for y, q in zip(y_true, p) if y and q >= thr)
    fn = sum(1 for y, q in zip(y_true, p) if y and q < thr)
    fp = sum(1 for y, q in zip(y_true, p) if not y and q >= thr)
    tn = sum(1 for y, q in zip(y_true, p) if not y and q < thr)
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(threshold=thr, accuracy=(tp + tn) / len(y_true), recall=rec, precision=prec, f1=f1,
                tp=tp, fn=fn, fp=fp, tn=tn)


def tune(y_true, p, target_recall):
    """Highest threshold whose recall >= target (max accuracy/precision under the recall constraint).
    Placed midway to the next lower score so the cut does not sit exactly on a sample."""
    cands = sorted(set(p), reverse=True)
    for i, thr in enumerate(cands):
        m = metrics(y_true, p, thr)
        if m["recall"] >= target_recall:
            if i + 1 < len(cands):
                mid = (thr + cands[i + 1]) / 2
                mm = metrics(y_true, p, mid)
                if mm["recall"] >= target_recall and mm["accuracy"] >= m["accuracy"]:
                    return mm
            return m
    return metrics(y_true, p, min(p))


def split_files(root: Path, split: str):
    files, y = [], []
    for cname in ("fractured", "non_fractured"):
        fs = sorted((root / split / cname).glob("*.jpg"))
        files += fs
        y += [cname == "fractured"] * len(fs)
    return files, y
