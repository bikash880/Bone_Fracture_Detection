"""Final, one-time TEST evaluation of the VAL-selected models. Run once, after all selection is done.

Usage: python evaluate_test.py --det-run detect_clean --cls-run fracatlas_yolov8s_cls_512
                               [--val-candidates fracatlas_yolov8m_cls_512]
- Detection: test mAP50 / mAP50-95 / precision / recall of the chosen detection run only.
- Classification: the chosen run's VAL-tuned threshold applied to the test set, once.
- --val-candidates only adds other classifier runs' saved VAL numbers to results.md (no test inference).
- Writes results.md (summary, data cleaning, val-based selection) and copies models to final_models/.
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from clf import Classifier, chunked_predict, metrics, split_files

ROOT = Path(__file__).resolve().parent
DET_DATA = ROOT / "datasets" / "det" / "data.yaml"
CLS_TEST = ROOT / "datasets" / "cls" / "test"
DAMAGED = ROOT / "damaged_images.txt"
DUPES = ROOT / "duplicates_removed.txt"
FINAL = ROOT / "final_models"
SPLITS = ("train", "val", "test")


def table_rows(path):
    return [l.split("\t") for l in path.read_text(encoding="utf-8").splitlines()
            if l.count("\t") >= 3 and l.split("\t")[0] in SPLITS]


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def box_stats(model, imgs, conf, iou_t=0.5):
    """Box-level recall / false alarms at a fixed confidence (greedy matching, highest confidence first)."""
    gt_n = found = n_pred = wrong = 0
    normal_imgs = normal_with_alarm = 0
    for p, r in zip(imgs, chunked_predict(model, imgs, imgsz=640, conf=conf, device=0, verbose=False, batch=8)):
        h, w = r.orig_shape
        lbl = ROOT / "datasets" / "det" / "labels" / "test" / f"{p.stem}.txt"
        gts = []
        for line in lbl.read_text().splitlines() if lbl.exists() else []:
            q = line.split()
            if len(q) == 5:
                _, cx, cy, bw, bh = map(float, q)
                gts.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
        used = [False] * len(gts)
        img_wrong = 0
        for b in sorted(r.boxes, key=lambda b: -float(b.conf)):
            box = b.xyxy[0].tolist()
            cand = [(_iou(box, g), j) for j, g in enumerate(gts) if not used[j]]
            best = max(cand, default=(0.0, None))
            if best[1] is not None and best[0] >= iou_t:
                used[best[1]] = True
            else:
                img_wrong += 1
            n_pred += 1
        gt_n += len(gts)
        found += sum(used)
        wrong += img_wrong
        if not gts:
            normal_imgs += 1
            normal_with_alarm += img_wrong > 0
    return dict(conf=conf, gt=gt_n, found=found, missed=gt_n - found, recall=found / gt_n if gt_n else 0.0,
                pred=n_pred, wrong=wrong, precision=(n_pred - wrong) / n_pred if n_pred else 0.0,
                normal_imgs=normal_imgs, normal_with_alarm=normal_with_alarm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-run", required=True)
    ap.add_argument("--cls-run", required=True)
    ap.add_argument("--val-candidates", nargs="*", default=[])
    a = ap.parse_args()
    det_dir = ROOT / "runs" / "detect" / a.det_run

    # ---- Detection test (chosen run only) ----
    det_model = YOLO(str(det_dir / "weights" / "best.pt"))
    dm = det_model.val(
        data=str(DET_DATA), split="test", imgsz=640, batch=16, device=0,
        project=str(ROOT / "runs" / "detect"), name=f"{a.det_run}_test", exist_ok=True)
    det = dict(map50=dm.box.map50, map50_95=dm.box.map, precision=dm.box.mp, recall=dm.box.mr)
    det_arch = next((l.split(":", 1)[1].strip() for l in (det_dir / "args.yaml").read_text().splitlines()
                     if l.startswith("model:")), "?")
    test_imgs = sorted((ROOT / "datasets" / "det" / "images" / "test").glob("*.jpg"))
    det["by_conf"] = {c: box_stats(det_model, test_imgs, c) for c in (0.25, 0.10)}

    # ---- Classification (chosen run or ensemble, VAL-tuned threshold) ----
    info = json.loads((ROOT / "runs" / "classify" / a.cls_run / "val_threshold.json").read_text())
    thr, imgsz = info["val_tuned"]["threshold"], info["imgsz"]
    clf = Classifier(info["members"])
    # train accuracy: original train images (no oversampled duplicates), same threshold
    tr_files, tr_y = split_files(ROOT / "datasets" / "cls", "train")
    train_m = metrics(tr_y, clf.probs(tr_files), thr)
    # test: single pass
    te_files, te_y = split_files(ROOT / "datasets" / "cls", "test")
    test = metrics(te_y, clf.probs(te_files), thr)
    tp, fn, fp, tn = test["tp"], test["fn"], test["fp"], test["tn"]
    n = tp + fn + fp + tn
    rec, prec, f1 = test["recall"], test["precision"], test["f1"]
    test["n"] = n
    info_out = {k: v for k, v in info.items() if k not in ("val_probs", "val_labels")}
    (ROOT / "runs" / "classify" / a.cls_run / "test_results.json").write_text(
        json.dumps({**info_out, "train": train_m, "test": test}, indent=2))
    (det_dir / "test_results.json").write_text(json.dumps(det, indent=2))

    # ---- final_models ----
    if FINAL.exists():
        shutil.rmtree(FINAL)
    FINAL.mkdir()
    shutil.copy2(det_dir / "weights" / "best.pt", FINAL / "detect_best.pt")
    members = []
    for m in info["members"]:
        src = Path(m["weights"])
        dst = f"classify_{src.parent.parent.name}_best.pt"  # e.g. classify_fracatlas_yolov8s_cls_512_best.pt
        shutil.copy2(src, FINAL / dst)
        members.append({"weights": dst, "imgsz": m["imgsz"]})
    (FINAL / "classifier_threshold.json").write_text(json.dumps(
        {"members": members, "fractured_threshold": thr, "source_run": a.cls_run,
         "note": "P(fractured) = mean over members; predict fractured when P >= threshold "
                 f"(tuned on VAL for recall >= {info['target_recall']:.2f})"}, indent=2))
    (FINAL / "detector_info.json").write_text(json.dumps(
        {"weights": "detect_best.pt", "source_run": a.det_run, "imgsz": 640}, indent=2))

    # ---- data cleaning facts ----
    drows = table_rows(DAMAGED)
    dmg = {(s, c): sum(r[0] == s and r[1] == c for r in drows) for s in SPLITS for c in ("fractured", "non_fractured")}
    dupes = table_rows(DUPES) if DUPES.exists() else []
    df = pd.read_csv(ROOT / "FracAtlas" / "dataset.csv").set_index("image_id")
    lbl = ROOT / "datasets" / "det" / "labels" / "test" / "IMG0003298.txt"
    img98_ok = (df.loc["IMG0003298.jpg", "fracture_count"] > 0 and (CLS_TEST / "fractured" / "IMG0003298.jpg").exists()
                and lbl.exists() and lbl.read_text().strip() != "")
    counts = {s: (len(list((ROOT / "datasets" / "cls" / s / "fractured").glob("*.jpg"))),
                  len(list((ROOT / "datasets" / "cls" / s / "non_fractured").glob("*.jpg")))) for s in SPLITS}

    v, va = info["val_tuned"], info["val_argmax"]
    L = ["# FracAtlas results", "",
         "## Summary (test set, evaluated once)", "",
         "| Metric | Value |", "|---|---|",
         f"| Detection model | {det_arch}, imgsz 640, run `{a.det_run}` |",
         f"| Detection test mAP50 | {det['map50']:.3f} |",
         f"| Detection test mAP50-95 | {det['map50_95']:.3f} |",
         f"| Detection test precision | {det['precision']:.3f} |",
         f"| Detection test recall | {det['recall']:.3f} |",
         f"| Classifier | {info['model']}, imgsz {imgsz}, threshold {thr:.4f} (val-tuned) |",
         f"| Classification train accuracy | {train_m['accuracy']:.3f} |",
         f"| Classification val accuracy | {v['accuracy']:.3f} |",
         f"| Classification test accuracy | {test['accuracy']:.3f} |",
         f"| Classification test fractured recall | {rec:.3f} |",
         f"| Classification test precision | {prec:.3f} |",
         f"| Classification test F1 | {f1:.3f} |",
         f"| Damaged images removed | {len(drows)} |", "",
         f"Classification test confusion matrix (n = {n}):", "",
         "| true \\ predicted | fractured | non_fractured |", "|---|---|---|",
         f"| fractured | {tp} | {fn} |", f"| non_fractured | {fp} | {tn} |", "",
         "Localization recall vs false alarms on test (box level, IoU >= 0.5):", "",
         "| Confidence | True boxes | Found | Missed | Box recall | Predicted boxes | Wrong boxes | Box precision | Normal images with a false box |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c, s in det["by_conf"].items():
        L.append(f"| {c:.2f} | {s['gt']} | {s['found']} | {s['missed']} | {s['recall']:.3f} | {s['pred']} | "
                 f"{s['wrong']} | {s['precision']:.3f} | {s['normal_with_alarm']} / {s['normal_imgs']} |")
    L += ["", "Classification by split (same val-tuned threshold; train = original images, no oversampled copies):", "",
         "| Split | Images | Accuracy | Fractured recall | Precision | F1 |", "|---|---|---|---|---|---|",
         f"| Train | {len(tr_y)} | {train_m['accuracy']:.3f} | {train_m['recall']:.3f} | {train_m['precision']:.3f} | {train_m['f1']:.3f} |",
         f"| Val | {v['tp'] + v['fn'] + v['fp'] + v['tn']} | {v['accuracy']:.3f} | {v['recall']:.3f} | {v['precision']:.3f} | {v['f1']:.3f} |",
         f"| Test | {n} | {test['accuracy']:.3f} | {rec:.3f} | {prec:.3f} | {f1:.3f} |", "",
         "> **Note:** val accuracy is slightly optimistic, because the model and the decision threshold were both "
         "chosen on the val set. **The test accuracy is the honest result**: the test set was used once, after "
         "every choice was fixed. Train accuracy is higher still, because the model learned from those images.", "",
         "## Data cleaning", "",
         "Damaged images (JPEGs truncated in the original FracAtlas zip; 24 of them cut off at exactly 16,384 bytes) "
         "were removed from every split of both datasets:", "",
         "| Split | Fractured | Non-fractured | Total |", "|---|---|---|---|"]
    for s in SPLITS:
        L.append(f"| {s} | {dmg[(s, 'fractured')]} | {dmg[(s, 'non_fractured')]} | "
                 f"{dmg[(s, 'fractured')] + dmg[(s, 'non_fractured')]} |")
    L += [f"| **all** | {sum(dmg[(s, 'fractured')] for s in SPLITS)} | "
          f"{sum(dmg[(s, 'non_fractured')] for s in SPLITS)} | **{len(drows)}** |", "",
          f"Cross-split duplicates: {len(dupes)} byte-identical images were found in two splits under different "
          "names; the copy in the later split was removed (train < val < test):", ""]
    L += [f"- `{r[2]}` ({r[1]}) removed from **{r[0]}**: {r[3]}" for r in dupes]
    L += ["",
          f"IMG0003298: {'verified fractured' if img98_ok else 'CHECK FAILED'} "
          "(dataset.csv fracture_count = 1, official test split, 1 YOLO box, in cls test/fractured). "
          "It was already labelled correctly, so no change was needed.", "",
          "Final image counts after cleaning (fractured / non-fractured): "
          + ", ".join(f"{s} {counts[s][0]} / {counts[s][1]}" for s in SPLITS) + ".", ""]

    L += ["## Model selection (VAL only)", ""]
    sel = ROOT / "runs" / "detect" / "detect_selection.json"
    if sel.exists():
        sj = json.loads(sel.read_text())
        L += ["Detection, by val mAP50 on the cleaned val set:", "",
              "| Run | Val mAP50 | Val mAP50-95 | Val precision | Val recall |", "|---|---|---|---|---|"]
        for r, m in sj["val"].items():
            L.append(f"| {r}{' (chosen)' if r == sj['chosen'] else ''} | {m['map50']:.3f} | {m['map50_95']:.3f} | "
                     f"{m['precision']:.3f} | {m['recall']:.3f} |")
        L.append("")
    L += ["Classification: candidates tuned on val for fractured recall >= 0.85 (s, m, s+m ensemble). None reached "
          "90% tuned val accuracy (best: m, 79.8%). The imgsz 640 / 150-epoch step was estimated at 4-6 h and "
          "skipped by the user. From the val trade-off table (`analysis/val_recall_tradeoff.md`) the user chose "
          f"yolov8m-cls with the threshold tuned on val for recall >= {info['target_recall']:.2f} "
          "(the chosen row below uses that threshold; the others use 0.85).", "",
          "| Run | Val acc (0.5) | Val acc (tuned) | Val recall (tuned) | Val precision (tuned) | Val F1 (tuned) | Threshold |",
          "|---|---|---|---|---|---|---|"]
    for run in [a.cls_run] + a.val_candidates:
        j = json.loads((ROOT / "runs" / "classify" / run / "val_threshold.json").read_text())
        t = j["val_tuned"]
        L.append(f"| {run}{' (chosen)' if run == a.cls_run else ''} | {j['val_argmax']['accuracy']:.3f} | "
                 f"{t['accuracy']:.3f} | {t['recall']:.3f} | {t['precision']:.3f} | {t['f1']:.3f} | {t['threshold']:.4f} |")
    tradeoff = ROOT / "analysis" / "val_recall_tradeoff.md"
    if tradeoff.exists():
        chosen_thr = f"{thr:.4f}"
        rows = []
        for line in tradeoff.read_text(encoding="utf-8").splitlines():
            if line.startswith("| " + a.cls_run + " |") and f"| {chosen_thr} |" in line:
                line = line.replace(f"| {a.cls_run} |", f"| **{a.cls_run} (chosen)** |", 1)
            rows.append(line)
        L += ["", "### Val accuracy vs fractured recall trade-off (deliberate threshold choice)", "",
              "Because no candidate reached 90% val accuracy at 85% recall, each candidate's val accuracy was "
              "computed at thresholds giving 85%, 80% and 75% fractured recall (val only; test never used). "
              f"The user chose {a.cls_run} at {info['target_recall']:.0%} recall (threshold {chosen_thr}): "
              "it gives up about 5 points of recall (4 more missed val fractures than at 85%) in exchange for "
              "about 6 points of accuracy and 31 fewer false alarms. The 75% option was closer to 90% accuracy "
              "but would miss 1 in 4 fractures.", ""] + rows
    L += ["", "## Notes", "",
          "- Fractured = fracture_count > 0. Fractured images use the official FracAtlas split; non-fractured "
          "images were split randomly (seed 42) in the same ~80/11/9 ratio.",
          "- Classifier training: fractured train images oversampled 3x (train only); "
          "the training data folder contained no test images.",
          "- Test images were never used for training, early stopping, threshold tuning or model selection.",
          "- Models: final_models/ (classifier threshold in classifier_threshold.json).", ""]
    (ROOT / "results.md").write_text("\n".join(L), encoding="utf-8")

    print(f"\n=== Detection TEST ({a.det_run}) ===")
    print(f"mAP50 {det['map50']:.4f} | mAP50-95 {det['map50_95']:.4f} | P {det['precision']:.4f} | R {det['recall']:.4f}")
    for c, s in det["by_conf"].items():
        print(f"conf {c:.2f}: box recall {s['recall']:.3f} ({s['found']}/{s['gt']}) | wrong boxes {s['wrong']} | "
              f"normal images with a false box {s['normal_with_alarm']}/{s['normal_imgs']}")
    print(f"\n=== Classification TEST ({a.cls_run}, threshold {thr:.4f}) ===")
    print(f"accuracy train {train_m['accuracy']:.4f} | val {v['accuracy']:.4f} | test {test['accuracy']:.4f}")
    print(f"test recall {rec:.4f} | precision {prec:.4f} | F1 {f1:.4f}")
    print(f"confusion: TP={tp} FN={fn} FP={fp} TN={tn}  (n={n})")
    print("wrote results.md and final_models/")


if __name__ == "__main__":
    main()
