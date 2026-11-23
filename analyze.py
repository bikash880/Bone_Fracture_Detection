"""Post-hoc analysis of the FINAL (already chosen) models on the test set. No tuning happens here.

Run after evaluate_test.py. Classification and localization are analysed separately.

CLASSIFICATION (classifier only, per image, fixed val-tuned threshold)
  confusion matrix, accuracy / recall / precision / F1, overall and per region;
  errors/classification/: 10 most confident missed fractures, 10 most confident false alarms.
LOCALIZATION (detector only, per box, IoU >= 0.5)
  mAP50 / recall / precision (Ultralytics), overall and per region, PR curve;
  box-level found/missed counts at conf >= 0.25 (greedy matching, highest confidence first);
  errors/localization/: 10 largest missed true boxes, 10 most confident wrong boxes (green = true, red = predicted).
REGIONS
  "overlapping" groups use the dataset.csv flags (an image can be in several groups);
  "single-region only" keeps images with exactly one of hand/leg/hip/shoulder and mixed = 0.
Writes/replaces the "## Analysis" section of results.md (plain-English summary is added afterwards).
"""
import json
import shutil
from pathlib import Path
from statistics import median

import cv2
import pandas as pd
from ultralytics import YOLO

from clf import Classifier, chunked_predict, split_files

ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "final_models"
DET = ROOT / "datasets" / "det"
CLS_TEST = ROOT / "datasets" / "cls" / "test"
OUT = ROOT / "analysis"
ERR = ROOT / "errors"
REGIONS = ["hand", "leg", "hip", "shoulder", "mixed"]
SINGLE = ["hand", "leg", "hip", "shoulder"]
IOU_T, CONF_T = 0.5, 0.25


def fmt(x, d=3):
    return "-" if x is None or x != x else f"{x:.{d}f}"


def prf(tp, fn, fp, tn=0):
    n = tp + fn + fp + tn
    rec = tp / (tp + fn) if tp + fn else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp else (0.0 if tp + fn else float("nan"))
    return dict(n=n, acc=(tp + tn) / n if n else float("nan"), recall=rec, precision=prec, f1=f1,
                tp=tp, fn=fn, fp=fp, tn=tn)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def gt_boxes(stem, w, h):
    p = DET / "labels" / "test" / f"{stem}.txt"
    out = []
    for line in p.read_text().splitlines() if p.exists() else []:
        parts = line.split()
        if len(parts) == 5:
            _, cx, cy, bw, bh = map(float, parts)
            out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


def region_groups(df_like):
    """name -> boolean mask; overlapping groups + single-region-only groups."""
    single_ok = (df_like[SINGLE].sum(axis=1) == 1) & (df_like["mixed"] == 0)
    g = {f"{r}": df_like[r] == 1 for r in REGIONS}
    g |= {f"{r} (single)": single_ok & (df_like[r] == 1) for r in SINGLE}
    return g


def banner(im, text):
    cv2.rectangle(im, (0, 0), (im.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(im, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def view_of(row):
    return "+".join(v for v in ("frontal", "lateral", "oblique") if row[v] == 1) or "none"


def main():
    OUT.mkdir(exist_ok=True)
    if ERR.exists():
        shutil.rmtree(ERR)
    for sub in ("classification/missed_fractures", "classification/false_alarms",
                "localization/missed_boxes", "localization/wrong_boxes"):
        (ERR / sub).mkdir(parents=True)

    df = pd.read_csv(ROOT / "FracAtlas" / "dataset.csv").set_index("image_id")
    cfg = json.loads((FINAL / "classifier_threshold.json").read_text())
    thr = cfg["fractured_threshold"]
    clf = Classifier(cfg["members"], base=FINAL)
    det = YOLO(str(FINAL / "detect_best.pt"))
    test_imgs = sorted((DET / "images" / "test").glob("*.jpg"))

    # ===================== CLASSIFICATION =====================
    files, y = split_files(ROOT / "datasets" / "cls", "test")
    recs = [dict(image=f.name, path=str(f), true=t, p=p, pred=p >= thr)
            for f, t, p in zip(files, y, clf.probs(files))]
    cp = pd.DataFrame(recs).set_index("image").join(df)
    cp["view"] = cp.apply(view_of, axis=1)

    def ccounts(d):
        return dict(tp=int((d.true & d.pred).sum()), fn=int((d.true & ~d.pred).sum()),
                    fp=int((~d.true & d.pred).sum()), tn=int((~d.true & ~d.pred).sum()))

    cls_all = prf(**ccounts(cp))
    cls_reg = {g: prf(**ccounts(cp[m])) for g, m in region_groups(cp).items()}

    def save_cls(d, folder):
        for i, (name, row) in enumerate(d.iterrows(), 1):
            im = cv2.imread(row.path)
            h, w = im.shape[:2]
            regs = ",".join(r for r in REGIONS if row[r] == 1)
            banner(im, f"true={'fractured' if row.true else 'normal'} P(frac)={row.p:.3f} thr={thr:.3f} | "
                       f"{regs} | {row.view} | hw={int(row.hardware)} | {w}x{h}")
            cv2.imwrite(str(ERR / "classification" / folder / f"{i:02d}_{Path(name).stem}_p{row.p:.3f}.jpg"), im)

    cls_fn = cp[cp.true & ~cp.pred].sort_values("p").head(10)
    cls_fp = cp[~cp.true & cp.pred].sort_values("p", ascending=False).head(10)
    save_cls(cls_fn, "missed_fractures")
    save_cls(cls_fp, "false_alarms")

    def cls_stats(d):
        if not len(d):
            return dict(n=0)
        px = [cv2.imread(p).shape[:2] for p in d.path]
        return dict(n=len(d), hardware=float(d.hardware.mean()),
                    mixed=float((d.mixed == 1).mean()),
                    median_mpix=median(h * w for h, w in px) / 1e6,
                    views=d.view.value_counts().to_dict(),
                    regions={r: int(d[r].sum()) for r in REGIONS if d[r].sum()})

    cls_patterns = {"Missed fractures (FN)": cls_stats(cp[cp.true & ~cp.pred]),
                    "Caught fractures (TP)": cls_stats(cp[cp.true & cp.pred]),
                    "False alarms (FP)": cls_stats(cp[~cp.true & cp.pred]),
                    "Correct normals (TN)": cls_stats(cp[~cp.true & ~cp.pred])}

    # ===================== LOCALIZATION =====================
    # Ultralytics mAP50 / P / R per region via image-list subsets
    meta = df.loc[[p.name for p in test_imgs]]
    groups = {"all": pd.Series(True, index=meta.index)} | region_groups(meta)
    loc_reg = {}
    for g, mask in groups.items():
        imgs = [p for p in test_imgs if mask.get(p.name, False)]
        n_fr = int(sum(df.loc[p.name, "fracture_count"] > 0 for p in imgs))
        if n_fr == 0:
            loc_reg[g] = dict(n=len(imgs), n_fractured=0)
            continue
        tag = g.replace(" ", "_").replace("(", "").replace(")", "")
        lst = OUT / f"test_{tag}.txt"
        lst.write_text("\n".join(p.as_posix() for p in imgs) + "\n")
        yml = OUT / f"test_{tag}.yaml"
        yml.write_text(f"path: {DET.as_posix()}\ntrain: images/train\nval: images/val\ntest: {lst.as_posix()}\n"
                       "nc: 1\nnames:\n  0: fractured\n")
        m = det.val(data=str(yml), split="test", imgsz=640, batch=16, device=0, plots=(g == "all"),
                    iou=0.7, project=str(OUT), name=f"det_test_{tag}", exist_ok=True, verbose=False)
        loc_reg[g] = dict(n=len(imgs), n_fractured=n_fr, map50=m.box.map50, precision=m.box.mp, recall=m.box.mr)
    pr_curve = None
    for cand in ("BoxPR_curve.png", "PR_curve.png"):
        if (OUT / "det_test_all" / cand).exists():
            pr_curve = OUT / "detection_PR_curve_test.png"
            shutil.copy2(OUT / "det_test_all" / cand, pr_curve)
            break

    # Box-level matching at conf >= CONF_T, IoU >= IOU_T
    gts, preds = [], []  # gts: (image, box, area_frac, matched); preds: (image, box, conf, matched, best_iou)
    for p, r in zip(test_imgs, chunked_predict(det, test_imgs, imgsz=640, conf=CONF_T, device=0,
                                               verbose=False, batch=8)):
        h, w = r.orig_shape
        g = gt_boxes(p.stem, w, h)
        pb = sorted(((b.xyxy[0].tolist(), float(b.conf)) for b in r.boxes), key=lambda x: -x[1])
        used = [False] * len(g)
        for box, conf in pb:
            ious = [iou(box, gb) for gb in g]
            best = max(range(len(g)), key=lambda j: ious[j] if not used[j] else -1, default=None)
            hit = best is not None and not used[best] and ious[best] >= IOU_T
            if hit:
                used[best] = True
            preds.append(dict(image=p.name, path=str(p), box=box, conf=conf, matched=hit,
                              best_iou=max(ious, default=0.0), on_normal=not g))
        for gb, u in zip(g, used):
            gts.append(dict(image=p.name, path=str(p), box=gb,
                            area=(gb[2] - gb[0]) * (gb[3] - gb[1]) / (w * h), matched=u))
    gdf = pd.DataFrame(gts)
    pdf = pd.DataFrame(preds, columns=["image", "path", "box", "conf", "matched", "best_iou", "on_normal"])
    n_gt, n_found = len(gdf), int(gdf.matched.sum())
    n_pred = len(pdf)
    n_wrong = int((~pdf.matched).sum()) if n_pred else 0

    def box_counts(names):
        g = gdf[gdf.image.isin(names)]
        q = pdf[pdf.image.isin(names)] if n_pred else pdf
        return dict(gt=len(g), found=int(g.matched.sum()), missed=int((~g.matched).sum()),
                    wrong=int((~q.matched).sum()) if len(q) else 0)

    for g, mask in groups.items():
        loc_reg[g].update(box_counts(set(mask[mask].index)))

    def draw_loc(rows, folder, key):
        for i, row in enumerate(rows.itertuples(), 1):
            im = cv2.imread(row.path)
            h, w = im.shape[:2]
            for gb in gt_boxes(Path(row.image).stem, w, h):
                cv2.rectangle(im, (int(gb[0]), int(gb[1])), (int(gb[2]), int(gb[3])), (0, 200, 0), 2)
            r = det.predict(row.path, imgsz=640, conf=CONF_T, device=0, verbose=False)[0]
            for b in r.boxes:
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(im, f"{float(b.conf):.2f}", (x1, max(30, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            x1, y1, x2, y2 = (int(v) for v in row.box)  # highlight the box this file is about
            cv2.rectangle(im, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 255, 255), 1)
            m = df.loc[row.image]
            regs = ",".join(rg for rg in REGIONS if m[rg] == 1)
            extra = f"box area={row.area:.4f}" if key == "area" else f"conf={row.conf:.2f} bestIoU={row.best_iou:.2f}"
            banner(im, f"{extra} | {regs} | {view_of(m)} | hw={int(m.hardware)} | {w}x{h}")
            val = row.area if key == "area" else row.conf
            cv2.imwrite(str(ERR / "localization" / folder / f"{i:02d}_{Path(row.image).stem}_{key}{val:.3f}.jpg"), im)

    loc_missed = gdf[~gdf.matched].sort_values("area", ascending=False).head(10)
    loc_wrong = pdf[~pdf.matched].sort_values("conf", ascending=False).head(10) if n_pred else pdf
    draw_loc(loc_missed, "missed_boxes", "area")
    if len(loc_wrong):
        draw_loc(loc_wrong, "wrong_boxes", "conf")

    def gt_stats(d):
        if not len(d):
            return dict(n=0)
        m = df.loc[d.image]
        return dict(n=len(d), median_area=float(d.area.median()), hardware=float(m.hardware.mean()),
                    mixed=float((m.mixed == 1).mean()), views=m.apply(view_of, axis=1).value_counts().to_dict())

    loc_patterns = {"Missed true boxes": gt_stats(gdf[~gdf.matched]), "Found true boxes": gt_stats(gdf[gdf.matched])}
    wrong_on_normal = int(pdf[~pdf.matched].on_normal.sum()) if n_pred else 0
    small_t = float(gdf.area.median())
    small_recall = gdf[gdf.area < small_t].matched.mean()
    large_recall = gdf[gdf.area >= small_t].matched.mean()

    cp.reset_index()[["image", "true", "p", "pred", *REGIONS, "hardware", "view"]].to_csv(OUT / "classification_test_predictions.csv", index=False)
    gdf.drop(columns=["path"]).to_csv(OUT / "localization_test_true_boxes.csv", index=False)
    if n_pred:
        pdf.drop(columns=["path"]).to_csv(OUT / "localization_test_pred_boxes.csv", index=False)

    # ===================== WRITE =====================
    def stat_row(name, s):
        if not s.get("n"):
            return f"| {name} | 0 | - | - | - | - | - |"
        return (f"| {name} | {s['n']} | {fmt(s['hardware'])} | {fmt(s['mixed'])} | {fmt(s.get('median_mpix'))} | "
                f"{', '.join(f'{k} {v}' for k, v in s['views'].items())} | "
                f"{', '.join(f'{k} {v}' for k, v in s.get('regions', {}).items())} |")

    c = cls_all
    L = ["## Analysis", "",
         f"Final models on the test set; the classifier threshold ({thr:.4f}) was fixed on val. Nothing here was tuned on test.", "",
         "**Region note:** region groups come from the dataset.csv flags and **overlap**: a mixed image is also counted "
         "under every body part it is flagged with (IMG0003298 is hand + leg + mixed). The *(single)* rows keep only "
         "images with exactly one of hand / leg / hip / shoulder and mixed = 0, so those rows do not overlap.", "",
         "### Classification (classifier only, per image)", "",
         "| true \\ predicted | fractured | non_fractured |", "|---|---|---|",
         f"| fractured | {c['tp']} | {c['fn']} |", f"| non_fractured | {c['fp']} | {c['tn']} |", "",
         "| Group | Images | Fractured | Accuracy | Recall | Precision | F1 | TP / FN / FP / TN |",
         "|---|---|---|---|---|---|---|---|"]
    for g, m in [("**all**", cls_all)] + list(cls_reg.items()):
        L.append(f"| {g} | {m['n']} | {m['tp'] + m['fn']} | {fmt(m['acc'])} | {fmt(m['recall'])} | "
                 f"{fmt(m['precision'])} | {fmt(m['f1'])} | {m['tp']} / {m['fn']} / {m['fp']} / {m['tn']} |")
    L += ["", "'-' = undefined (no fractured images or no positive predictions in that group). "
          "Groups with only a few fractured images give unstable numbers.", "",
          f"Errors: `errors/classification/missed_fractures/` ({len(cls_fn)} most confidently missed, lowest P(fractured)) and "
          f"`errors/classification/false_alarms/` ({len(cls_fp)} most confident, highest P(fractured)).", "",
          "| Group | n | Hardware | Mixed | Median image (MP) | Views | Regions |", "|---|---|---|---|---|---|---|"]
    L += [stat_row(k, v) for k, v in cls_patterns.items()]

    L += ["", "### Localization (detector only, per box, IoU >= 0.5)", "",
          f"Test set: **{n_gt} true fracture boxes** in {int(df.loc[[p.name for p in test_imgs], 'fracture_count'].gt(0).sum())} "
          f"fractured images (the 924 boxes in FracAtlas are spread over all splits). At conf >= {CONF_T}: "
          f"**{n_found} found, {n_gt - n_found} missed** (box recall {n_found / n_gt:.3f}); "
          f"{n_pred} predicted boxes, of which {n_wrong} wrong ({wrong_on_normal} on non-fractured images, "
          f"{n_wrong - wrong_on_normal} on fractured images but in the wrong place).", "",
          "mAP50, precision and recall are Ultralytics values (P/R at the confidence with best F1); "
          "Found / Missed / Wrong are the box counts at conf >= 0.25.", "",
          "| Group | Images | Fractured images | mAP50 | Precision | Recall | True boxes | Found | Missed | Wrong boxes |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for g, m in loc_reg.items():
        L.append(f"| {'**all**' if g == 'all' else g} | {m['n']} | {m['n_fractured']} | {fmt(m.get('map50'))} | "
                 f"{fmt(m.get('precision'))} | {fmt(m.get('recall'))} | {m.get('gt', 0)} | {m.get('found', 0)} | "
                 f"{m.get('missed', 0)} | {m.get('wrong', 0)} |")
    L += ["", f"Box size: recall on boxes smaller than the median ({small_t:.4f} of image area) = {fmt(small_recall)}, "
          f"on larger boxes = {fmt(large_recall)}.", ""]
    if pr_curve:
        L += [f"Precision-recall curve (test, all images): `analysis/{pr_curve.name}`", "",
              f"![Detection PR curve](analysis/{pr_curve.name})", ""]
    L += [f"Errors: `errors/localization/missed_boxes/` ({len(loc_missed)} largest missed true boxes) and "
          f"`errors/localization/wrong_boxes/` ({len(loc_wrong)} most confident wrong boxes). "
          "Green = true box, red = predicted box with confidence, yellow outline = the box the file is about.", "",
          "| Group | n | Median box area | Hardware | Mixed | Views |", "|---|---|---|---|---|---|"]
    for k, s in loc_patterns.items():
        L.append(f"| {k} | {s['n']} | {fmt(s.get('median_area'), 4)} | {fmt(s.get('hardware'))} | {fmt(s.get('mixed'))} | "
                 f"{', '.join(f'{a} {b}' for a, b in s.get('views', {}).items())} |" if s.get("n") else f"| {k} | 0 | - | - | - | - |")
    L.append("")

    sel = ROOT / "runs" / "detect" / "detect_selection.json"
    if sel.exists():
        sj = json.loads(sel.read_text())
        L += ["### Detection models on val", "",
              "| Run | Val mAP50 | Val mAP50-95 | Val precision | Val recall |", "|---|---|---|---|---|"]
        for r, m in sj["val"].items():
            L.append(f"| {r}{' (chosen)' if r == sj['chosen'] else ''} | {m['map50']:.3f} | {m['map50_95']:.3f} | "
                     f"{m['precision']:.3f} | {m['recall']:.3f} |")
        L += ["", f"`{sj['chosen']}` was chosen because it has the higher val mAP50 on the cleaned val set "
              "(rule fixed before testing). Only this model was evaluated on test.", ""]
    L += ["### Summary", "", "<!-- SUMMARY -->", ""]

    res = (ROOT / "results.md").read_text(encoding="utf-8")
    if "## Analysis" in res:
        res = res[:res.index("## Analysis")].rstrip() + "\n\n"
    (ROOT / "results.md").write_text(res + "\n".join(L), encoding="utf-8")

    summary = dict(classification=dict(all=cls_all, groups=cls_reg, patterns=cls_patterns),
                   localization=dict(groups=loc_reg, n_gt=n_gt, found=n_found, n_pred=n_pred, wrong=n_wrong,
                                     wrong_on_normal=wrong_on_normal, small_recall=small_recall,
                                     large_recall=large_recall, patterns=loc_patterns))
    (OUT / "analysis.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
