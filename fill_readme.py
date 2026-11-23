"""Fill the README.md placeholders from the pipeline outputs. Safe to re-run (idempotent).

Each placeholder <!-- KEY --> becomes <!-- KEY -->value<!-- /KEY -->; on later runs the value between the
markers is replaced. Reads: datasets/cls, damaged_images.txt, final_models/*.json, runs/*/<run>/ files.
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"
SPLITS = ("train", "val", "test")


def yaml_val(path, key):
    for line in path.read_text().splitlines():
        if line.startswith(key + ":"):
            return line.split(":", 1)[1].strip()
    return "?"


def counts():
    n = {(s, c): len(list((ROOT / "datasets" / "cls" / s / c).glob("*.jpg")))
         for s in SPLITS for c in ("fractured", "non_fractured")}
    fr = " / ".join(f"{n[s, 'fractured']:,}" for s in SPLITS)
    nf = " / ".join(f"{n[s, 'non_fractured']:,}" for s in SPLITS)
    al = " / ".join(f"{n[s, 'fractured'] + n[s, 'non_fractured']:,}" for s in SPLITS)
    return fr, nf, al


def damaged():
    rows = [l.split("\t") for l in (ROOT / "damaged_images.txt").read_text(encoding="utf-8").splitlines()
            if l.count("\t") >= 3 and l.split("\t")[0] in SPLITS]
    per = {s: sum(r[0] == s for r in rows) for s in SPLITS}
    nfr = sum(r[1] == "fractured" for r in rows)
    who = "all non-fractured" if nfr == 0 else f"{nfr} fractured"
    return (f"In total {len(rows)} damaged images were removed ({who}): "
            f"train {per['train']}, val {per['val']}, test {per['test']}.")


def run_epochs(run_dir, metric):
    with open(run_dir / "results.csv", newline="") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    best = max(rows, key=lambda r: float(r[metric]))
    return len(rows), int(float(best["epoch"]))


def main():
    fr, nf, al = counts()
    vals = {"COUNTS_FR": fr, "COUNTS_NF": nf, "COUNTS_ALL": al, "DAMAGED": damaged()}

    cinfo_p, dinfo_p = ROOT / "final_models" / "classifier_threshold.json", ROOT / "final_models" / "detector_info.json"
    if cinfo_p.exists() and dinfo_p.exists():
        cinfo, dinfo = json.loads(cinfo_p.read_text()), json.loads(dinfo_p.read_text())
        crun, drun = cinfo["source_run"], dinfo["source_run"]
        cdir, ddir = ROOT / "runs" / "classify" / crun, ROOT / "runs" / "detect" / drun
        cval = json.loads((cdir / "val_threshold.json").read_text())
        ctest = json.loads((cdir / "test_results.json").read_text())
        dtest = json.loads((ddir / "test_results.json").read_text())

        arch = Path(cval["model"]).stem.replace("yolov8", "YOLOv8")
        c_ep, c_best = run_epochs(cdir, "metrics/accuracy_top1")
        vals["CLS_MODEL"] = f"{arch} (ImageNet-pretrained), run `{crun}`"
        vals["CLS_SETTINGS"] = (f"imgsz {cval['imgsz']}, batch {cval['batch']}, up to {cval['epochs']} epochs "
                                f"(patience {cval['patience']}; ran {c_ep}, best epoch {c_best}), seed 42; fractured "
                                f"train images oversampled 3x; threshold {cinfo['fractured_threshold']:.4f} tuned on "
                                f"val for fractured recall >= {cval['target_recall']:.2f}")
        d_arch = Path(yaml_val(ddir / "args.yaml", "model")).stem.replace("yolov8", "YOLOv8")
        d_ep, d_best = run_epochs(ddir, "metrics/mAP50(B)")
        vals["DET_MODEL"] = d_arch
        vals["DET_RUN"] = f"`{drun}`"
        vals["DET_SETTINGS"] = (f"imgsz {yaml_val(ddir / 'args.yaml', 'imgsz')}, batch "
                                f"{yaml_val(ddir / 'args.yaml', 'batch')}, up to 100 epochs (patience "
                                f"{yaml_val(ddir / 'args.yaml', 'patience')}; ran {d_ep}, best epoch {d_best}), "
                                "seed 42; chosen by val mAP50 among all detection runs")

        tr, va, te = ctest["train"], cval["val_tuned"], ctest["test"]
        bc = dtest["by_conf"]
        b25, b10 = bc.get("0.25"), bc.get("0.1")
        L = ["**Classification** (fractured = positive class, threshold "
             f"{cinfo['fractured_threshold']:.4f} chosen on val)", "",
             "| | Train | Val | Test |", "|---|---|---|---|",
             f"| Accuracy | {tr['accuracy']:.1%} | {va['accuracy']:.1%} | **{te['accuracy']:.1%}** |",
             f"| Fractured recall | {tr['recall']:.1%} | {va['recall']:.1%} | **{te['recall']:.1%}** |",
             f"| Fractured precision | {tr['precision']:.1%} | {va['precision']:.1%} | **{te['precision']:.1%}** |",
             f"| Fractured F1 | {tr['f1']:.3f} | {va['f1']:.3f} | **{te['f1']:.3f}** |", "",
             f"Test confusion matrix: {te['tp']} fractures found, {te['fn']} missed, {te['fp']} false alarms, "
             f"{te['tn']} normal images correctly cleared.", "",
             f"**Localization** (test set, run `{drun}`)", "",
             "| mAP50 | mAP50-95 | Precision | Recall |", "|---|---|---|---|",
             f"| **{dtest['map50']:.3f}** | {dtest['map50_95']:.3f} | {dtest['precision']:.3f} | {dtest['recall']:.3f} |",
             ""]
        if b25 and b10:
            L += ["| Confidence | Boxes found (IoU >= 0.5) | Box recall | Wrong boxes | Normal images with a false box |",
                  "|---|---|---|---|---|"]
            for c, b in (("0.25", b25), ("0.10", b10)):
                L.append(f"| {c} | {b['found']} / {b['gt']} | {b['recall']:.1%} | {b['wrong']} | "
                         f"{b['normal_with_alarm']} / {b['normal_imgs']} |")
            L.append("")
        L += ["Val numbers are slightly optimistic because the model and threshold were chosen on val; "
              "the test numbers are the honest estimate. Full report: `results.md`."]
        vals["RESULTS"] = "\n" + "\n".join(L) + "\n"
    else:
        vals["RESULTS"] = "\n*Filled in after `evaluate_test.py` has run. See `results.md` for the full report.*\n"

    s = README.read_text(encoding="utf-8")
    for k, v in vals.items():
        pat = re.compile(rf"<!-- {k} -->(.*?<!-- /{k} -->)?", re.S)
        if not pat.search(s):
            print(f"warning: placeholder {k} not found in README.md")
            continue
        s = pat.sub(lambda _: f"<!-- {k} -->{v}<!-- /{k} -->", s, count=1)
    README.write_text(s, encoding="utf-8")
    print("README.md placeholders filled:", ", ".join(vals))


if __name__ == "__main__":
    main()
