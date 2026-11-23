"""Build YOLO detection and classification datasets from FracAtlas.

- Fractured = fracture_count > 0 (from dataset.csv, the source of truth).
- Fractured images keep the official Utilities/Fracture Split train/valid/test split.
- Non-fractured images are split randomly (seed 42) in the same ratio as the official split.
- Detection: datasets/det/{images,labels}/{train,val,test}, empty .txt for normal images, data.yaml.
- Classification: datasets/cls/{train,val,test}/{fractured,non_fractured}.
Files are copied so the original FracAtlas images are never modified.
"""
import random
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "FracAtlas"
OUT = ROOT / "datasets"
SEED = 42
SPLITS = {"train": "train", "valid": "val", "test": "test"}  # official name -> YOLO name


def place(src: Path, dst: Path):
    # Real copies, not hard links: Ultralytics rewrites slightly corrupt JPEGs in place
    # ("corrupt JPEG restored and saved"), which would also modify the original files.
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    df = pd.read_csv(SRC / "dataset.csv")
    df["is_fractured"] = df["fracture_count"] > 0
    fractured = set(df.loc[df.is_fractured, "image_id"])
    normal = sorted(df.loc[~df.is_fractured, "image_id"])

    # Official split for fractured images
    split_of = {}
    official = {}
    for name, yname in SPLITS.items():
        ids = pd.read_csv(SRC / "Utilities" / "Fracture Split" / f"{name}.csv")["image_id"].tolist()
        official[yname] = len(ids)
        for i in ids:
            assert i in fractured, f"{i} is in the {name} split but fracture_count == 0"
            split_of[i] = yname
    missing = fractured - split_of.keys()
    assert not missing, f"fractured images missing from official split: {sorted(missing)[:5]}"

    # Same ratio for non-fractured images
    total = sum(official.values())
    rng = random.Random(SEED)
    rng.shuffle(normal)
    n_train = round(len(normal) * official["train"] / total)
    n_val = round(len(normal) * official["val"] / total)
    for i, img in enumerate(normal):
        split_of[img] = "train" if i < n_train else "val" if i < n_train + n_val else "test"

    if OUT.exists():
        shutil.rmtree(OUT)

    det, cls = OUT / "det", OUT / "cls"
    counts = {s: {"fractured": 0, "non_fractured": 0} for s in SPLITS.values()}
    for img, split in split_of.items():
        is_fr = img in fractured
        # Always read fractured images from Fractured/ (2 of them are duplicated in Non_fractured/)
        src_img = SRC / "images" / ("Fractured" if is_fr else "Non_fractured") / img
        stem = Path(img).stem
        cname = "fractured" if is_fr else "non_fractured"
        counts[split][cname] += 1

        # Detection
        place(src_img, det / "images" / split / img)
        lbl = det / "labels" / split / f"{stem}.txt"
        lbl.parent.mkdir(parents=True, exist_ok=True)
        if is_fr:
            shutil.copy2(SRC / "Annotations" / "YOLO" / f"{stem}.txt", lbl)
        else:
            lbl.write_text("")

        # Classification
        place(src_img, cls / split / cname / img)

    (det / "data.yaml").write_text(
        f"path: {det.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        "  0: fractured\n"
    )

    print(f"{'split':<6}{'fractured':>11}{'non_fract':>11}{'total':>8}{'share':>8}")
    grand = sum(sum(c.values()) for c in counts.values())
    for s, c in counts.items():
        t = sum(c.values())
        print(f"{s:<6}{c['fractured']:>11}{c['non_fractured']:>11}{t:>8}{t / grand:>8.1%}")
    print(f"\nDetection dataset:      {det}\nClassification dataset: {cls}\ndata.yaml:              {det / 'data.yaml'}")


if __name__ == "__main__":
    main()
