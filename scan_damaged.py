"""Find damaged FracAtlas images, list them, and remove them from every split of both datasets.

1) Replace any remaining hard links under datasets/ with real copies; verify none remain.
2) Scan the ORIGINAL images (FracAtlas/images, verified identical to FracAtlas.zip), read-only.
   The dataset copies are not used for the scan because Ultralytics already re-saved some of them.
     unreadable - cannot be fully decoded (truncated data)
     repairable - decodes, but the JPEG end marker is missing (file is cut off)
   Both count as damaged.
3) Delete every damaged image (and its label) from datasets/det and datasets/cls, all splits,
   delete stale Ultralytics label caches, write damaged_images.txt, print removed counts per split.
"""
import os
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
ORIG = ROOT / "FracAtlas" / "images"
DET = ROOT / "datasets" / "det"
CLS = ROOT / "datasets" / "cls"
OUT = ROOT / "damaged_images.txt"
SPLITS = ("train", "val", "test")


def unlink_all(root: Path) -> int:
    n = 0
    for p in root.rglob("*"):
        if p.is_file() and os.stat(p).st_nlink > 1:
            tmp = p.with_name(p.name + ".tmpcopy")
            shutil.copy2(p, tmp)
            os.replace(tmp, p)
            n += 1
    return n


def check(p: Path):
    Image.LOAD_TRUNCATED_IMAGES = False
    try:
        with Image.open(p) as im:
            im.load()
    except Exception:
        return "unreadable"
    with open(p, "rb") as f:
        f.seek(-2, 2)
        if f.read() != b"\xff\xd9":
            return "repairable"
    return None


def main():
    n = unlink_all(ROOT / "datasets")
    remaining = sum(1 for p in (ROOT / "datasets").rglob("*") if p.is_file() and os.stat(p).st_nlink > 1)
    print(f"hard links converted: {n} | hard links remaining under datasets/: {remaining}")
    assert remaining == 0

    split_of = {p.name: s for s in SPLITS for p in (DET / "images" / s).glob("*.jpg")}

    rows = []
    for folder, cname in (("Fractured", "fractured"), ("Non_fractured", "non_fractured")):
        for p in sorted((ORIG / folder).glob("*.jpg")):
            if p.name not in split_of:
                continue  # e.g. the 2 fractured images duplicated in Non_fractured/ are used from Fractured/
            if folder == "Non_fractured" and (ORIG / "Fractured" / p.name).exists():
                continue
            status = check(p)
            if status:
                rows.append((split_of[p.name], cname, p.name, status, p.stat().st_size))

    removed = Counter()
    for split, cname, name, status, _ in rows:
        stem = Path(name).stem
        for f in (DET / "images" / split / name, DET / "labels" / split / f"{stem}.txt", CLS / split / cname / name):
            if f.exists():
                f.unlink()
        removed[(split, cname)] += 1
    for c in (DET / "labels").glob("*.cache"):
        c.unlink()

    lines = ["# Damaged FracAtlas images (scan of originals; identical to FracAtlas.zip). Removed from all datasets.",
             "split\tclass\timage\tstatus\tbytes"]
    lines += [f"{s}\t{c}\t{n}\t{st}\t{b}" for s, c, n, st, b in sorted(rows)]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sizes = Counter(b for *_, b in rows)
    print(f"damaged images: {len(rows)}  ({Counter(r[3] for r in rows)})  most common sizes: {sizes.most_common(3)}")
    print(f"{'split':<6}{'fractured':>11}{'non_fract':>11}{'removed':>9}{'left(det)':>11}{'left(cls)':>11}")
    for s in SPLITS:
        left_det = len(list((DET / "images" / s).glob("*.jpg")))
        left_cls = len(list((CLS / s).rglob("*.jpg")))
        assert left_det == left_cls
        fr, nf = removed[(s, "fractured")], removed[(s, "non_fractured")]
        print(f"{s:<6}{fr:>11}{nf:>11}{fr + nf:>9}{left_det:>11}{left_cls:>11}")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
