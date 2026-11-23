"""Remove cross-split duplicates (byte-identical images with different names) to prevent leakage.

Hashes the ORIGINAL FracAtlas files (dataset copies may have been re-saved by Ultralytics).
For each group of identical images spread over several splits, keeps the copies in the earliest split
(train < val < test) and deletes the later-split copies from datasets/det (image + label) and datasets/cls.
Writes duplicates_removed.txt. Run after scan_damaged.py.
"""
import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
ORIG = ROOT / "FracAtlas"
DET = ROOT / "datasets" / "det"
CLS = ROOT / "datasets" / "cls"
ORDER = {"train": 0, "val": 1, "test": 2}


def main():
    df = pd.read_csv(ORIG / "dataset.csv").set_index("image_id")
    split_of = {p.name: s for s in ORDER for p in (DET / "images" / s).glob("*.jpg")}

    groups = defaultdict(list)
    for name, split in split_of.items():
        fr = df.loc[name, "fracture_count"] > 0
        src = ORIG / "images" / ("Fractured" if fr else "Non_fractured") / name
        groups[hashlib.md5(src.read_bytes()).hexdigest()].append((split, name, "fractured" if fr else "non_fractured"))

    removed = []
    for members in groups.values():
        splits = {s for s, _, _ in members}
        if len(splits) < 2:
            continue
        labels = {c for _, _, c in members}
        keep = min(splits, key=ORDER.get)
        for split, name, cname in members:
            if split == keep:
                continue
            stem = Path(name).stem
            for f in (DET / "images" / split / name, DET / "labels" / split / f"{stem}.txt", CLS / split / cname / name):
                if f.exists():
                    f.unlink()
            kept = [n for s, n, _ in members if s == keep]
            removed.append(f"{split}\t{cname}\t{name}\tduplicate of {keep}:{','.join(kept)}"
                           + ("\tCONFLICTING LABELS" if len(labels) > 1 else ""))
    for c in (DET / "labels").glob("*.cache"):
        c.unlink()

    (ROOT / "duplicates_removed.txt").write_text(
        "# Byte-identical images found in more than one split; later-split copy removed\n"
        "split\tclass\timage\tnote\n" + "\n".join(removed) + "\n", encoding="utf-8")
    print(f"cross-split duplicates removed: {len(removed)}")
    for r in removed:
        print("  " + r)

    # Verify: no content hash remains in more than one split
    left = defaultdict(set)
    for members in groups.values():
        for split, name, _ in members:
            if (DET / "images" / split / name).exists():
                left[id(members)].add(split)
    assert all(len(s) <= 1 for s in left.values()), "cross-split duplicates remain"
    print("verified: no identical image remains in more than one split")


if __name__ == "__main__":
    main()
