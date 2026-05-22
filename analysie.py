import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def classification(root):
    print("--- Classification ---")
    base = Path(root) / "classification"
    for split in ["images", "train", "val", "test"]:
        path = base / split
        if not path.exists():
            continue
        print(f"  {split}:")
        for d in sorted(path.iterdir()):
            if d.is_dir():
                n = sum(1 for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
                print(f"    {d.name}: {n}")


def detection(root):
    print("--- Detection ---")
    base = Path(root) / "detection"
    for split in ["", "train", "val", "test"]:
        ann = base / split / "_annotations.coco.json" if split else base / "_annotations.coco.json"
        if not ann.exists():
            continue
        with open(ann) as f:
            coco = json.load(f)
        names = {c["id"]: c["name"] for c in coco["categories"]}
        counts = Counter(a["category_id"] for a in coco["annotations"])
        print(f"  {split or 'all'}: {len(coco['images'])} images, {len(coco['annotations'])} boxes")
        for cid, name in names.items():
            print(f"    {name}: {counts[cid]}")


def segmentation(root):
    print("--- Segmentation ---")
    base = Path(root) / "segmentation"
    for split in ["masks", "train/masks", "val/masks", "test/masks"]:
        path = base / split
        if not path.exists():
            continue
        pixels = Counter()
        files = [p for p in path.iterdir() if p.suffix.lower() in IMG_EXT]
        for p in files:
            mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                for v, c in zip(*np.unique(mask, return_counts=True)):
                    pixels[v] += c
        total = sum(pixels.values()) or 1
        print(f"  {split}: {len(files)} masks")
        for v in sorted(pixels):
            print(f"    value {v}: {pixels[v] / total * 100:.1f}%")


if __name__ == "__main__":
    root = "dataset"
    classification(root)
    detection(root)
    segmentation(root)
