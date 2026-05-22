import argparse
import json
import os
import random
import shutil
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def split_indices(n, val_frac, test_frac, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(round(n * val_frac)))
    n_test = max(1, int(round(n * test_frac)))
    n_train = n - n_val - n_test
    return sorted(idx[:n_train]), sorted(idx[n_train:n_train + n_val]), sorted(idx[n_train + n_val:])


def symlink(src, dst):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(os.path.relpath(src, dst.parent), dst)


def split_segmentation(root, val, test, seed):
    imgs = sorted(p for p in (root / "segmentation/images").iterdir() if p.suffix.lower() in IMG_EXT)
    masks = {p.stem: p for p in (root / "segmentation/masks").iterdir() if p.suffix.lower() in IMG_EXT}
    pairs = [(p, masks[p.stem]) for p in imgs if p.stem in masks]
    train, val_idx, test_idx = split_indices(len(pairs), val, test, seed)
    for name, idxs in [("train", train), ("val", val_idx), ("test", test_idx)]:
        for i in idxs:
            img, mask = pairs[i]
            symlink(img, root / "segmentation" / name / "images" / img.name)
            symlink(mask, root / "segmentation" / name / "masks" / mask.name)
    print(f"segmentation: train={len(train)} val={len(val_idx)} test={len(test_idx)}")


def split_detection(root, val, test, seed):
    src = root / "detection/images"
    with open(root / "detection/_annotations.coco.json") as f:
        coco = json.load(f)
    images = coco["images"]
    train, val_idx, test_idx = split_indices(len(images), val, test, seed)
    for name, idxs in [("train", train), ("val", val_idx), ("test", test_idx)]:
        out = root / "detection" / name
        out.mkdir(parents=True, exist_ok=True)
        ids = {images[i]["id"] for i in idxs}
        for i in idxs:
            im = images[i]
            symlink(src / im["file_name"], out / "images" / im["file_name"])
        sub = {
            "categories": coco["categories"],
            "images": [images[i] for i in idxs],
            "annotations": [a for a in coco["annotations"] if a["image_id"] in ids],
        }
        with open(out / "_annotations.coco.json", "w") as f:
            json.dump(sub, f)
    print(f"detection: train={len(train)} val={len(val_idx)} test={len(test_idx)}")


def split_classification(root, val, test, seed):
    src = root / "classification/images"
    totals = {"train": 0, "val": 0, "test": 0}
    for i, cls in enumerate(sorted(d for d in src.iterdir() if d.is_dir())):
        files = sorted(p for p in cls.iterdir() if p.suffix.lower() in IMG_EXT)
        train, val_idx, test_idx = split_indices(len(files), val, test, seed + i)
        for name, idxs in [("train", train), ("val", val_idx), ("test", test_idx)]:
            for j in idxs:
                symlink(files[j], root / "classification" / name / cls.name / files[j].name)
            totals[name] += len(idxs)
    print(f"classification: {totals}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if args.force:
        for task in ("segmentation", "detection", "classification"):
            for split in ("train", "val", "test"):
                p = root / task / split
                if p.exists():
                    shutil.rmtree(p)

    split_segmentation(root, args.val, args.test, args.seed)
    split_detection(root, args.val, args.test, args.seed)
    split_classification(root, args.val, args.test, args.seed)

    with open(root / "split_info.json", "w") as f:
        json.dump({"seed": args.seed, "val_frac": args.val, "test_frac": args.test}, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
