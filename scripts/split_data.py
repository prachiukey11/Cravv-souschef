
import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path


IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def _seeded_shuffle(items, seed):
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    return items


def _split_indices(n, val_frac, test_frac, seed):
    """Returns three disjoint lists of indices (train, val, test)."""
    indices = _seeded_shuffle(range(n), seed)
    n_val = max(1, int(round(n * val_frac)))
    n_test = max(1, int(round(n * test_frac)))
    n_train = n - n_val - n_test
    assert n_train > 0, f"split too aggressive: n={n}, val={n_val}, test={n_test}"
    train = sorted(indices[:n_train])
    val = sorted(indices[n_train : n_train + n_val])
    test = sorted(indices[n_train + n_val :])
    return train, val, test


def _make_relative_symlink(src: Path, dst: Path):
    """Symlink dst -> src, using a path relative to dst's parent so the link
    survives if the dataset is moved together."""
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(src, dst.parent)
    os.symlink(rel, dst)


def split_segmentation(root: Path, val_frac, test_frac, seed):
    src_imgs = root / "segmentation" / "images"
    src_masks = root / "segmentation" / "masks"
    images = sorted([p for p in src_imgs.iterdir() if p.suffix.lower() in IMG_EXT])
    masks_by_stem = {p.stem: p for p in src_masks.iterdir() if p.suffix.lower() in IMG_EXT}
    paired = [(p, masks_by_stem[p.stem]) for p in images if p.stem in masks_by_stem]
    n = len(paired)
    train_idx, val_idx, test_idx = _split_indices(n, val_frac, test_frac, seed)

    for split_name, idxs in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        for i in idxs:
            img, mask = paired[i]
            _make_relative_symlink(img, root / "segmentation" / split_name / "images" / img.name)
            _make_relative_symlink(mask, root / "segmentation" / split_name / "masks" / mask.name)
    print(f"  segmentation: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")


def split_detection(root: Path, val_frac, test_frac, seed):
    src_imgs = root / "detection" / "images"
    src_json = root / "detection" / "_annotations.coco.json"
    with open(src_json) as f:
        coco = json.load(f)
    images = coco["images"]
    n = len(images)
    train_idx, val_idx, test_idx = _split_indices(n, val_frac, test_frac, seed)

    for split_name, idxs in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        img_ids = {images[i]["id"] for i in idxs}
        split_dir = root / "detection" / split_name
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        for i in idxs:
            im = images[i]
            src = src_imgs / im["file_name"]
            _make_relative_symlink(src, split_dir / "images" / im["file_name"])
        sub_coco = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "categories": coco["categories"],
            "images": [images[i] for i in idxs],
            "annotations": [a for a in coco["annotations"] if a["image_id"] in img_ids],
        }
        with open(split_dir / "_annotations.coco.json", "w") as f:
            json.dump(sub_coco, f)
    print(f"  detection:    train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")


def split_classification(root: Path, val_frac, test_frac, seed):
    src_root = root / "classification" / "images"
    class_dirs = sorted([d for d in src_root.iterdir() if d.is_dir()])
    totals = {"train": 0, "val": 0, "test": 0}
    for cls_dir in class_dirs:
        files = sorted([p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXT])
        n = len(files)
        # Use a per-class seed offset so each class is split independently,
        # giving roughly the same per-class fraction in each split (stratified-ish).
        train_idx, val_idx, test_idx = _split_indices(n, val_frac, test_frac, seed + abs(hash(cls_dir.name)) % 10000)
        for split_name, idxs in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
            for i in idxs:
                src = files[i]
                _make_relative_symlink(src, root / "classification" / split_name / cls_dir.name / src.name)
            totals[split_name] += len(idxs)
    print(f"  classification: train={totals['train']} val={totals['val']} test={totals['test']} "
          f"(stratified by class)")


def _existing_split_seed(root: Path) -> int:
    info = root / "split_info.json"
    if info.exists():
        with open(info) as f:
            return json.load(f).get("seed")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="re-create the split even if one already exists with a different seed")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"ERROR: dataset root not found at {root}. Run scripts/prepare_data.sh first.")
        sys.exit(1)

    prev_seed = _existing_split_seed(root)
    if prev_seed is not None and prev_seed != args.seed and not args.force:
        print(f"ERROR: split already exists with seed={prev_seed}, refusing to re-split with "
              f"seed={args.seed}. Re-run with --force if you really want to (this will invalidate "
              f"existing best.pt checkpoints because they were selected on the old val set).")
        sys.exit(1)

    if args.force:
        for task in ("segmentation", "detection", "classification"):
            for split in ("train", "val", "test"):
                p = root / task / split
                if p.exists():
                    shutil.rmtree(p)

    print(f"Splitting {root} with val={args.val} test={args.test} seed={args.seed}")
    split_segmentation(root, args.val, args.test, args.seed)
    split_detection(root, args.val, args.test, args.seed)
    split_classification(root, args.val, args.test, args.seed)

    with open(root / "split_info.json", "w") as f:
        json.dump({"seed": args.seed, "val_frac": args.val, "test_frac": args.test}, f, indent=2)
    print(f"Wrote {root}/split_info.json")
    print("Done. The model will now only see train/. Infer on val/ or test/ to check generalization.")


if __name__ == "__main__":
    main()
