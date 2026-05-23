import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms.functional as TF

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def to_tensor(img):
    t = TF.to_tensor(img)
    return TF.normalize(t, MEAN, STD)


class SegmentationDataset(Dataset):
    def __init__(self, root, split="train", image_size=512):
        self.size = image_size
        self.train = split == "train"
        base = Path(root) / "segmentation" / split
        imgs = sorted(p for p in (base / "images").iterdir() if p.suffix.lower() in IMG_EXT)
        masks = {p.stem: p for p in (base / "masks").iterdir() if p.suffix.lower() in IMG_EXT}
        self.pairs = [(p, masks[p.stem]) for p in imgs if p.stem in masks]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = TF.resize(img, (self.size, self.size))
        mask = TF.resize(mask, (self.size, self.size), interpolation=TF.InterpolationMode.NEAREST)

        if self.train:
            if random.random() > 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() > 0.5:
                img, mask = TF.vflip(img), TF.vflip(mask)

        mask = (np.array(mask) > 127).astype(np.int64)
        return to_tensor(img), torch.from_numpy(mask)


class DetectionDataset(Dataset):
    def __init__(self, root, split="train", image_size=512):
        self.size = image_size
        self.train = split == "train"
        base = Path(root) / "detection" / split
        self.img_dir = base / "images"
        with open(base / "_annotations.coco.json") as f:
            coco = json.load(f)

        self.anns_by_img = {im["id"]: [] for im in coco["images"]}
        for a in coco["annotations"]:
            if a["image_id"] in self.anns_by_img:
                self.anns_by_img[a["image_id"]].append(a)
        self.images = [im for im in coco["images"] if (self.img_dir / im["file_name"]).exists()]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im = self.images[idx]
        img = Image.open(self.img_dir / im["file_name"]).convert("RGB")
        w0, h0 = img.size
        img = TF.resize(img, (self.size, self.size))

        conf = torch.zeros(3)
        box = torch.zeros(3, 4)
        for a in self.anns_by_img[im["id"]]:
            cid = a["category_id"]
            if cid in (1, 2, 3):
                i = cid - 1
                x, y, w, h = a["bbox"]
                conf[i] = 1.0
                box[i] = torch.tensor([
                    (x + w / 2) / w0,
                    (y + h / 2) / h0,
                    w / w0,
                    h / h0,
                ]).clamp(0, 1)

        if self.train:
            if random.random() > 0.5:
                img = TF.hflip(img)
                for c in range(3):
                    if conf[c] > 0:
                        box[c, 0] = 1.0 - box[c, 0]

            if random.random() > 0.5:
                img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
                img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))
                img = TF.adjust_saturation(img, random.uniform(0.8, 1.2))

            if random.random() > 0.5:
                pad = random.randint(8, 48)
                img = TF.pad(img, [pad, pad, pad, pad], fill=0)
                img = TF.resize(img, (self.size, self.size))
                s = self.size / (self.size + 2 * pad)
                for c in range(3):
                    if conf[c] > 0:
                        box[c, 0] = 0.5 + (box[c, 0] - 0.5) * s
                        box[c, 1] = 0.5 + (box[c, 1] - 0.5) * s
                        box[c, 2] *= s
                        box[c, 3] *= s

        return to_tensor(img), conf, box


class ClassificationDataset(Dataset):
    CLASS_MAP = {"0": 0, "0.25": 1, "0.50": 2, "0.75": 3, "1.0": 4}

    def __init__(self, root, split="train", image_size=512):
        self.size = image_size
        self.train = split == "train"
        base = Path(root) / "classification" / split
        self.samples = []
        for name, idx in self.CLASS_MAP.items():
            d = base / name
            if d.is_dir():
                for p in d.iterdir():
                    if p.suffix.lower() in IMG_EXT:
                        self.samples.append((p, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = TF.resize(img, (self.size, self.size))
        if self.train:
            if random.random() > 0.5:
                img = TF.hflip(img)
            if random.random() > 0.5:
                img = TF.vflip(img)
        return to_tensor(img), label


def get_class_weights_and_sampler(dataset):
    labels = [lbl for _, lbl in dataset.samples]
    counts = np.bincount(labels)
    sample_weights = [1.0 / counts[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    loss_w = len(labels) / (len(counts) * counts)
    loss_w = loss_w / loss_w.min()
    return torch.tensor(loss_w, dtype=torch.float32), sampler
