import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import yaml
from PIL import Image

from model.multitask_model import MultiTaskModel
from utils.transform import MEAN, STD

DET_CLASSES = ["pan", "stirrer", "tap"]
DET_COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255)]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess(path, size):
    img = Image.open(path).convert("RGB")
    img = TF.resize(img, (size, size))
    t = TF.normalize(TF.to_tensor(img), MEAN, STD)
    return t.unsqueeze(0)


def visualize(img_path, out, class_names):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    seg = out["seg"][0].argmax(0).cpu().numpy().astype(np.uint8)
    seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay = img.copy()
    overlay[seg == 1] = [0, 255, 0]
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    conf, box = out["det"]
    conf = torch.sigmoid(conf)[0].cpu().numpy()
    box = box[0].cpu().numpy()
    for c in range(3):
        if conf[c] > 0.5:
            xc, yc, bw, bh = box[c]
            x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
            x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), DET_COLORS[c], 2)
            cv2.putText(img, f"{DET_CLASSES[c]} {conf[c]*100:.0f}%", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, DET_COLORS[c], 2)

    probs = torch.softmax(out["cls"][0], 0).cpu().numpy()
    idx = int(np.argmax(probs))
    label = f"State: {class_names[idx]} ({probs[idx]*100:.0f}%)"
    cv2.rectangle(img, (10, 10), (400, 45), (0, 0, 0), -1)
    cv2.putText(img, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return img


def collect_samples(root):
    samples = []
    det_val = root / "detection" / "val" / "images"
    if det_val.exists():
        samples += sorted(det_val.iterdir())[:2]
    seg_val = root / "segmentation" / "val" / "images"
    if seg_val.exists():
        samples += sorted(seg_val.iterdir())[:2]
    cls_val = root / "classification" / "val"
    if cls_val.exists():
        for d in sorted(cls_val.iterdir()):
            if d.is_dir():
                files = sorted(d.iterdir())
                if files:
                    samples.append(files[0])
                    break
    return [p for p in samples if p.suffix.lower() in (".jpg", ".png")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    size = cfg["train"]["image_size"]
    class_names = cfg["data"]["class_names"]

    device = get_device()
    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    model = MultiTaskModel(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in collect_samples(Path(cfg["data"]["root"])):
        x = preprocess(path, size).to(device)
        with torch.no_grad():
            out = model(x)
        vis = visualize(path, out, class_names)
        save_path = out_dir / f"vis_{path.stem}.jpg"
        cv2.imwrite(str(save_path), vis)
        print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
