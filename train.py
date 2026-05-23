import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model.multitask_model import MultiTaskModel
from utils.transform import (
    SegmentationDataset,
    DetectionDataset,
    ClassificationDataset,
    get_class_weights_and_sampler,
)


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def giou_loss(pred, target, eps=1e-7):
    px1, py1 = pred[:, 0] - pred[:, 2] / 2, pred[:, 1] - pred[:, 3] / 2
    px2, py2 = pred[:, 0] + pred[:, 2] / 2, pred[:, 1] + pred[:, 3] / 2
    tx1, ty1 = target[:, 0] - target[:, 2] / 2, target[:, 1] - target[:, 3] / 2
    tx2, ty2 = target[:, 0] + target[:, 2] / 2, target[:, 1] + target[:, 3] / 2

    inter_w = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0)
    inter_h = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(min=0)
    inter = inter_w * inter_h
    union = (px2 - px1) * (py2 - py1) + (tx2 - tx1) * (ty2 - ty1) - inter + eps
    iou_val = inter / union

    enc_w = torch.max(px2, tx2) - torch.min(px1, tx1)
    enc_h = torch.max(py2, ty2) - torch.min(py1, ty1)
    enc_area = enc_w * enc_h + eps
    giou = iou_val - (enc_area - union) / enc_area
    return (1 - giou).mean()


def iou(b1, b2):
    x1 = max(b1[0] - b1[2] / 2, b2[0] - b2[2] / 2)
    y1 = max(b1[1] - b1[3] / 2, b2[1] - b2[3] / 2)
    x2 = min(b1[0] + b1[2] / 2, b2[0] + b2[2] / 2)
    y2 = min(b1[1] + b1[3] / 2, b2[1] + b2[3] / 2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter + 1e-6
    return inter / union


def average_precision(preds, gts, thresh=0.5):
    if not gts:
        return 0.0
    preds = sorted(preds, key=lambda p: p["score"], reverse=True)
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    matched = set()
    for i, p in enumerate(preds):
        if p["img_id"] in gts and iou(p["box"], gts[p["img_id"]]) >= thresh and p["img_id"] not in matched:
            tp[i] = 1
            matched.add(p["img_id"])
        else:
            fp[i] = 1
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / len(gts)
    prec = tp_cum / (tp_cum + fp_cum + 1e-6)
    return float(np.trapz(prec, rec))


@torch.no_grad()
def validate(model, seg_loader, det_loader, cls_loader, device):
    model.eval()

    seg_ious = []
    for imgs, masks in seg_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs, task="seg")["seg"].argmax(1)
        for p, t in zip(preds, masks):
            inter = ((p == 1) & (t == 1)).sum().item()
            union = ((p == 1) | (t == 1)).sum().item()
            seg_ious.append(inter / (union + 1e-6))
    mIoU = float(np.mean(seg_ious)) if seg_ious else 0.0

    correct = total = 0
    for imgs, labels in cls_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs, task="cls")["cls"].argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    cls_acc = correct / total if total else 0.0

    preds_by_class = {c: [] for c in range(3)}
    gts_by_class = {c: {} for c in range(3)}
    for batch_idx, (imgs, confs, boxes) in enumerate(det_loader):
        imgs = imgs.to(device)
        pc, pb = model(imgs, task="det")["det"]
        pc = torch.sigmoid(pc).cpu().numpy()
        pb = pb.cpu().numpy()
        confs, boxes = confs.numpy(), boxes.numpy()
        for b in range(imgs.size(0)):
            img_id = f"{batch_idx}_{b}"
            for c in range(3):
                if confs[b, c] > 0.5:
                    gts_by_class[c][img_id] = boxes[b, c]
                preds_by_class[c].append({"img_id": img_id, "score": pc[b, c], "box": pb[b, c]})
    mAP = float(np.mean([average_precision(preds_by_class[c], gts_by_class[c]) for c in range(3)]))

    return mIoU, mAP, cls_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--log", default="training_log.csv", help="CSV file for per-epoch metrics")
    parser.add_argument("--patience", type=int, default=7, help="early stop after N epochs without improvement")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["data"]["root"])
    size = cfg["train"]["image_size"]
    epochs = args.epochs or cfg["train"]["epochs"]
    device = get_device()
    print(f"Device: {device}")

    train_seg = SegmentationDataset(root, "train", size)
    val_seg = SegmentationDataset(root, "val", size)
    train_det = DetectionDataset(root, "train", size)
    val_det = DetectionDataset(root, "val", size)
    train_cls = ClassificationDataset(root, "train", size)
    val_cls = ClassificationDataset(root, "val", size)

    cls_weights, cls_sampler = get_class_weights_and_sampler(train_cls)
    cls_weights = cls_weights.to(device)

    seg_loader = DataLoader(train_seg, batch_size=cfg["train"]["batch_size_seg"], shuffle=True, drop_last=True)
    det_loader = DataLoader(train_det, batch_size=cfg["train"]["batch_size_det"], shuffle=True, drop_last=True)
    cls_loader = DataLoader(train_cls, batch_size=cfg["train"]["batch_size_cls"], sampler=cls_sampler, drop_last=True)
    val_seg_loader = DataLoader(val_seg, batch_size=8)
    val_det_loader = DataLoader(val_det, batch_size=8)
    val_cls_loader = DataLoader(val_cls, batch_size=16)

    train_eval_seg = DataLoader(Subset(train_seg, range(min(64, len(train_seg)))), batch_size=8)
    train_eval_det = DataLoader(Subset(train_det, range(min(64, len(train_det)))), batch_size=8)
    train_eval_cls = DataLoader(Subset(train_cls, range(min(64, len(train_cls)))), batch_size=16)

    det_iter = cycle(det_loader)
    cls_iter = cycle(cls_loader)

    model = MultiTaskModel(pretrained=True).to(device)

    loss_seg = nn.CrossEntropyLoss()
    loss_cls = nn.CrossEntropyLoss(weight=cls_weights)
    loss_conf = nn.BCEWithLogitsLoss()

    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if name.startswith(("stem", "layer")):
            backbone_params.append(p)
        else:
            head_params.append(p)
    optim = torch.optim.AdamW([
        {"params": backbone_params, "lr": float(cfg["train"]["backbone_lr"])},
        {"params": head_params, "lr": float(cfg["train"]["head_lr"])},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    w = cfg["loss_weights"]
    best = 0.0
    epochs_since_best = 0

    log_path = Path(args.log)
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow([
        "epoch", "loss_total", "loss_seg", "loss_det", "loss_cls",
        "val_mIoU", "val_mAP", "val_acc",
        "train_mIoU", "train_mAP", "train_acc",
        "avg_score",
    ])
    log_file.flush()

    for epoch in range(1, epochs + 1):
        model.train()
        sum_total = sum_seg = sum_det = sum_cls = 0.0
        n_batches = 0
        pbar = tqdm(seg_loader, desc=f"Epoch {epoch}/{epochs}")
        for seg_imgs, seg_masks in pbar:
            seg_imgs, seg_masks = seg_imgs.to(device), seg_masks.to(device)
            det_imgs, det_confs, det_boxes = [t.to(device) for t in next(det_iter)]
            cls_imgs, cls_labels = [t.to(device) for t in next(cls_iter)]

            optim.zero_grad()
            l_seg = loss_seg(model(seg_imgs, task="seg")["seg"], seg_masks)
            pc, pb = model(det_imgs, task="det")["det"]
            l_det = loss_conf(pc, det_confs)
            mask = det_confs == 1
            if mask.any():
                l_det = l_det + 2.0 * giou_loss(pb[mask], det_boxes[mask])
            l_cls = loss_cls(model(cls_imgs, task="cls")["cls"], cls_labels)

            total = w["seg"] * l_seg + w["det"] * l_det + w["cls"] * l_cls
            total.backward()
            optim.step()

            sum_total += total.item()
            sum_seg += l_seg.item()
            sum_det += l_det.item()
            sum_cls += l_cls.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{total.item():.3f}")

        sched.step()

        avg_total = sum_total / n_batches
        avg_seg = sum_seg / n_batches
        avg_det = sum_det / n_batches
        avg_cls = sum_cls / n_batches

        mIoU, mAP, acc = validate(model, val_seg_loader, val_det_loader, val_cls_loader, device)
        t_mIoU, t_mAP, t_acc = validate(model, train_eval_seg, train_eval_det, train_eval_cls, device)
        score = (mIoU + mAP + acc) / 3
        print(f"Epoch {epoch} | loss {avg_total:.3f} (seg {avg_seg:.3f} det {avg_det:.3f} cls {avg_cls:.3f})")
        print(f"  val  : mIoU {mIoU:.3f} | mAP {mAP:.3f} | acc {acc:.3f} | avg {score:.3f}")
        print(f"  train: mIoU {t_mIoU:.3f} | mAP {t_mAP:.3f} | acc {t_acc:.3f}  "
              f"(gap: mIoU {t_mIoU-mIoU:+.3f}, mAP {t_mAP-mAP:+.3f}, acc {t_acc-acc:+.3f})")

        writer.writerow([epoch, f"{avg_total:.4f}", f"{avg_seg:.4f}", f"{avg_det:.4f}", f"{avg_cls:.4f}",
                         f"{mIoU:.4f}", f"{mAP:.4f}", f"{acc:.4f}",
                         f"{t_mIoU:.4f}", f"{t_mAP:.4f}", f"{t_acc:.4f}",
                         f"{score:.4f}"])
        log_file.flush()

        if score > best:
            best = score
            epochs_since_best = 0
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  --> Saved best_model.pt (score {best:.3f})")
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"Early stop: no improvement for {args.patience} epochs (best {best:.3f})")
                break

    log_file.close()
    print(f"Per-epoch metrics saved to {log_path}")


if __name__ == "__main__":
    main()
