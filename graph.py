import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_log(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    rows.sort(key=lambda r: r["epoch"])
    return rows


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"saved {path}")


def plot(epochs, series, title, ylabel, out_path, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, values in series.items():
        ax.plot(epochs, values, marker="o", markersize=3, linewidth=1.5, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend()
    save(fig, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="training_log.csv")
    parser.add_argument("--out", default="graphs")
    args = parser.parse_args()

    rows = read_log(args.log)
    epochs = [r["epoch"] for r in rows]
    out = Path(args.out)
    loss_dir = out / "loss"
    acc_dir = out / "accuracy"

    plot(
        epochs,
        {"total": [r["loss_total"] for r in rows]},
        "Total training loss", "loss",
        loss_dir / "total.png",
    )
    plot(
        epochs,
        {"seg": [r["loss_seg"] for r in rows]},
        "Segmentation loss", "loss",
        loss_dir / "seg.png",
    )
    plot(
        epochs,
        {"det": [r["loss_det"] for r in rows]},
        "Detection loss", "loss",
        loss_dir / "det.png",
    )
    plot(
        epochs,
        {"cls": [r["loss_cls"] for r in rows]},
        "Classification loss", "loss",
        loss_dir / "cls.png",
    )
    plot(
        epochs,
        {
            "seg": [r["loss_seg"] for r in rows],
            "det": [r["loss_det"] for r in rows],
            "cls": [r["loss_cls"] for r in rows],
        },
        "Per-task loss (overlay)", "loss",
        loss_dir / "per_task.png",
    )

    plot(
        epochs,
        {
            "val mIoU": [r["val_mIoU"] for r in rows],
            "train mIoU": [r["train_mIoU"] for r in rows],
        },
        "Segmentation mIoU", "mIoU",
        acc_dir / "miou.png",
        ylim=(0, 1.05),
    )
    plot(
        epochs,
        {
            "val mAP": [r["val_mAP"] for r in rows],
            "train mAP": [r["train_mAP"] for r in rows],
            "target 0.90": [0.9] * len(epochs),
        },
        "Detection mAP@50", "mAP",
        acc_dir / "map.png",
        ylim=(0, 1.05),
    )
    plot(
        epochs,
        {
            "val acc": [r["val_acc"] for r in rows],
            "train acc": [r["train_acc"] for r in rows],
        },
        "Classification accuracy", "accuracy",
        acc_dir / "cls_acc.png",
        ylim=(0, 1.05),
    )
    plot(
        epochs,
        {"avg score": [r["avg_score"] for r in rows]},
        "Average score (mIoU + mAP + acc) / 3", "avg score",
        acc_dir / "avg_score.png",
        ylim=(0, 1.05),
    )
    plot(
        epochs,
        {
            "mIoU gap": [r["train_mIoU"] - r["val_mIoU"] for r in rows],
            "mAP gap": [r["train_mAP"] - r["val_mAP"] for r in rows],
            "acc gap": [r["train_acc"] - r["val_acc"] for r in rows],
        },
        "Overfitting gap (train - val)", "gap",
        acc_dir / "overfit_gap.png",
    )

    best = max(rows, key=lambda r: r["avg_score"])
    print(
        f"best epoch {int(best['epoch'])} | "
        f"mIoU {best['val_mIoU']:.3f} | mAP {best['val_mAP']:.3f} | "
        f"acc {best['val_acc']:.3f} | avg {best['avg_score']:.3f}"
    )


if __name__ == "__main__":
    main()
