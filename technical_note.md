# Technical Note — Cravv SousChef

One model. Three jobs from one frame: segment the cookware, box the pan/stirrer/tap, predict the cooking state. One forward pass at inference.

## Architecture

```
Input 512×512×3
     │
ResNet-34 (ImageNet pretrained)
     │
C1   C2   C3   C4         (stride 4 → 32)
 └────┴────┴────┘
     FPN (128ch)
 ┌────┬────┬────┐
P2   P3   P4   P5
 │    │    │    │
SegHead  DetHead  ClsHead
 │        │         │
mask    3 boxes   5-class
512²    + conf    logits
```

~21M params. One backbone, three light heads.

## Why FPN

The three tasks want features at very different scales. Segmentation needs pixel-level detail (stride 4). Classification needs one global summary (stride 32). Detection needs both. FPN gives each head the level it wants without running three separate backbones — top-down fusion pushes high-level semantics back into the fine layers, lateral connections preserve spatial detail.

Without FPN: either bad segmentation boundaries, or 3× the compute.

## How the heads are wired

- **Seg head** — consumes P2 through P5, upsamples them to P2 resolution, concatenates, convs, then bilinearly upsamples the 2-channel prediction back to 512×512.
- **Det head** — consumes P3, P4, P5. Each pooled to 2×2, concatenated, MLP. Outputs (confidence_logit, x, y, w, h) per class for the three classes. No anchors, no NMS. Direct regression because we have 60 total detection images — anchor-based heads need an order of magnitude more.
- **Cls head** — consumes P5 only, global-average-pools to 128-d, two FC layers with dropout 0.3, 5 logits for {0, 0.25, 0.5, 0.75, 1.0}.

## Multi-task loss balancing

```
L = 1.0 · L_seg + 2.0 · L_det + 0.5 · L_cls
```

Weights tuned by observation. Seg converges easily → 1.0. Detection lagged on val → doubled. Classification overfits fast (train_acc → 1.0 in 5 epochs) → halved.

Inside `L_det`: BCE on confidence + **2.0 · GIoU** on box coordinates. I started with L1 and hit the classic trap — L1 drops to 0.02 while mAP@50 stalls at 0.75 because L1 and IoU only loosely correlate. Switching to GIoU lifted val mAP from 0.75 to 0.85 in the same epoch budget. GIoU directly optimizes the metric we report.

## Per-head training approach

All three heads train **together** — every step touches every head. The smaller loaders (detection, classification) cycle against the segmentation loader.

- **LR groups:** backbone 1e-5, heads 1e-3. Pretrained ImageNet features stay intact while heads converge fast.
- **Class imbalance (cls):** `WeightedRandomSampler` balances every batch *and* class weights are passed into `CrossEntropyLoss`. State `1.0` is 4× rarer than state `0`; either trick alone is too weak.
- **Augmentation, scaled to dataset size:** detection (42 train images, smallest) gets hflip + color jitter + zoom-out pad with box correction. Seg and cls get only flips.
- **Early stopping** with patience on `avg(mIoU, mAP, acc)`. Stopped at epoch 25; best checkpoint was epoch 18.

## Results & honest read

**mIoU 0.982 ✓ | mAP 0.852 ✗ (target 0.90) | cls acc 0.966 ✓ | avg 0.933**

Two targets cleared. Detection missed the bar. Train mAP is 0.976 — the model *can* learn it. The 0.18 train-val gap is **data, not capacity**: 42 train images, 9 val. mAP@50 on 9 images has a noise floor of ±0.10 from seed alone. Crossing 0.90 here is partly a coin flip with the val set we have.

## Pilot roadmap (in ROI order)

1. **Label 300-500 detection frames.** The only intervention that moves the val noise floor. Everything else is downstream.
2. **Anchor-free FCOS-style detection head.** Current per-class regression breaks the day a frame has two pans.
3. **Uncertainty-based loss weighting** (Kendall et al.) — replaces hand-tuned weights with learned task-uncertainties. Scales when we add more heads.
4. **Mosaic + MixUp augmentation** for detection. Cheap synthetic samples that help small datasets.
5. **Temporal smoothing for classification.** Cooking state is slow-changing; a small ConvLSTM over 4-frame windows kills boundary jitter.
6. **TTA at inference.** ~2-3 mAP for free.
7. **Edge deployment.** TorchScript export + INT8 quantization, target sub-50ms on the kitchen device.

The POC proves the architecture. The pilot scales the data, hardens detection, and proves edge latency on real hardware.
