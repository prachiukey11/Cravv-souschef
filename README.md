# Cravv SousChef

A small computer vision project for a connected kitchen. One model does three jobs.

## Problem

In a smart kitchen, the camera needs to understand what is happening on the stove. We need to answer three questions from one frame:

1. **Where are the objects?** (segmentation mask)
2. **What objects are present?** (pan, stirrer, tap with bounding boxes)
3. **How cooked is the food?** (state: 0, 0.25, 0.5, 0.75, 1.0)

Running three separate models is slow and expensive. Goal: one model, three outputs.

## Approach

One shared backbone (ResNet-34, pretrained on ImageNet) reads the image. A small Feature Pyramid Network (FPN) builds multi-scale features. Three light heads sit on top:

- **Segmentation head** — predicts a pixel mask.
- **Detection head** — predicts one box per class (pan, stirrer, tap).
- **Classification head** — predicts the cooking state (5 classes).

All heads train together. Total loss is a weighted sum of the three task losses.

The classification dataset is imbalanced (some states have many more images than others). To handle this:
- A `WeightedRandomSampler` makes every batch balanced.
- Class weights are passed to `CrossEntropyLoss` so rare classes are not ignored.

The backbone uses a small learning rate (fine-tune). The heads use a larger one.

## Folder Structure

```
configs/default.yaml      training config
dataset/                   images, masks, annotations
scripts/split_data.py      makes train/val/test splits
model/multitask_model.py   ResNet-34 + FPN + 3 heads
utils/transform.py         3 dataset classes + augmentations
train.py                   trains all heads together
inference.py               loads checkpoint, draws visualizations
analysie.py                prints dataset distribution
```

## How to Run

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Split the data
```bash
python scripts/split_data.py --root dataset --force
```

### 3. Train
```bash
python train.py --epochs 30
```
Best weights are saved to `best_model.pt`.

### 4. Run inference
```bash
python inference.py --checkpoint best_model.pt
```
Output images are saved to `outputs/`.

## Metrics

Reported on the validation set after every epoch:
- **mIoU** for segmentation
- **mAP@50** for detection
- **Top-1 accuracy** for classification

The best model is the one with the highest average of these three.
