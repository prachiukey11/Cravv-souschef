# Cravv SousChef

A small computer vision project for a connected kitchen. One model does three jobs.

## Problem

In a smart kitchen, the camera needs to understand what is happening on the stove. We need to answer three questions from one frame:

1. **Where are the objects?** (segmentation mask)
2. **What objects are there?** (pan, stirrer, tap with bounding boxes)
3. **How cooked is the food?** (state: 0, 0.25, 0.5, 0.75, 1.0)

Running three separate models is slow and expensive. Goal: one model, three outputs.

## Approach

One shared backbone (ResNet-34, pretrained on ImageNet) reads the image. A small Feature Pyramid Network (FPN) builds multi-scale features. Three light heads sit on top:

- **Segmentation head** — predicts a pixel mask.
- **Detection head** — predicts one box per class (pan, stirrer, tap).
- **Classification head** — predicts the cooking state (5 classes).

All heads train together. Total loss is a weighted sum of the three task losses.

### How class imbalance is handled
Some cooking states have many more images than others. To stop the model from ignoring the rare ones:
- A `WeightedRandomSampler` makes every batch balanced.
- Class weights are passed to `CrossEntropyLoss` so rare classes are not ignored.

### Other small but important choices
- **Backbone uses a small learning rate** (fine-tune); heads use a larger one.
- **Detection box loss is GIoU**, not L1 — it directly drives the IoU metric up.
- **Early stopping** ends training when the score stops improving.
- **Per-epoch metrics** are saved to `training_log.csv` so you can see the model getting better (or not).

## Results

Best epoch on validation set:

| Task                | Metric  | Target | Got       |
| ------------------- | ------- | :----: | :-------: |
| Segmentation        | mIoU    | ≥ 0.90 | **0.982** |
| Detection           | mAP@50  | ≥ 0.90 | **0.852** |
| Cooking state       | Top-1   | ≥ 0.90 | **0.966** |
| Average             | mean    |   —    | **0.933** |

Graphs of loss and metrics per epoch are in [`graphs/`](graphs/). Sample inference outputs are in [`samples/`](samples/). The architecture and design choices are written up in [`technical_note.md`](technical_note.md).

## Folder Structure

```
configs/default.yaml       training config
dataset/                   images, masks, annotations
scripts/split_data.py      makes train/val/test splits
model/multitask_model.py   ResNet-34 + FPN + 3 heads
utils/transform.py         3 dataset classes + augmentations
train.py                   trains all heads together
inference.py               loads checkpoint, draws visualizations
graph.py                   reads training_log.csv, saves PNG plots
analysie.py                prints dataset distribution
graphs/                    plots of loss and metrics per epoch
samples/                   example inference visualizations
technical_note.md          one-page architecture and design memo
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
Best weights are saved to `best_model.pt`. Per-epoch metrics are saved to `training_log.csv`. Training stops early if the score does not improve for 7 epochs.

### 4. Run inference
```bash
python inference.py --checkpoint best_model.pt
```
Output images are saved to `outputs/`.

### 5. See training graphs
```bash
python graph.py
```
Saves plots to `graphs/loss/` and `graphs/accuracy/`.
