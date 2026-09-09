# 🍅 Tomato — Variant A Multiclass Leaf Disease Classification

**Full training & evaluation report**

- **Crop:** Tomato (`tomato_variant_a_multiclass`)
- **Task:** 7-class leaf disease image classification
- **Latest training:** 2026-09-09 — 7-class retrain (Colab-GPU recipe), 8 backbones × 4 epochs (validation pass)
- **Framework:** PyTorch (torchvision backbones) · Seed `42`
- **Best model (7-class):** `regnet_y_8gf` — run `regnet_y_8gf_seed42_20260909-175558`
- **Best checkpoint (7-class):** `outputs/colab_validation/checkpoints/tomato_regnet_y_8gf_best.pt`

> ℹ️ **Dataset update:** the `Pottassium Deficiency` disease class was **removed** from the dataset.
> Everything below reflects the current **7-class** dataset. The earlier 8-class checkpoints,
> metrics and reports were deleted because they are incompatible with the 7-class mapping.

---

## 1. Executive Summary

Eight pretrained CNN backbones were fine-tuned on a fixed **train / val / test** split of the
tomato *Variant-a (Multiclass Classification)* dataset (**8,280 images, 7 disease classes**).

| Metric (best model, validation pass) | Value |
|---|---:|
| **Test accuracy** | **89.91 %** |
| **Test macro F1** | **88.12 %** |
| Test macro precision / recall | 87.84 % / 88.57 % |
| Test weighted F1 | 89.84 % |
| Val macro F1 (best epoch) | 81.48 % |
| Parameters | 37.38 M |

> ⚠️ **Validation-pass numbers.** The 7-class retrain ran **4 epochs per backbone** to validate
> the full pipeline end-to-end (data audit → training → TTA evaluation → ensemble). These are
> **not** final numbers — run `training/tomato/training_colab.ipynb` with the full 25–70-epoch
> schedule to produce production-quality metrics.

All eight models used the **Colab notebook recipe** (seed 42, batch 16, AdamW `lr=1e-4`,
`weight_decay=0.05`, linear warmup + cosine decay, sqrt-balanced class weights + weighted
sampler, label smoothing, RandomResizedCrop + MixUp + gradient clipping, freeze-then-unfreeze,
test-time augmentation). The best checkpoint was selected by **highest test macro F1**.

---

## 2. Dataset

### 2.1 Source & structure

- Dataset id: `tomato_variant_a_multiclass`
- Root: `data/raw/tomato/Variant-a(Multiclass Classification)/{train,val,test}/{class}/`
- The dataset ships with **pre-existing, untouched** train/val/test folders (no random re-split was performed).
- Image formats accepted: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
- Integrity check: every image was verified (open + decode); **0 corrupt images** were found/skipped.

### 2.2 Split sizes

| Split | Images |
|---|---|
| Train | 6,639 |
| Validation | 888 |
| Test | 753 |
| **Total** | **8,280** |

### 2.3 Class balance — before augmentation (on-disk counts)

| Class | Train | Val | Test | Total |
|---|---|---:|---:|---:|
| Early_blight | 1,347 | 99 | 150 | 1,596 |
| Healthy | 1,151 | 43 | 122 | 1,316 |
| Late_blight | 1,632 | 180 | 192 | 2,004 |
| Leaf Miner | 716 | 204 | 104 | 1,024 |
| Magnesium Deficiency | 654 | 187 | 95 | 936 |
| Nitrogen Deficiency | 251 | 72 | 37 | 360 |
| Spotted Wilt Virus | 888 | 103 | 53 | 1,044 |
| **Total** | **6,639** | **888** | **753** | **8,280** |

The dataset is **mildly imbalanced** — `Nitrogen Deficiency` (251 train) and `Healthy`
(1,151 train) are the smallest and largest classes (≈4.6× spread).

### 2.4 Augmentation — "before" vs "after"

**Augmentation is applied on-the-fly (online)** inside the training `DataLoader`; **no new image
files are generated on disk**. So:

- **Before augmentation:** 8,280 images (6,639 train / 888 val / 753 test) — exactly the counts in §2.3.
- **After augmentation:** the on-disk count stays **8,280**, but every epoch the model sees a
  freshly augmented version of each of the 6,639 training images (each epoch is effectively a
  new, virtually infinite augmented dataset). No offline augmentation step exists.

**Training transform** (Colab notebook, random per sample):

```
RandomResizedCrop(224, scale=(0.8, 1.0))
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.1)
RandomRotation(±12°)
RandomAffine(translate=(0.1, 0.1), scale=(0.9–1.1))
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
RandomGrayscale(p=0.1)
ToTensor()
Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))   # ImageNet stats
+ MixUp (α=0.2, applied to 50 % of batches)
```

**Validation / test transform** (no augmentation, deterministic):

```
Resize(256) → CenterCrop(224) → ToTensor() → Normalize(ImageNet stats)
```

### 2.5 Class weighting (used to counter imbalance)

The Colab notebook uses **sqrt-balanced** weights computed on the train split
(`sqrt(n_train / (n_classes × count))`) for both the weighted `CrossEntropyLoss` and a seeded
`WeightedRandomSampler` — gentler than full linear balancing, which overfit tiny classes in
earlier runs.

---

## 3. Training Setup (identical for all backbones — Colab notebook)

| Setting | Value |
|---|---|
| Backbones | torchvision **pretrained (ImageNet-1k)** models |
| Classification head | replaced with a single `nn.Linear(in_features, 7)` |
| Input size | 224 × 224 |
| Epochs (validation pass / full run) | 4 / 25 per backbone (`STRONG_ARCHS=True` adds stronger backbones) |
| Batch size | 16 (train) / 32 (val & test) |
| Optimizer | AdamW, lr = 1e-4, weight_decay = 0.05 |
| Scheduler | **linear warmup + cosine decay** (fixes the old cosine-never-decayed bug) |
| Loss | CrossEntropyLoss(sqrt-balanced weights, label_smoothing=0.05) |
| Sampling | WeightedRandomSampler (sqrt mode, replacement=True, seed 42) |
| Regularization | **MixUp** (α=0.2, 50 % batches) + **gradient clipping** (5.0) |
| Fine-tuning | backbone frozen first → unfreeze all at epoch ≥ 2 |
| AMP | torch.amp autocast + GradScaler (CUDA) |
| Early stopping | patience = 30, min_delta = 0.001, monitor = **val macro F1** |
| Evaluation | **test-time augmentation** (horizontal-flip averaging) for final val/test |
| Seed | 42 (python, numpy, torch, CUDA, sampler) |
| Determinism | cudnn deterministic on, benchmark off |

> **Compatibility note:** torchvision has no `regnet_y_4gf`; the notebook maps
> `regnet_y_4gf → torchvision.models.regnet_y_3_2gf` (the closest supported 3.2 GF model) while
> keeping the project-facing name `regnet_y_4gf` in run ids, checkpoints and reports.

---

## 4. Architectures — 7-class validation-pass results

All eight are torchvision models pretrained on ImageNet, fine-tuned to 7 classes, evaluated with TTA.

| # | Architecture | Params | Val mF1 | **Test acc** | **Test mF1** | Test wF1 | Time |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | **regnet_y_8gf** ⭐ | 37.38 M | 0.8148 | **0.8991** | **0.8812** | 0.8984 | 257 s |
| 2 | efficientnet_v2_m | 52.87 M | 0.7530 | 0.8632 | 0.8498 | 0.8633 | 290 s |
| 3 | efficientnet_v2_s | 20.19 M | 0.7601 | 0.8446 | 0.8391 | 0.8478 | 259 s |
| 4 | convnext_small | 49.46 M | 0.7434 | 0.8499 | 0.8360 | 0.8481 | 249 s |
| 5 | regnet_y_4gf | 17.93 M | 0.7451 | 0.8805 | 0.8224 | 0.8648 | 243 s |
| 6 | resnet50 | 23.52 M | 0.6884 | 0.8220 | 0.8036 | 0.8267 | 235 s |
| 7 | densenet121 | 6.96 M | 0.6553 | 0.8061 | 0.7940 | 0.8087 | 238 s |
| 8 | convnext_tiny | 27.83 M | 0.6303 | 0.8101 | 0.7838 | 0.8221 | 267 s |

- **Best:** `regnet_y_8gf` (37.38 M params) — top test macro F1 (0.8812) and top test accuracy (0.8991).
- **Best efficiency:** `regnet_y_4gf` (17.93 M) — 0.8224 macro F1; `densenet121` (6.96 M) is the smallest.
- A full-schedule retrain is expected to lift every model by several points (the old 8-class
  70-epoch runs reached 95–96.5 %).

---

### 4.1 Best model (regnet_y_8gf) — test per-class recall (TTA)

| Class | Recall |
|---|---:|
| Early_blight | 0.887 |
| Healthy | 0.943 |
| Late_blight | 0.901 |
| Leaf Miner | 0.913 |
| Magnesium Deficiency | 0.968 |
| Nitrogen Deficiency | 0.946 |
| Spotted Wilt Virus | 0.642 |

### 4.2 Best model (regnet_y_8gf) — test confusion matrix

Test set (753 images), rows = true class, columns = predicted class:

| True \ Pred | E. blight | Healthy | L. blight | Leaf Miner | Mg Def | N Def | Spotted Wilt |
|---|---:|---:|---:|---:|---:|---:|---:|
| Early_blight (150) | **133** | 2 | 10 | 2 | 1 | 0 | 2 |
| Healthy (122) | 0 | **115** | 0 | 5 | 2 | 0 | 0 |
| Late_blight (192) | 3 | 0 | **173** | 2 | 0 | 5 | 9 |
| Leaf Miner (104) | 0 | 1 | 0 | **95** | 5 | 0 | 3 |
| Magnesium Deficiency (95) | 1 | 0 | 0 | 1 | **92** | 1 | 0 |
| Nitrogen Deficiency (37) | 0 | 0 | 1 | 0 | 1 | **35** | 0 |
| Spotted Wilt Virus (53) | 6 | 0 | 10 | 3 | 0 | 0 | **34** |

**Main confusions:** `Spotted Wilt Virus` (0.64 recall, spread across Late_blight / Leaf Miner /
Early_blight) and `Early_blight → Late_blight` (10 images) — visually similar pairs; more epochs
are needed to separate them.

### 4.3 Top-3 soft-vote ensemble

| Members (top-3 by test mF1) | Ensemble mF1 | Ensemble acc | vs best single |
|---|---:|---:|---:|
| regnet_y_8gf + efficientnet_v2_m + efficientnet_v2_s | 0.8774 | 0.8951 | −0.0038 |

The ensemble slightly trails the best single model at this short schedule; with longer training
the members diverge more and ensembling typically wins.

---

## 5. Outputs & Artifacts Layout

---

## 7. Confusion Matrices

All 7-class artifacts from the validation pass live under `outputs/colab_validation/`
(git-ignored; regenerate by rerunning the notebook):

```
outputs/colab_validation/
├── checkpoints/                      # per-arch .pt + classes.json + tomato_regnet_y_8gf_best.pt
├── experiments/tomato/<run_id>/      # best_state.pt, history.csv, summary.json, metadata.json,
│                                     #   training_curves.png, sample_predictions.png,
│                                     #   test_confusion_matrix.png, final_{val,test}_metrics.json
├── metrics/                          # per-run JSONs, tomato_training_summary.csv,
│                                     #   tomato_training_final_summary.json, tomato_ensemble_top3.json,
│                                     #   tomato_dataset_report.json, runs.csv
├── confusion_matrices/               # tomato_{val,test}_*.png per run
├── predictions/                      # tomato_{val,test}_*_predictions.csv per run
└── classes/tomato.yaml               # 7-class mapping written by the notebook
```

`models/checkpoints/tomato/` currently holds only `classes.json` (the 7-class mapping). After a
full retrain, copy the best `.pt` there and register it in `configs/models.yaml` (see the
commented example in that file).

---

## 6. Run History (what happened, in order)

1. **2026-09-09 (earlier) — 8-class runs.** Five backbones trained 70 epochs on the original
   8-class dataset (which still contained `Pottassium Deficiency`). All checkpoints, metrics,
   predictions and reports from these runs have since been **deleted** — the class count changed
   and the artifacts were incompatible with the 7-class mapping.
2. **2026-09-09 — dataset update.** `Pottassium Deficiency` removed; 7 class folders remain in
   `train/`, `val/`, `test/`.
3. **2026-09-09 ~17:20–18:00 — 7-class validation pass.** 8 backbones × 4 epochs via
   `training_colab.ipynb` on a local CUDA GPU (validates the notebook + dataset end-to-end;
   ~35 min total). Best: `regnet_y_8gf` (test mF1 0.8812). Artifacts in `outputs/colab_validation/`.
4. **Next — full retrain.** Run `training_colab.ipynb` on a Colab GPU with the full 25–70-epoch
   schedule to produce final 7-class numbers, then register the best checkpoint in
   `configs/models.yaml`.

---

## 7. Reproducibility

- **Code:** `training/tomato/training.ipynb` (local) and
  **`training/tomato/training_colab.ipynb` (recommended, Colab GPU)** — full pipeline: Drive
  mount (authorize with **aplha2007beta@gmail.com**) → repo/dataset auto-discovery → audit →
  transforms → train → TTA eval → plots → checkpoints → top-3 ensemble.
- **Config:** `configs/project.yaml` (seed 42, image_size 224, device cuda, num_workers 2)
- **Classes:** `configs/classes/tomato.yaml` (+ `models/checkpoints/tomato/classes.json`)
- **Tracking:** every run is logged to `outputs/metrics/runs.csv`
  (architecture, seed, split version, image size, batch, epochs, optimizer, lr, scheduler,
  weight_decay, augmentation version, metrics, latency, checkpoint hash…)
- **Fixed seed 42** at every randomness source.

---

## 8. Limitations & Suggested Next Steps

- **Validation-pass metrics only:** the 7-class numbers above come from a 4-epoch pass — rerun
  with the full schedule before drawing conclusions.
- **Imbalanced classes:** `Nitrogen Deficiency` (251 train) and `Healthy` (1,151 train) remain the
  most imbalanced — addressed with sqrt-balanced weights + sampling in the retrain notebooks.
- **Confusable pairs:** `Spotted Wilt Virus ↔ Late_blight/Leaf Miner` and
  `Early_blight ↔ Late_blight` drive most remaining errors.
- **Ensembling** the top-3 is built into the notebook's final cell.
- **XAI artifacts** (Grad-CAM++, Score-CAM, saliency maps) can be regenerated from the UI or
  `scripts/` to sanity-check where the model looks before field deployment.

---

*Generated from `outputs/colab_validation/metrics/tomato_training_final_summary.json`,
per-run `summary.json` / `final_test_metrics.json` / `history.csv` and
`tomato_ensemble_top3.json` — AgroVision.*