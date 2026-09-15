# 🍅 Tomato — Variant A Multiclass Leaf Disease Classification

**Full training & evaluation report**

- **Crop:** Tomato (`tomato_variant_a_multiclass`)
- **Task:** 7-class leaf disease image classification
- **Latest training:** 2026-09-10 — five pretrained backbones, up to 70 epochs, CUDA
- **Framework:** PyTorch (torchvision backbones) · Seed `42`
- **Best model:** `efficientnet_v2_s` — run `efficientnet_v2_s_seed42_20260910-042630`
- **Best checkpoint:** `models/checkpoints/tomato/tomato_efficientnet_v2_s_best.pt`

> ℹ️ **Dataset update:** the `Pottassium Deficiency` disease class was **removed** from the dataset.
> Everything below reflects the current **7-class** dataset. The earlier 8-class checkpoints,
> metrics and reports were deleted because they are incompatible with the 7-class mapping.

---

## 1. Executive Summary

| Metric (best model) | Value |
|---|---:|
| **Test accuracy** | **97.87 %** |
| **Test macro F1** | **97.20 %** |
| Test macro precision / recall | 96.87 % / 97.58 % |
| Test weighted F1 | 97.88 % |
| Val macro F1 (best epoch) | 94.80 % |
| Parameters | 20.19 M |

> These are the latest 70-epoch local training results. The comparison selected the best run by
> test macro F1, so a final scientific benchmark should select using validation macro F1 first.

All five models used the notebook recipe (seed 42, batch 16, AdamW `lr=1e-4`,
`weight_decay=0.05`, cosine annealing, class weights + weighted sampler, label smoothing,
online image augmentation, and freeze-then-unfreeze). The best checkpoint was selected by
**highest test macro F1**.

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
| Train | 6,637 |
| Validation | 888 |
| Test | 752 |
| **Total** | **8,277** |

### 2.3 Class balance — before augmentation (on-disk counts)

| Class | Train | Val | Test | Total |
|---|---|---:|---:|---:|
| Early_blight | 1,347 | 99 | 150 | 1,596 |
| Healthy | 1,151 | 43 | 122 | 1,316 |
| Late_blight | 1,632 | 180 | 191 | 2,003 |
| Leaf Miner | 716 | 204 | 104 | 1,024 |
| Magnesium Deficiency | 654 | 187 | 95 | 936 |
| Nitrogen Deficiency | 251 | 72 | 37 | 360 |
| Spotted Wilt Virus | 886 | 103 | 53 | 1,042 |
| **Total** | **6,637** | **888** | **752** | **8,277** |

The dataset is **mildly imbalanced** — `Nitrogen Deficiency` (251 train) and `Healthy`
(1,151 train) are the smallest and largest classes (≈4.6× spread).

### 2.4 Augmentation — "before" vs "after"

**Augmentation is applied on-the-fly (online)** inside the training `DataLoader`; **no new image
files are generated on disk**. So:

- **Before augmentation:** 8,277 images (6,637 train / 888 val / 752 test) — exactly the counts in §2.3.
- **After augmentation:** the on-disk count stays **8,277**, but every epoch the model sees a
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
| Epochs | 70 maximum per backbone |
| Batch size | 16 (train) / 32 (val & test) |
| Optimizer | AdamW, lr = 1e-4, weight_decay = 0.05 |
| Scheduler | Cosine annealing |
| Loss | CrossEntropyLoss(sqrt-balanced weights, label_smoothing=0.05) |
| Sampling | WeightedRandomSampler (sqrt mode, replacement=True, seed 42) |
| Regularization | Label smoothing `0.05` |
| Fine-tuning | backbone frozen first → unfreeze all at epoch ≥ 2 |
| AMP | torch.amp autocast + GradScaler (CUDA) |
| Early stopping | patience = 25, min_delta = 0.001, monitor = **val macro F1** |
| Evaluation | Deterministic validation/test preprocessing |
| Seed | 42 (python, numpy, torch, CUDA, sampler) |
| Determinism | cudnn deterministic on, benchmark off |

> **Compatibility note:** torchvision has no `regnet_y_4gf`; the notebook maps
> `regnet_y_4gf → torchvision.models.regnet_y_3_2gf` (the closest supported 3.2 GF model) while
> keeping the project-facing name `regnet_y_4gf` in run ids, checkpoints and reports.

---

## 4. Architectures — latest 70-epoch results

The table shows the latest successful run for each architecture, ranked by test macro F1.

| # | Architecture | Params | Best epoch | **Test acc** | **Test mF1** | Test wF1 | Time |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | **efficientnet_v2_s** | 20.19 M | 69 | **0.9787** | **0.9720** | 0.9788 | 3,621 s |
| 2 | densenet121 | 6.96 M | 69 | 0.9774 | 0.9707 | 0.9773 | 3,754 s |
| 3 | resnet50 | 23.52 M | 54 | 0.9774 | 0.9706 | 0.9772 | 3,199 s |
| 4 | regnet_y_4gf | 17.93 M | 57 | 0.9761 | 0.9700 | 0.9761 | 3,643 s |
| 5 | convnext_tiny | 27.83 M | 57 | 0.9668 | 0.9590 | 0.9667 | 3,273 s |

- **Best:** `efficientnet_v2_s` — top test macro F1 (0.9720) and top test accuracy (0.9787).
- **Best parameter efficiency:** `densenet121` — 6.96 M parameters and 0.9707 test macro F1.
- The models are close: the winner leads `densenet121` by only 0.13 percentage points in macro F1.

---

### 4.1 Best model (efficientnet_v2_s) — test per-class recall

| Class | Recall |
|---|---:|
| Early_blight | 1.000 |
| Healthy | 0.951 |
| Late_blight | 0.984 |
| Leaf Miner | 0.971 |
| Magnesium Deficiency | 1.000 |
| Nitrogen Deficiency | 1.000 |
| Spotted Wilt Virus | 0.925 |

### 4.2 Best model (efficientnet_v2_s) — test confusion matrix

Test set (752 images), rows = true class, columns = predicted class:

| True \ Pred | E. blight | Healthy | L. blight | Leaf Miner | Mg Def | N Def | Spotted Wilt |
|---|---:|---:|---:|---:|---:|---:|---:|
| Early_blight (150) | **150** | 0 | 0 | 0 | 0 | 0 | 0 |
| Healthy (122) | 0 | **116** | 0 | 5 | 0 | 0 | 1 |
| Late_blight (191) | 0 | 0 | **188** | 0 | 0 | 2 | 1 |
| Leaf Miner (104) | 0 | 0 | 0 | **101** | 0 | 0 | 3 |
| Magnesium Deficiency (95) | 0 | 0 | 0 | 0 | **95** | 0 | 0 |
| Nitrogen Deficiency (37) | 0 | 0 | 0 | 0 | 0 | **37** | 0 |
| Spotted Wilt Virus (53) | 0 | 0 | 2 | 2 | 0 | 0 | **49** |

**Main confusion:** `Spotted Wilt Virus` is still the weakest class, with errors to `Late_blight`
and `Leaf Miner`. `Healthy` also has five predictions as `Leaf Miner`.

### 4.3 Ensemble status

No ensemble was generated by the latest local five-model run. The older top-3 ensemble remains
under `outputs/colab_validation/` and is not comparable to this latest table.

---

## 5. Training Curves

Each completed run saves its per-epoch history to
`outputs/experiments/tomato/<run_id>/history.csv` with training loss, validation loss,
validation accuracy, validation macro precision/recall/F1, weighted F1, and sample count.
The same run folder contains `training_curves.png` and `sample_predictions.png`.

| Model | Training curves | Sample predictions | History (CSV) |
|---|---|---|---|
| **efficientnet_v2_s** ⭐ | [training_curves.png](../outputs/experiments/tomato/efficientnet_v2_s_seed42_20260910-042630/training_curves.png) | [sample_predictions.png](../outputs/experiments/tomato/efficientnet_v2_s_seed42_20260910-042630/sample_predictions.png) | [history.csv](../outputs/experiments/tomato/efficientnet_v2_s_seed42_20260910-042630/history.csv) |
| resnet50 | [training_curves.png](../outputs/experiments/tomato/resnet50_seed42_20260910-052653/training_curves.png) | [sample_predictions.png](../outputs/experiments/tomato/resnet50_seed42_20260910-052653/sample_predictions.png) | [history.csv](../outputs/experiments/tomato/resnet50_seed42_20260910-052653/history.csv) |
| convnext_tiny | [training_curves.png](../outputs/experiments/tomato/convnext_tiny_seed42_20260910-062014/training_curves.png) | [sample_predictions.png](../outputs/experiments/tomato/convnext_tiny_seed42_20260910-062014/sample_predictions.png) | [history.csv](../outputs/experiments/tomato/convnext_tiny_seed42_20260910-062014/history.csv) |
| regnet_y_4gf | [training_curves.png](../outputs/experiments/tomato/regnet_y_4gf_seed42_20260910-071448/training_curves.png) | [sample_predictions.png](../outputs/experiments/tomato/regnet_y_4gf_seed42_20260910-071448/sample_predictions.png) | [history.csv](../outputs/experiments/tomato/regnet_y_4gf_seed42_20260910-071448/history.csv) |
| densenet121 | [training_curves.png](../outputs/experiments/tomato/densenet121_seed42_20260910-081532/training_curves.png) | [sample_predictions.png](../outputs/experiments/tomato/densenet121_seed42_20260910-081532/sample_predictions.png) | [history.csv](../outputs/experiments/tomato/densenet121_seed42_20260910-081532/history.csv) |

Example from the best run (`efficientnet_v2_s_seed42_20260910-042630`): training loss drops
from `1.610` at epoch 1 to `0.242` at epoch 70. Validation macro F1 rises from `0.183` at
epoch 1 to a best of `0.948` at epoch 69, then finishes at `0.945` in epoch 70.

---

## 6. Outputs, Logs, and Artifacts

Yes. The latest successful runs saved the training and evaluation outputs. Each completed run
folder contains `history.csv`, `training_curves.png`, `sample_predictions.png`,
`test_confusion_matrix.png`, `final_val_metrics.json`, `final_test_metrics.json`, `summary.json`,
`metadata.json`, `training_config.json`, and `best_state.pt`.

The latest local artifacts are organized as follows:

```
outputs/experiments/tomato/<run_id>/  # histories, curves, summaries, predictions and checkpoints
outputs/metrics/                       # aggregate CSV/JSON metrics and runs.csv
outputs/logs/tomato_training_main.log  # training log
outputs/confusion_matrices/            # validation/test confusion-matrix PNGs
outputs/predictions/                   # validation/test prediction CSVs
models/checkpoints/tomato/             # model checkpoints and classes.json
```

The filesystem audit found 14 confusion-matrix PNGs, 14 prediction CSVs, 24 JSON metrics files,
one training log, and eight `.pt` checkpoint files. The incomplete earlier
`convnext_tiny_seed42_20260909-222204` folder contains only `history.csv` and is excluded from
the comparison. The latest local run did not generate an ensemble; the older ensemble is under
`outputs/colab_validation/`.

---

## 7. Run History (what happened, in order)

1. The original eight-class artifacts are obsolete because `Pottassium Deficiency` was removed.
2. The earlier four-epoch, eight-backbone validation pass is preserved under
  `outputs/colab_validation/`.
3. The latest local CUDA runs completed five architectures for up to 70 epochs. The latest
  successful run was `densenet121_seed42_20260910-081532`.
4. The registered best model is `efficientnet_v2_s`, recorded in
  `outputs/metrics/tomato_training_final_summary.json`.

---

## 8. Reproducibility

- **Code:** `training/tomato/training.ipynb` — repository/data setup, audit, transforms, training,
  evaluation, plots, checkpoints, and inference smoke test.
- **Config:** `configs/project.yaml` (seed 42, image_size 224, device cuda, num_workers 2)
- **Classes:** `configs/classes/tomato.yaml` (+ `models/checkpoints/tomato/classes.json`)
- **Tracking:** every run is logged to `outputs/metrics/runs.csv`
  (architecture, seed, split version, image size, batch, epochs, optimizer, lr, scheduler,
  weight_decay, augmentation version, metrics, latency, checkpoint hash…)
- **Fixed seed 42** at every randomness source.
- **Latest best run artifacts:** `outputs/experiments/tomato/efficientnet_v2_s_seed42_20260910-042630/`

---

## 9. Limitations & Suggested Next Steps

- **Model selection:** the comparison selected using test macro F1. For a final benchmark, select
  using validation macro F1 and report the held-out test result once.
- **Imbalanced classes:** `Nitrogen Deficiency` (251 train) and `Healthy` (1,151 train) remain the
  most imbalanced — addressed with sqrt-balanced weights + sampling in the retrain notebooks.
- **Confusable pairs:** `Spotted Wilt Virus ↔ Late_blight/Leaf Miner` and
  `Early_blight ↔ Late_blight` drive most remaining errors.
- **Ensembling:** no ensemble was generated by this latest local run.
- **XAI artifacts** (Grad-CAM++, Score-CAM, saliency maps) can be regenerated from the UI or
  `scripts/` to sanity-check where the model looks before field deployment.

---

*Updated from `outputs/metrics/runs.csv`, `tomato_training_final_summary.json`, the winning
run's `final_val_metrics.json` and `final_test_metrics.json`, per-run histories, and generated
artifact directories.*