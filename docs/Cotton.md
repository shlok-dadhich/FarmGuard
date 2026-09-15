# Cotton - Leaf Disease Classification

**Full training and evaluation report**

- **Crop:** Cotton (`Cotton_data`)
- **Task:** 4-class leaf disease image classification
- **Latest training:** 2026-09-10 - DenseNet-121, up to 40 epochs, CUDA
- **Framework:** PyTorch and torchvision
- **Best model:** DenseNet-121
- **Best checkpoint:** [`models/checkpoints/cotton/best_densenet.pth`](../models/checkpoints/cotton/best_densenet.pth)

> This report documents the latest run produced by
> [`cotton_training.py`](../training/tomato/cotton/cotton_training.py). Unlike the tomato
> report, the cotton script trains one DenseNet-121 model; it does not run a multi-backbone
> comparison or experiment-tracking workflow.

---

## 1. Executive Summary

| Metric | Value |
|---|---:|
| **Test accuracy** | **99.00%** |
| **Test macro F1** | **99.09%** |
| Test macro precision / recall | 99.45% / 98.75% |
| Test weighted F1 | 99.00% |
| Best validation accuracy | 99.66% |
| Best validation epochs | 17 and 26 |
| Test loss | 0.2542 |
| Parameters | 6.96 M trainable |

The model was saved whenever validation accuracy improved. The training log reports early
stopping after epoch 27. The checkpoint was then reloaded before the held-out test evaluation.

---

## 2. Dataset

### 2.1 Source and structure

- Dataset root: `data/raw/Cotton_data/{train,val,test}/{class}/`
- The script expects pre-existing `train`, `val`, and `test` ImageFolder directories.
- Classes: `bacterial blight`, `curl virus`, `fussarium wilt`, and `healthy`
- Images are loaded with `torchvision.datasets.ImageFolder`.
- The script does not perform an integrity audit or create a new random split.

### 2.2 Split sizes

| Split | Images |
|---|---:|
| Train | 1,904 |
| Validation | 585 |
| Test | 598 |
| **Total** | **3,087** |

### 2.3 Class balance - on-disk counts

| Class | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| Bacterial blight | 372 | 131 | 136 | 639 |
| Curl virus | 298 | 96 | 97 | 491 |
| Fussarium wilt | 312 | 103 | 98 | 513 |
| Healthy | 922 | 255 | 267 | 1,444 |
| **Total** | **1,904** | **585** | **598** | **3,087** |

The training set is imbalanced: `healthy` contains 922 images, while `curl virus` contains 298.
The training loader addresses this with an inverse-frequency `WeightedRandomSampler`.

### 2.4 Augmentation and preprocessing

**Training transform:**

```text
RandomResizedCrop(224)
RandomHorizontalFlip()
ToTensor()
Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
```

**Validation and test transform:**

```text
Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize(ImageNet statistics)
```

Augmentation is applied online by the training `DataLoader`; no augmented image files are
written to disk.

---

## 3. Training Setup

| Setting | Value |
|---|---|
| Backbone | torchvision DenseNet-121 pretrained on ImageNet |
| Classification head | Replaced with `nn.Linear(in_features, 4)` |
| Input size | 224 x 224 |
| Epochs | 40 maximum |
| Completed epochs | 27 |
| Batch size | 32 |
| Optimizer | AdamW, learning rate `1e-4`, weight decay `0.05` |
| Scheduler | ReduceLROnPlateau, factor `0.5`, patience `10`, monitors validation loss |
| Loss | CrossEntropyLoss with label smoothing `0.05` |
| Sampling | Inverse-frequency `WeightedRandomSampler`, replacement enabled |
| AMP | CUDA automatic mixed precision |
| Early stopping | Patience `10`, monitors validation accuracy |
| DataLoader workers | 4 |
| Device | CUDA |

The script reports 6,957,956 trainable parameters. It does not set Python, NumPy, or PyTorch
random seeds, so repeated executions are not guaranteed to produce identical results.

---

## 4. Latest Model Results

This run contains one trained architecture, so there is no cross-backbone ranking table.

| Architecture | Params | Best val acc | Test acc | Test macro F1 | Test weighted F1 | Completed epochs |
|---|---:|---:|---:|---:|---:|---:|
| **DenseNet-121** | **6.96 M** | **0.9966** | **0.9900** | **0.9909** | **0.9900** | **27** |

### 4.1 Test per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Bacterial blight | 1.000 | 0.971 | 0.985 | 136 |
| Curl virus | 1.000 | 1.000 | 1.000 | 97 |
| Fussarium wilt | 1.000 | 0.980 | 0.990 | 98 |
| Healthy | 0.978 | 1.000 | 0.989 | 267 |
| **Macro average** | **0.995** | **0.988** | **0.991** | **598** |
| **Weighted average** | **0.990** | **0.990** | **0.990** | **598** |

### 4.2 Test confusion matrix

The generated confusion matrix is available at
[`outputs/figures/cotton/confusion_matrix.png`](../outputs/figures/cotton/confusion_matrix.png).
Rows are true classes and columns are predicted classes.

| True \\ Pred | Bacterial blight | Curl virus | Fussarium wilt | Healthy |
|---|---:|---:|---:|---:|
| Bacterial blight (136) | **132** | 0 | 0 | 4 |
| Curl virus (97) | 0 | **97** | 0 | 0 |
| Fussarium wilt (98) | 0 | 0 | **96** | 2 |
| Healthy (267) | 0 | 0 | 0 | **267** |

The six test errors were bacterial-blight leaves predicted as healthy (4) and fussarium-wilt
leaves predicted as healthy (2). No curl-virus or healthy test images were misclassified.

---

## 5. Training Curves

The complete training and validation curves are available at
[`outputs/figures/cotton/training_curves.png`](../outputs/figures/cotton/training_curves.png).

Selected logged epochs are shown below:

| Epoch | Train loss | Train acc | Val loss | Val acc |
|---:|---:|---:|---:|---:|
| 1 | 0.8660 | 0.7001 | 0.5047 | 0.8769 |
| 5 | 0.3517 | 0.9454 | 0.2836 | 0.9761 |
| 10 | 0.3146 | 0.9611 | 0.2510 | 0.9846 |
| 17 | 0.2767 | 0.9764 | 0.2367 | **0.9966** |
| 26 | 0.2659 | 0.9806 | 0.2361 | **0.9966** |
| 27 | 0.2547 | 0.9858 | 0.2346 | 0.9915 |

Validation accuracy improved rapidly during the first ten epochs and remained above 98% for
the remainder of the run. The best checkpoint was retained from the first 99.66% validation
accuracy result at epoch 17.

---

## 6. Outputs, Logs, and Artifacts

The latest run generated the following local artifacts:

| Artifact | Path |
|---|---|
| Training script | [`cotton_training.py`](../training/tomato/cotton/cotton_training.py) |
| Training log | [`cotton_training.log`](../outputs/logs/cotton_training.log) |
| Best checkpoint | [`best_densenet.pth`](../models/checkpoints/cotton/best_densenet.pth) |
| Classification report | [`test_classification_report.json`](../outputs/metrics/cotton/test_classification_report.json) |
| Training curves | [`training_curves.png`](../outputs/figures/cotton/training_curves.png) |
| Confusion matrix | [`confusion_matrix.png`](../outputs/figures/cotton/confusion_matrix.png) |
| XAI samples | [`outputs/figures/cotton/xai/`](../outputs/figures/cotton/xai/) |

The XAI directory contains five Integrated Gradients outputs: `sample_0.png` through
`sample_4.png`. The script currently passes normalized tensors directly to `imshow`, which
causes a clipping warning; the input panels should be inverse-normalized for a faithful display.

---

## 7. Run History

1. The script resolved the repository root and loaded the existing cotton train, validation,
   and test folders.
2. DenseNet-121 was fine-tuned with weighted sampling and CUDA AMP.
3. Validation accuracy peaked at 99.66% at epoch 17, then matched that value again at epoch 26.
4. The training log reported early stopping after epoch 27.
5. The best checkpoint was reloaded and evaluated on 598 held-out test images.
6. The run produced 99.00% test accuracy and 99.09% test macro F1.

---

## 8. Reproducibility

- **Code:** [`training/tomato/cotton/cotton_training.py`](../training/tomato/cotton/cotton_training.py)
- **Dataset root:** `data/raw/Cotton_data/`
- **Classes:** derived from the training `ImageFolder` directory names
- **Checkpoint selection:** highest validation accuracy
- **Metrics:** [`test_classification_report.json`](../outputs/metrics/cotton/test_classification_report.json)
- **Run log:** [`cotton_training.log`](../outputs/logs/cotton_training.log)

Run from the repository root:

```powershell
python training/tomato/cotton/cotton_training.py
```

The script requires the project dependencies, the prepared cotton split directories, and a
CUDA-capable environment for the documented device configuration. On a CPU-only machine,
the script selects CPU for the model but the CUDA AMP configuration may require adjustment.

---

## 9. Limitations and Suggested Next Steps

- **No fixed seed:** repeated runs can differ because random seeds are not configured.
- **Early-stopping record:** the script declares patience `10`, but the available log reports
  stopping immediately after epoch 27, so the exact patience-counter state is not independently
  recoverable from the log.
- **Single-model evidence:** this report does not establish that DenseNet-121 is superior to
  other backbones because the attached script does not train comparison models.
- **Dataset shift:** the held-out test set comes from the same prepared dataset family; field
  performance under different cultivars, lighting, cameras, and backgrounds needs validation.
- **Class imbalance:** weighted sampling is used, but the healthy class remains much larger than
  the disease classes.
- **XAI display:** inverse-normalize images before plotting to avoid clipped input visualizations.
- **API modernization:** `pretrained=True` and the imported legacy AMP symbols can be updated to
  current torchvision and torch APIs in a separate code change.

---

*Updated from the cotton training script, `cotton_training.log`, the generated classification
report, split counts, checkpoint, curves, confusion matrix, and Integrated Gradients artifacts.*
