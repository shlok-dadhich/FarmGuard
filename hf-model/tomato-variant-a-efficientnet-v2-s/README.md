---
license: mit
library_name: pytorch
tags:
- tomato
- plant-disease
- image-classification
- computer-vision
- pytorch
- torchvision
- efficientnet-v2
pipeline_tag: image-classification
---

# Tomato Variant A - EfficientNet-V2-S

A 7-class tomato leaf disease image classification model trained with PyTorch and torchvision.

## Model

* Architecture: EfficientNet-V2-S
* Backbone: ImageNet-1K pretrained
* Input: 224 x 224 RGB
* Classes: 7
* Framework: PyTorch / torchvision
* Training seed: 42
* Best epoch: 69
* Run ID: `efficientnet_v2_s_seed42_20260910-042630`

## Classes

| ID | Class                |
| -: | -------------------- |
|  0 | Early_blight         |
|  1 | Healthy              |
|  2 | Late_blight          |
|  3 | Leaf Miner           |
|  4 | Magnesium Deficiency |
|  5 | Nitrogen Deficiency  |
|  6 | Spotted Wilt Virus   |

## Results

| Metric                   |      Score |
| ------------------------ | ---------: |
| Test Accuracy            | **97.87%** |
| Test Macro F1            | **97.20%** |
| Test Macro Precision     |     96.87% |
| Test Macro Recall        |     97.58% |
| Test Weighted F1         |     97.88% |
| Best Validation Accuracy |     95.95% |
| Best Validation Macro F1 |     94.80% |

## Dataset

Dataset ID: `tomato_variant_a_multiclass`

* Train: 6,637 images
* Validation: 888 images
* Test: 752 images
* Total: 8,277 images

The dataset uses pre-existing train/validation/test splits. No random re-split was performed.

The `Pottassium Deficiency` class was removed before this final 7-class training run. Older 8-class checkpoints are incompatible with this class mapping.

## Preprocessing

Validation and test:

```text
Resize(256)
CenterCrop(224)
ToTensor()
Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)
```

Training used online augmentation including random resized crops, flips, rotation, affine transforms, color jitter, and random grayscale.

## Training

* Optimizer: AdamW
* Learning rate: `1e-4`
* Weight decay: `0.05`
* Maximum epochs: 70
* Batch size: 16
* Loss: weighted CrossEntropyLoss
* Label smoothing: `0.05`
* Scheduler: cosine annealing
* Sampling: sqrt-balanced WeightedRandomSampler
* Backbone: ImageNet-1K pretrained
* Fine-tuning: frozen initially, then fully unfrozen
* AMP: CUDA mixed precision
* Seed: `42`

## Usage

Install dependencies:

```bash
pip install torch torchvision pillow
```

Run inference:

```bash
python inference.py path/to/tomato_leaf.jpg
```

Example output:

```text
Prediction: Healthy
Confidence: 93.45%
```

## Test-set Per-Class Recall

| Class                | Recall |
| -------------------- | -----: |
| Early_blight         | 100.0% |
| Healthy              |  95.1% |
| Late_blight          |  98.4% |
| Leaf Miner           |  97.1% |
| Magnesium Deficiency | 100.0% |
| Nitrogen Deficiency  | 100.0% |
| Spotted Wilt Virus   |  92.5% |

## Limitations

The weakest test-set class was `Spotted Wilt Virus`, with approximately 92.5% recall.

The main observed confusions were Spotted Wilt Virus with Late_blight and Leaf Miner, and Healthy with Leaf Miner.

The model was trained and evaluated on a specific tomato leaf image dataset. Performance may differ under different cameras, lighting conditions, cultivars, field environments, image quality, or unseen diseases.

This model is intended for research and experimental use and should not be treated as a definitive agricultural diagnosis.

## Reproducibility

Original training run:

`efficientnet_v2_s_seed42_20260910-042630`

Framework:

PyTorch + torchvision

The repository contains the trained checkpoint, class mapping, metadata, and inference script.

## License

MIT
