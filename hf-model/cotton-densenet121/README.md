---
license: mit
library_name: pytorch
tags:
- cotton
- plant-disease
- image-classification
- computer-vision
- pytorch
- torchvision
- densenet
pipeline_tag: image-classification
---

# FarmGuard Cotton DenseNet-121

A 4-class cotton leaf disease image-classification model trained with PyTorch and torchvision.

## Model

- Architecture: DenseNet-121 with an ImageNet-pretrained backbone
- Input: 224 x 224 RGB image
- Classes: 4
- Test accuracy: 98.9967%
- Test macro F1: 99.0914%
- Best validation accuracy: 99.66%

## Classes

| ID | Class |
| ---: | --- |
| 0 | bacterial blight |
| 1 | curl virus |
| 2 | fussarium wilt |
| 3 | healthy |

The class order is stored in `classes.json`.

## Preprocessing

```text
Resize(256)
CenterCrop(224)
ToTensor()
Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
```

## Usage

Install dependencies:

```bash
pip install torch torchvision pillow
```

Run inference on an image:

```bash
python inference.py path/to/cotton_leaf.jpg
```

## Limitations

This model was evaluated on a prepared held-out dataset and may perform differently with other cultivars, cameras, lighting, backgrounds, or unseen diseases. It is intended for research and experimental use and is not a definitive agricultural diagnosis.

## License

MIT
