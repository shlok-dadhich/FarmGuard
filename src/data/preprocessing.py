"""Preprocessing (deterministic val/test) + configurable train augmentation."""
from __future__ import annotations
from PIL import Image


def build_transforms(image_size=224, train=True, aug_version="aug-v1"):
    try:
        from torchvision import transforms
    except ImportError:
        return None
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if not train:
        return transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(image_size),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
    # shared, biologically-safe pipeline (no vertical flip by default)
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
