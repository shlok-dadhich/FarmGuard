"""Shared types."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Prediction:
    label: str
    confidence: float
    top_k: list = field(default_factory=list)
    model: str = ""
    crop: str = ""
    demo: bool = False


@dataclass
class RunRecord:
    timestamp: str = ""
    run_id: str = ""
    crop: str = ""
    architecture: str = ""
    seed: int = 0
    dataset_version: str = "v1"
    split_version: str = "v1"
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 0
    optimizer: str = ""
    learning_rate: float = 0.0
    scheduler: str = ""
    weight_decay: float = 0.0
    augmentation_version: str = ""
    train_loss: float = 0.0
    val_loss: float = 0.0
    accuracy: float = 0.0
    macro_f1: float = 0.0
    per_class_recall: dict = field(default_factory=dict)
    parameter_count: int = 0
    flops: float = 0.0
    training_time_s: float = 0.0
    inference_latency_ms: float = 0.0
    checkpoint_path: str = ""
