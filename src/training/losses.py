from __future__ import annotations
import torch.nn as nn
def build_loss(name="cross_entropy", label_smoothing=0.0, weight=None):
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=weight)
