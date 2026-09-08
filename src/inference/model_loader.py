from __future__ import annotations
from src.models.factory import build_model, load_checkpoint_weights
def load_model(arch, num_classes, checkpoint=None, device="cpu", pretrained=False):
    m = build_model(arch, num_classes, pretrained=pretrained)
    if checkpoint: load_checkpoint_weights(m, checkpoint, device)
    return m.eval().to(device)
