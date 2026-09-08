"""Saliency maps (Vanilla/Simple saliency).

Works for any differentiable model — no target layer required. The map is the
channel-wise max absolute gradient of the target logit w.r.t. the input image.

Interface matches the CAM methods: ``(saliency_map, class_idx)`` where the map is
a float np.ndarray in [0, 1] with the same HxW as the input tensor.
"""
from __future__ import annotations

import numpy as np
import torch


class Saliency:
    """Vanilla saliency: |d logit_c / d input| maxed across channels."""

    def __init__(self, model):
        self.model = model

    @torch.enable_grad()
    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)
        self.model.zero_grad()
        self.model.eval()
        logits = self.model(x)
        c = int(logits.argmax(1)[0]) if class_idx is None else class_idx
        if c >= logits.shape[1]:
            raise ValueError(f"class_idx {c} out of range (model has {logits.shape[1]} classes)")
        logits[0, c].backward()
        grad = x.grad if x.grad is not None else x.grad_fn  # type: ignore[union-attr]
        if grad is None:
            raise RuntimeError(
                "WHAT: saliency failed — no gradient available\n"
                "WHY: the model does not propagate gradients to the input (e.g. "
                "non-differentiable op or inference-only wrapper)\n"
                "HOW: XAI method unavailable for this model — use Grad-CAM++/Score-CAM, "
                "which operate on feature maps instead."
            )
        sal = grad.abs().max(dim=1)[0][0].cpu().numpy()  # type: ignore[union-attr]
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
        return sal, c