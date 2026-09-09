"""Grad-CAM++ (simplified alpha weighting, works for any conv target layer)."""
from __future__ import annotations
import numpy as np, torch
class GradCAMpp:
    def __init__(self, model, target_layer):
        self.model = model; self.layer = target_layer
        self.acts = None; self.grads = None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o.detach()))
        target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0].detach()))
    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        self.model.zero_grad()
        logits = self.model(x)
        c = int(logits.argmax(1)[0]) if class_idx is None else class_idx
        logits[0, c].backward()
        A, G = self.acts[0], self.grads[0]  # C,H,W
        # Grad-CAM++ weights: alpha_ij = g^2 / (2 g^2 + sum(A * g^3)), w_k = sum_ij alpha * relu(g)
        g2, g3 = G.pow(2), G.pow(3)
        alpha = g2 / (2 * g2 + (A * g3).sum(dim=(1, 2), keepdim=True) + 1e-8)
        w = (alpha * torch.relu(G)).sum(dim=(1, 2))
        cam = torch.relu((w[:, None, None] * A).sum(0)).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, c
