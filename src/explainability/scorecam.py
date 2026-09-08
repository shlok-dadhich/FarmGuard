"""Score-CAM (mask-perturbation, subsampled for CPU)."""
from __future__ import annotations
import numpy as np, torch
import torch.nn.functional as _F
class ScoreCAM:
    def __init__(self, model, target_layer, max_masks=16):
        self.model = model; self.layer = target_layer; self.max_masks = max_masks
        self.acts = None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o.detach()))
    @torch.no_grad()
    def __call__(self, x, class_idx=None):
        self.model.eval()
        base = self.model(x)
        c = int(base.argmax(1)[0]) if class_idx is None else class_idx
        A = self.acts[0]  # C,H,W
        C = min(A.shape[0], self.max_masks)
        H, W = x.shape[2], x.shape[3]
        cams = []
        s0 = base.softmax(1)[0, c].item()
        for k in range(C):
            m = A[k:k+1]
            m = (m - m.min()) / (m.max() - m.min() + 1e-8)
            up = _F.interpolate(m.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
            out = self.model(x * up)
            cams.append(max(0.0, out.softmax(1)[0, c].item() - 0.0) * up[0].cpu().numpy())
        cam = np.sum(cams, axis=0) if cams else np.zeros((H, W))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, c
