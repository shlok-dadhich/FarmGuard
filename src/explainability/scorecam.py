"""Score-CAM (mask-perturbation attribution).

Each activation channel is upsampled to the input size, used as a soft mask on
the input, and weighted by the class score change it causes. Masks are processed
in batches (``batch_size``) so the method stays fast on both CUDA and CPU.

Interface matches the other CAM methods: ``(saliency_map, class_idx)`` where the
map is a float np.ndarray in [0, 1] with the same HxW as the input tensor.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as _F


class ScoreCAM:
    def __init__(self, model, target_layer, max_masks: int = 16, batch_size: int = 8):
        self.model = model
        self.layer = target_layer
        self.max_masks = max_masks
        self.batch_size = max(1, batch_size)
        self.acts = None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o.detach()))

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        self.model.eval()
        base = self.model(x)
        c = int(base.argmax(1)[0]) if class_idx is None else class_idx
        A = self.acts[0]  # (C, H, W)
        C = min(A.shape[0], self.max_masks)
        H, W = x.shape[2], x.shape[3]

        cams = np.zeros((H, W), dtype=np.float64)
        pending: list[torch.Tensor] = []

        def _flush():
            nonlocal cams, pending
            if not pending:
                return
            batch = torch.cat(pending, dim=0)  # (B, 1, H, W) masked inputs' masks
            # x: (1, 3, H, W) * batch: (B, 1, H, W) -> (B, 3, H, W) one batched forward
            outs = self.model(x * batch)
            scores = torch.relu(outs.softmax(1)[:, c])
            for j, up in enumerate(pending):
                cams += float(scores[j]) * up[0, 0].cpu().numpy()
            pending = []

        for k in range(C):
            m = A[k : k + 1]
            m = (m - m.min()) / (m.max() - m.min() + 1e-8)
            # keep the (1, 1, H, W) mask shape so a batch stacks to (B, 1, H, W)
            up = _F.interpolate(m.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)
            pending.append(up)
            if len(pending) >= self.batch_size:
                _flush()
        _flush()

        cam = (cams - cams.min()) / (cams.max() - cams.min() + 1e-8)
        return cam, c