"""RegNetY-4GF adapter.

The project-facing name is ``regnet_y_4gf``, but torchvision has no 4 GF model.
The training notebook (training/tomato/training.ipynb) trained on torchvision's
``regnet_y_3_2gf`` (retaining the 4GF-facing name), so the adapter builds the
SAME architecture by default so trained checkpoints load exactly. timm
``regnety_040`` (true 4 GF) is only a fallback when torchvision is unavailable.
"""
from __future__ import annotations
import torch.nn as nn
from src.models.adapters._base import _TorchvisionAdapter


class RegNetY4GFAdapter(_TorchvisionAdapter):
    architecture = "regnet_y_4gf"
    tv_name = "regnet_y_3_2gf"  # architecture used by the trained checkpoints

    def build(self, num_classes: int, pretrained: bool = False):
        # 1) torchvision regnet_y_3_2gf FIRST: matches the trained checkpoint exactly.
        try:
            import torchvision.models as M
            fn = getattr(M, "regnet_y_3_2gf")
            try:
                m = fn(weights="DEFAULT" if pretrained else None)
            except TypeError:
                m = fn(pretrained=pretrained)
            m.fc = nn.Linear(m.fc.in_features, num_classes)
            return m
        except Exception:
            pass
        # 2) timm regnety_040 (true 4 GF) fallback when torchvision is unavailable.
        try:
            import timm
            m = timm.create_model("regnety_040", pretrained=pretrained, num_classes=num_classes)
            return m
        except Exception:
            pass
        for cand in ("regnet_y_800mf", "regnet_y_400mf"):
            try:
                import torchvision.models as M
                fn = getattr(M, cand)
                try:
                    m = fn(weights="DEFAULT" if pretrained else None)
                except TypeError:
                    m = fn(pretrained=pretrained)
                m.fc = nn.Linear(m.fc.in_features, num_classes)
                return m
            except Exception:
                continue
        return super().build(num_classes, pretrained)

    def target_layers(self, module):
        for attr in ("trunk_output", "trunk", "features"):
            if hasattr(module, attr):
                return [getattr(module, attr)]
        return [module]
