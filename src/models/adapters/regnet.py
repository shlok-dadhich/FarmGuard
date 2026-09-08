"""RegNetY-4GF adapter: tries timm regnety_040, then torchvision candidates."""
from __future__ import annotations
import torch.nn as nn
from src.models.adapters._base import _TorchvisionAdapter


class RegNetY4GFAdapter(_TorchvisionAdapter):
    architecture = "regnet_y_4gf"
    tv_name = "regnet_y_3_2gf"  # closest torchvision sibling; timm preferred

    def build(self, num_classes: int, pretrained: bool = False):
        # Prefer timm regnety_040 (== 4GF) when available
        try:
            import timm
            m = timm.create_model("regnety_040", pretrained=pretrained, num_classes=num_classes)
            return m
        except Exception:
            pass
        for cand in ("regnet_y_3_2gf", "regnet_y_800mf", "regnet_y_400mf"):
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
