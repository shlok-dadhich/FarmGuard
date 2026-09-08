"""Torchvision/timm-backed adapters. They NEVER fabricate weights; missing checkpoints raise clearly."""
from __future__ import annotations
import torch.nn as nn
from src.models.interface import ModelAdapter


class _TorchvisionAdapter(ModelAdapter):
    tv_name: str = ""
    timm_name: str = ""
    _size: int = 224

    def _make(self, num_classes, pretrained=False):
        try:
            import torchvision.models as M
            fn = getattr(M, self.tv_name, None)
            if fn is None:
                raise RuntimeError(f"torchvision has no {self.tv_name}")
            try:
                m = fn(weights="DEFAULT" if pretrained else None)
            except TypeError:
                m = fn(pretrained=pretrained)
            # replace head
            if hasattr(m, "fc") and isinstance(m.fc, nn.Linear):
                m.fc = nn.Linear(m.fc.in_features, num_classes)
            elif hasattr(m, "classifier"):
                c = m.classifier
                if isinstance(c, nn.Linear):
                    m.classifier = nn.Linear(c.in_features, num_classes)
                elif isinstance(c, nn.Sequential):
                    last = [i for i, l in enumerate(c) if isinstance(l, nn.Linear)]
                    if last:
                        idx = last[-1]
                        c[idx] = nn.Linear(c[idx].in_features, num_classes)
            elif hasattr(m, "head") and isinstance(m.head, nn.Linear):
                m.head = nn.Linear(m.head.in_features, num_classes)
            return m
        except Exception as e:
            raise RuntimeError(f"Could not construct {self.architecture}: {e}. Install torchvision/timm.") from e

    def build(self, num_classes: int, pretrained: bool = False):
        return self._make(num_classes, pretrained)

    def parameter_count(self, module) -> int:
        return sum(p.numel() for p in module.parameters())

    def input_size(self) -> int:
        return self._size
