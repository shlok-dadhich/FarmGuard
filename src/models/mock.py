"""Deterministic tiny mock model for DEMO/plumbing only. NEVER a research result."""
from __future__ import annotations
import torch
import torch.nn as nn
from src.models.interface import ModelAdapter


class TinyMockNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        torch.manual_seed(0)
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16 * 16, num_classes))
        # deterministic init
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.constant_(m.weight, 0.05)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x))


class MockDemoAdapter(ModelAdapter):
    architecture = "mock_demo"

    def build(self, num_classes: int, pretrained: bool = False):
        return TinyMockNet(num_classes)

    def target_layers(self, module):
        return [module.features[-3]]  # last conv

    def parameter_count(self, module) -> int:
        return sum(p.numel() for p in module.parameters())

    def input_size(self) -> int:
        return 64
