"""Registry: maps architecture key -> adapter dotted path. No weights fabricated here."""
from __future__ import annotations

MODEL_REGISTRY = {
    "efficientnet_v2_s": "src.models.adapters.efficientnet:EfficientNetV2SAdapter",
    "resnet50": "src.models.adapters.resnet:ResNet50Adapter",
    "convnext_tiny": "src.models.adapters.convnext:ConvNeXtTinyAdapter",
    "regnet_y_4gf": "src.models.adapters.regnet:RegNetY4GFAdapter",
    "densenet121": "src.models.adapters.densenet:DenseNet121Adapter",
    "mock_demo": "src.models.mock:MockDemoAdapter",
}
