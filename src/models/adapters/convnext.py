from src.models.adapters._base import _TorchvisionAdapter


class ConvNeXtTinyAdapter(_TorchvisionAdapter):
    architecture = "convnext_tiny"
    tv_name = "convnext_tiny"

    def target_layers(self, m):
        try:
            return [m.features[-1]]
        except Exception:
            return [m]
