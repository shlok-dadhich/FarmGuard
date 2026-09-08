from src.models.adapters._base import _TorchvisionAdapter


class DenseNet121Adapter(_TorchvisionAdapter):
    architecture = "densenet121"
    tv_name = "densenet121"

    def target_layers(self, m):
        try:
            return [m.features.denseblock4]
        except Exception:
            return [m]
