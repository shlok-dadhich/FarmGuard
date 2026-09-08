from src.models.adapters._base import _TorchvisionAdapter


class ResNet50Adapter(_TorchvisionAdapter):
    architecture = "resnet50"
    tv_name = "resnet50"

    def target_layers(self, m):
        try:
            return [m.layer4[-1]]
        except Exception:
            return [m]
