from src.models.adapters._base import _TorchvisionAdapter


class EfficientNetV2SAdapter(_TorchvisionAdapter):
    architecture = "efficientnet_v2_s"
    tv_name = "efficientnet_v2_s"

    def target_layers(self, m):
        try:
            return [m.features[-1]]
        except Exception:
            return [m]
