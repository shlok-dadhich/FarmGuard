"""Model adapter contract. External teams implement against this; app never assumes internals."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class ModelAdapter(ABC):
    architecture: str = "unknown"

    @abstractmethod
    def build(self, num_classes: int, pretrained: bool = False) -> Any:
        """Return torch.nn.Module."""

    @abstractmethod
    def target_layers(self, module: Any) -> Any:
        """Return list of candidate target layers for CAM."""

    @abstractmethod
    def parameter_count(self, module: Any) -> Any:
        ...

    def flops(self, module: Any, input_size: int = 224) -> Any:
        return 0.0

    @abstractmethod
    def input_size(self) -> Any:
        ...

    def metadata(self) -> Any:
        return {"architecture": self.architecture, "adapter": type(self).__name__}
