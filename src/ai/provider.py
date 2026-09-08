from __future__ import annotations
from abc import ABC, abstractmethod
class AIProvider(ABC):
    @abstractmethod
    def explain(self, context: dict) -> str: ...
