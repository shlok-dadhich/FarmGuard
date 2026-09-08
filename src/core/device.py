"""Device + seed + types helpers."""
from __future__ import annotations

import random

import numpy as np


def resolve_device(preference: str = "auto") -> str:
    if preference in ("cuda", "gpu"):
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        except Exception:
            return "cpu"
    if preference == "cpu":
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
