from __future__ import annotations
from pathlib import Path
import torch
def save_checkpoint(module, path, meta=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": module.state_dict(), "meta": meta or {}}, str(path))
    return str(path)
