"""Factory: instantiate adapters + models, load checkpoints with clear errors."""
from __future__ import annotations
import importlib
from pathlib import Path
from src.models.registry import MODEL_REGISTRY
from src.core.exceptions import UnsupportedArchitectureError, CheckpointError, ModelNotFoundError


def get_adapter(arch: str):
    if arch not in MODEL_REGISTRY:
        raise UnsupportedArchitectureError(
            f"Unknown architecture '{arch}'",
            f"registry has {sorted(MODEL_REGISTRY)}",
            "HOW: check configs/models.yaml or implement adapter in src/models/adapters/",
        )
    dotted = MODEL_REGISTRY[arch]
    mod, cls = dotted.split(":")
    module = importlib.import_module(mod)
    return getattr(module, cls)()


def build_model(arch: str, num_classes: int, pretrained: bool = False):
    return get_adapter(arch).build(num_classes, pretrained=pretrained)


def load_checkpoint_weights(module, ckpt_path: str | Path, device="cpu"):
    import torch
    p = Path(ckpt_path)
    if not p.exists():
        raise CheckpointError(f"Checkpoint not found: {p}", "file missing",
                              "HOW: place externally supplied weights under models/checkpoints/ or train via scripts/train.py")
    try:
        state = torch.load(str(p), map_location=device)
    except Exception as e:
        raise CheckpointError(f"Could not load checkpoint {p}: {e}", "corrupt/incompatible file",
                              "HOW: verify torch version and that checkpoint is a state_dict")
    sd = state.get("state_dict", state) if isinstance(state, dict) else state
    if not isinstance(sd, dict):
        raise CheckpointError(f"Checkpoint {p} has unexpected structure {type(sd)}", "not a state_dict",
                              "HOW: save checkpoints as {'state_dict': model.state_dict(), ...}")
    missing, unexpected = module.load_state_dict(sd, strict=False), None
    return module
