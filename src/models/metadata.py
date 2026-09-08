"""Model instance metadata.

The architecture *registry* (configs/models.yaml -> ``models:``) defines *how to
build* each architecture. The *instance list* (configs/models.yaml -> ``instances:``)
binds a concrete checkpoint + crop + class mapping to an architecture so a user can
register a trained model by editing YAML instead of Python.

Example instance entry::

    instances:
      - id: efficientnet_crop1
        name: EfficientNetV2-S (crop1)
        architecture: efficientnet_v2_s
        crop: tomato
        checkpoint: models/checkpoints/tomato/efficientnet_v2_s.pt
        classes_file: configs/classes/tomato.yaml
        input_size: 224        # optional; defaults to the adapter's input size
        target_layer: auto     # optional; adapters resolve their own layers

An ``instances:`` list is optional. When it is absent (or empty) the app falls back
to the classic behaviour: all registered architectures are offered and checkpoints
are auto-discovered under models/checkpoints/ and outputs/checkpoints/.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.core.exceptions import ConfigError
from src.models.registry import MODEL_REGISTRY


@dataclass
class ModelSpec:
    """A configured, externally-supplied model instance."""

    id: str
    architecture: str
    name: str = ""
    crops: list[str] = field(default_factory=list)
    checkpoint: str | None = None
    classes_file: str | None = None
    input_size: int | None = None  # None -> use the adapter's input size
    target_layer: str = "auto"
    notes: str = ""

    @property
    def display_name(self) -> str:
        return self.name or self.id

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "architecture": self.architecture,
            "crops": list(self.crops),
            "checkpoint": self.checkpoint,
            "classes_file": self.classes_file,
            "input_size": self.input_size,
            "target_layer": self.target_layer,
            "notes": self.notes,
        }


def _load_classes_file(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(
            f"Classes file not found: {p}",
            "configured classes_file points at a missing file",
            "HOW: create configs/classes/<crop>.yaml with a 'classes:' list, or fix the path",
        )
    try:
        data = yaml.safe_load(p.read_text()) or {}
    except Exception as e:
        raise ConfigError(
            f"Could not parse classes file {p}: {e}",
            "malformed YAML",
            "HOW: fix configs/classes/<crop>.yaml",
        ) from e
    classes = data.get("classes", []) if isinstance(data, dict) else []
    if not isinstance(classes, list) or not classes:
        raise ConfigError(
            f"Classes file {p} has no non-empty 'classes' list",
            "missing classes key",
            "HOW: add 'classes: [class_a, class_b, ...]'",
        )
    return [str(c) for c in classes]


def load_model_specs(models_cfg: dict) -> list[ModelSpec]:
    """Parse and validate the ``instances:`` list from configs/models.yaml.

    Raises ConfigError with actionable messages on malformed entries. Never
    fabricates checkpoints or class names.
    """
    instances = models_cfg.get("instances", []) or []
    specs: list[ModelSpec] = []
    seen: set[str] = set()
    for i, raw in enumerate(instances):
        if not isinstance(raw, dict):
            raise ConfigError(
                f"models.yaml instances[{i}] must be a mapping",
                "malformed instance entry",
                "HOW: see the commented example at the top of configs/models.yaml",
            )
        mid = str(raw.get("id", "")).strip()
        if not mid:
            raise ConfigError(
                f"models.yaml instances[{i}] missing 'id'",
                "every instance needs a unique id",
                "HOW: add id: <unique-model-key>",
            )
        if mid in seen:
            raise ConfigError(
                f"Duplicate model instance id '{mid}'",
                "ids must be unique",
                "HOW: rename one of the instances",
            )
        seen.add(mid)
        arch = str(raw.get("architecture", "")).strip()
        if not arch:
            raise ConfigError(
                f"Instance '{mid}' missing 'architecture'",
                "architecture must reference a key from models.yaml -> models:",
                "HOW: add architecture: <key from models.yaml>",
            )
        if arch not in MODEL_REGISTRY:
            raise ConfigError(
                f"Instance '{mid}' references unknown architecture '{arch}'",
                f"registry has {sorted(MODEL_REGISTRY)}",
                "HOW: fix the architecture key, or register a new architecture adapter "
                "(see docs/model_integration.md)",
            )
        crops_raw = raw.get("crops") or ([raw["crop"]] if raw.get("crop") else [])
        crops = [str(c).strip() for c in crops_raw if str(c).strip()]
        if not crops:
            raise ConfigError(
                f"Instance '{mid}' has no crop(s)",
                "a model must declare at least one crop",
                "HOW: add crop: <crop-name> (must exist in configs/datasets.yaml)",
            )
        ckpt = raw.get("checkpoint")
        if ckpt is not None:
            ckpt = str(ckpt)
        isz = raw.get("input_size")
        if isz is not None:
            try:
                isz = int(isz)
            except (TypeError, ValueError):
                raise ConfigError(
                    f"Instance '{mid}' input_size must be an integer, got {isz!r}",
                    "bad input_size",
                    "HOW: fix input_size in configs/models.yaml",
                ) from None
        specs.append(
            ModelSpec(
                id=mid,
                name=str(raw.get("name", "") or "").strip(),
                architecture=arch,
                crops=crops,
                checkpoint=ckpt,
                classes_file=str(raw["classes_file"]) if raw.get("classes_file") else None,
                input_size=isz,
                target_layer=str(raw.get("target_layer", "auto")),
                notes=str(raw.get("notes", "") or ""),
            )
        )
    return specs


def models_for_crop(specs: list[ModelSpec], crop: str) -> list[ModelSpec]:
    """Instances declared for a crop (in config order)."""
    return [s for s in specs if crop in s.crops]


def class_names_for_crop(crop: str, specs: list[ModelSpec], datasets_cfg: dict) -> list[str]:
    """Class-name resolution priority:

    1. a classes_file declared on an instance for this crop
    2. class list from datasets.yaml -> crops.<crop>.classes
    3. generic fallback (never used silently in evaluation; UI only)
    """
    for s in specs:
        if crop in s.crops and s.classes_file:
            return _load_classes_file(s.classes_file)
    classes = (datasets_cfg.get("crops", {}).get(crop, {}) or {}).get("classes", [])
    if classes:
        return [str(c) for c in classes]
    return [f"class{i}" for i in range(4)]


def discover_checkpoint(crop: str, arch: str,
                        roots: tuple[Path, ...] = (Path("models/checkpoints"), Path("outputs/checkpoints"))) -> str | None:
    """Newest matching checkpoint for (crop, arch) under the given roots.

    Matches <crop>_<arch>*.pt/.pth and <arch>*.pt. Returns None when absent
    (never fabricates a path).
    """
    cands: list[Path] = []
    for d in roots:
        if d.exists():
            cands += list(d.glob(f"{crop}_{arch}*.pt")) + list(d.glob(f"{crop}_{arch}*.pth"))
            cands += list(d.glob(f"{arch}*.pt"))
    if not cands:
        return None
    return str(max(cands, key=lambda p: p.stat().st_mtime))


def validate_instance_checkpoints(specs: list[ModelSpec], root: Path | None = None) -> list[str]:
    """Return warnings for instances whose explicit checkpoint path is missing.

    Missing checkpoints never crash the app: the UI falls back to auto-discovery
    and labels the model as DEMO (random init) until weights are supplied.
    """
    warnings: list[str] = []
    for s in specs:
        if s.checkpoint:
            p = Path(s.checkpoint)
            if not p.is_absolute() and root is not None:
                p = root / p
            if not p.exists():
                warnings.append(
                    f"Instance '{s.id}': configured checkpoint does not exist: {s.checkpoint}. "
                    "Falling back to auto-discovery under models/checkpoints/ and "
                    "outputs/checkpoints/; if none is found the model runs as DEMO (random init)."
                )
    return warnings