"""Typed configuration layer with validation. No hard-coded experiment settings elsewhere."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.core.exceptions import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _expand_env(obj):  # type: ignore[no-untyped-def]
    if isinstance(obj, str):
        def repl(m):  # type: ignore[no-untyped-def]
            expr = m.group(1)
            if ":" in expr:
                var, default = expr.split(":", 1)
                return os.environ.get(var, default)
            return os.environ.get(expr, "") or ""
        return re.sub(r"\$\{([^}]+)\}", repl, obj)  # type: ignore[type-var]
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj


def load_yaml(name: str):  # type: ignore[no-untyped-def]
    path = CONFIG_DIR / name
    if not path.exists():
        raise ConfigError(f"Config file missing: {path}", "expected configs/*.yaml", "HOW: run from repo root; check CONFIG_DIR")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return _expand_env(data)


@dataclass
class AppConfig:
    project: dict = field(default_factory=dict)
    datasets: dict = field(default_factory=dict)
    models_cfg: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    explainability: dict = field(default_factory=dict)
    ai: dict = field(default_factory=dict)
    tracking: dict = field(default_factory=dict)

    @property
    def seed(self) -> int:
        return int(self.project.get("seed", 42))

    @property
    def device(self) -> str:
        return str(self.project.get("device", "auto"))

    @property
    def image_size(self) -> int:
        return int(self.project.get("image_size", 224))

    @property
    def demo_mode(self) -> bool:
        return bool(self.project.get("demo_mode", True))

    def validate(self) -> None:
        if "crops" not in self.datasets:
            raise ConfigError("datasets.yaml missing 'crops' key", "malformed config", "HOW: restore configs/datasets.yaml")
        if "models" not in self.models_cfg:
            raise ConfigError("models.yaml missing 'models' key", "malformed config", "HOW: restore configs/models.yaml")
        ratios = self.datasets.get("split_ratios", {})
        total = sum(float(ratios.get(k, 0)) for k in ("train", "val", "test"))
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"split_ratios must sum to 1.0, got {total}", "bad split config", "HOW: fix configs/datasets.yaml")


def load_config(config_dir: str | Path | None = None) -> AppConfig:
    global CONFIG_DIR
    if config_dir is not None:
        CONFIG_DIR = Path(config_dir)
    cfg = AppConfig(
        project=load_yaml("project.yaml"),
        datasets=load_yaml("datasets.yaml"),
        models_cfg=load_yaml("models.yaml"),
        training=load_yaml("training.yaml"),
        explainability=load_yaml("explainability.yaml"),
        ai=load_yaml("ai.yaml"),
        tracking=load_yaml("tracking.yaml"),
    )
    cfg.validate()
    return cfg
