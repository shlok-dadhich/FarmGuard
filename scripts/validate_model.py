"""Validate a configured model before running evaluations.

Usage:
    python scripts/validate_model.py --model <instance-id-or-arch> [--crop <crop>]

Checks, in order:
 1. the model resolves (instance from configs/models.yaml -> instances:, or a
    registered architecture key)
 2. a checkpoint is resolved (explicit instance path or auto-discovery)
 3. the checkpoint exists and loads against the built architecture
 4. class names resolve (split-manifest labels -> classes_file -> datasets.yaml)
    and their count is sane
 5. the input size is valid
 6. XAI target layers resolve for this architecture
 7. one forward pass succeeds at the input size

A missing checkpoint is reported as REQUIRES USER INPUT (the model then runs as
DEMO/random-init) and does NOT fail the validation; broken configs, unloadable
checkpoints or failed forward passes DO fail (exit code 1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from src.core.config import load_config
from src.models.factory import get_adapter, load_checkpoint_weights
from src.models.metadata import (class_names_for_crop, discover_checkpoint,
                                 load_model_specs)
from src.models.registry import MODEL_REGISTRY

CHECKS = ["resolve", "checkpoint", "load", "classes", "input_size", "target_layers", "forward"]


def _manifest_class_names(crop: str) -> list[str] | None:
    for split in ("train", "val", "test"):
        p = Path(f"data/splits/{crop}/{split}.csv")
        if p.exists():
            try:
                df = pd.read_csv(p)
                if not df.empty and {"label", "label_id"} <= set(df.columns):
                    df = df.sort_values("label_id").drop_duplicates("label_id")
                    return df["label"].astype(str).tolist()
            except Exception:
                pass
    return None


def _manifest_num_classes(crop: str) -> int | None:
    for split in ("train", "val", "test"):
        p = Path(f"data/splits/{crop}/{split}.csv")
        if p.exists():
            try:
                df = pd.read_csv(p)
                if not df.empty and "label_id" in df.columns:
                    return int(df["label_id"].nunique())
            except Exception:
                pass
    return None


def validate_model(model_key: str, crop: str | None = None, cfg=None) -> dict:
    """Validate one model. Returns a report dict; never raises for missing weights."""
    cfg = cfg or load_config()
    specs = load_model_specs(cfg.models_cfg)
    spec = next((s for s in specs if s.id == model_key), None)
    report = {"model": model_key, "checks": {}, "issues": [], "warnings": [], "status": "ok"}

    # 1. resolve
    if spec is not None:
        arch = spec.architecture
        crop = crop or (spec.crops[0] if spec.crops else None)
        ckpt = None
        if spec.checkpoint and Path(spec.checkpoint).exists():
            ckpt = spec.checkpoint
        else:
            ckpt = discover_checkpoint(crop or "", arch)
        input_size = spec.input_size
        report["checks"]["resolve"] = {"ok": True, "kind": "instance",
                                       "architecture": arch, "crop": crop}
    elif model_key in MODEL_REGISTRY:
        arch = model_key
        ckpt = discover_checkpoint(crop or "", arch)
        input_size = None
        report["checks"]["resolve"] = {"ok": True, "kind": "architecture",
                                       "architecture": arch, "crop": crop}
    else:
        report["checks"]["resolve"] = {"ok": False,
                                       "message": f"'{model_key}' is not a configured instance "
                                                  f"and not a registered architecture "
                                                  f"({sorted(MODEL_REGISTRY)})"}
        report["status"] = "error"
        return report

    # 4. class names
    class_names = _manifest_class_names(crop or "") or (
        class_names_for_crop(crop or "", specs, cfg.datasets) if spec else
        (cfg.datasets.get("crops", {}).get(crop or "", {}).get("classes") or []))
    if not class_names:
        class_names = [f"class{i}" for i in range(4)]
        report["warnings"].append("no class names resolved — using generic class0..3 "
                                  "(add classes_file or datasets.yaml classes)")
    report["classes"] = list(class_names)
    n_manifest = _manifest_num_classes(crop or "")
    if n_manifest is not None and n_manifest != len(class_names):
        report["warnings"].append(f"manifest has {n_manifest} classes but {len(class_names)} "
                                  "class names resolved — verify class ordering/coverage")

    # 2/3/5/6/7 need a built model
    adapter = get_adapter(arch)
    if input_size is None:
        input_size = adapter.input_size()
    report["input_size"] = int(input_size)
    report["checkpoint"] = ckpt

    try:
        model = adapter.build(max(1, len(class_names)), pretrained=False)
    except Exception as e:
        report["checks"]["build"] = {"ok": False, "message": str(e)}
        report["status"] = "error"
        return report

    # 2/3. checkpoint
    if ckpt:
        report["checks"]["checkpoint"] = {"ok": True, "path": ckpt}
        try:
            load_checkpoint_weights(model, ckpt)
            report["checks"]["load"] = {"ok": True,
                                        "params": adapter.parameter_count(model)}
        except Exception as e:
            report["checks"]["load"] = {"ok": False, "message": str(e)}
            report["status"] = "error"
    else:
        # missing weights are REQUIRES USER INPUT, not a validation failure: the
        # model still builds and runs as DEMO until a checkpoint is supplied
        report["checks"]["checkpoint"] = {"ok": None, "path": None,
                                          "message": "no checkpoint found - REQUIRES USER "
                                                     "INPUT; model runs as DEMO (random init)"}
        report["warnings"].append("weights not supplied - evaluation results will be DEMO "
                                  "until a checkpoint is placed under models/checkpoints/")
        report["checks"]["load"] = {"ok": None, "message": "skipped (no checkpoint)"}

    # 5. input size
    report["checks"]["input_size"] = {"ok": int(input_size) >= 32, "value": int(input_size)}

    # 6. target layers
    try:
        from src.explainability.target_layers import resolve_target_layers
        layers = resolve_target_layers(adapter, model)
        report["checks"]["target_layers"] = {"ok": True, "count": len(layers),
                                             "names": [type(l).__name__ for l in layers]}
    except Exception as e:
        report["checks"]["target_layers"] = {"ok": False, "message": str(e)}
        report["warnings"].append("XAI target layer unresolved — Grad-CAM++/Score-CAM "
                                  "unavailable for this model until the adapter implements "
                                  "target_layers(); Saliency may still work")

    # 7. forward pass
    try:
        model.eval()
        with torch.no_grad():
            model(torch.randn(1, 3, int(input_size), int(input_size)))
        report["checks"]["forward"] = {"ok": True, "input_size": int(input_size)}
    except Exception as e:
        report["checks"]["forward"] = {"ok": False, "message": str(e)}
        report["status"] = "error"

    return report


def main():
    ap = argparse.ArgumentParser(description="Validate a configured model.")
    ap.add_argument("--model", required=True, help="instance id or architecture key")
    ap.add_argument("--crop", default=None, help="crop key (needed for architecture keys)")
    a = ap.parse_args()
    cfg = load_config()
    report = validate_model(a.model, a.crop, cfg)
    print(json.dumps(report, indent=2, default=str))
    bad = [k for k, v in report["checks"].items() if v.get("ok") is False]
    if bad:
        print(f"\nVALIDATION FAILED on: {', '.join(bad)}")
        sys.exit(1)
    if report.get("warnings"):
        print("\nVALIDATION OK WITH WARNINGS:\n- " + "\n- ".join(report["warnings"]))
    else:
        print("\nVALIDATION OK")
    sys.exit(0)


if __name__ == "__main__":
    main()