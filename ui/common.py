"""Shared Streamlit helpers.

Caching discipline:
- st.cache_resource -> model instances, config, device (expensive global resources)
- st.cache_data     -> deterministic computations (runs table, class names, dataset info)
- st.session_state  -> per-image results (predictions, CAM arrays) keyed by content hash
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- config/device
@st.cache_resource(show_spinner=False)
def load_app_config():
    from src.core.config import load_config

    return load_config()


@st.cache_resource(show_spinner=False)
def get_device() -> str:
    from src.core.device import resolve_device

    try:
        pref = load_app_config().device
    except Exception:
        pref = "auto"
    return resolve_device(pref)


@st.cache_data(show_spinner=False)
def get_crops() -> list:
    try:
        return list(load_app_config().datasets.get("crops", {}).keys())
    except Exception:
        return ["tomato"]


@st.cache_data(show_spinner=False)
def get_architectures() -> list:
    try:
        return list(load_app_config().models_cfg.get("models", {}).keys())
    except Exception:
        return ["mock_demo"]


# ---------------------------------------------------------------- model instances
@st.cache_data(show_spinner=False)
def get_model_specs() -> list:
    """Configured model instances from configs/models.yaml -> instances: ([] when none)."""
    from src.models.metadata import load_model_specs

    try:
        return load_model_specs(load_app_config().models_cfg)
    except Exception:
        return []


def get_model_choices(crop: str) -> list:
    """Model keys offered for a crop: configured instance ids when instances exist,
    otherwise all registered architectures (classic behaviour)."""
    specs = get_model_specs()
    inst = [s.id for s in specs if crop in s.crops]
    if inst:
        return inst
    return get_architectures()


def model_info(crop: str, key: str) -> dict:
    """Resolve a model key (instance id or architecture) to build/load parameters.

    Returns: {key, name, architecture, checkpoint, input_size, target_layer,
    is_instance, demo}. Never raises for missing checkpoints — the caller decides
    how to surface the DEMO fallback.
    """
    specs = get_model_specs()
    for s in specs:
        if s.id == key:
            arch = s.architecture
            ckpt = s.checkpoint
            if ckpt and not Path(ckpt).exists():
                ckpt = None  # fall back to auto-discovery; warning shown elsewhere
            ckpt = ckpt or find_checkpoint(crop, arch)
            from src.models.factory import get_adapter

            return {
                "key": key,
                "name": s.display_name,
                "architecture": arch,
                "checkpoint": ckpt,
                "input_size": s.input_size or get_adapter(arch).input_size(),
                "target_layer": s.target_layer,
                "is_instance": True,
                "demo": arch == "mock_demo" or not ckpt,
            }
    # classic architecture key
    from src.models.factory import get_adapter

    return {
        "key": key,
        "name": key,
        "architecture": key,
        "checkpoint": find_checkpoint(crop, key),
        "input_size": get_adapter(key).input_size(),
        "target_layer": "auto",
        "is_instance": False,
        "demo": key == "mock_demo" or not find_checkpoint(crop, key),
    }


# ---------------------------------------------------------------- class names
@st.cache_data(show_spinner=False)
def get_class_names(crop: str) -> list:
    """Resolve class names: manifest labels first (eval ground truth), then an
    instance classes_file, then datasets.yaml, then a generic fallback."""
    for split in ("train", "val", "test"):
        p = REPO_ROOT / "data" / "splits" / crop / f"{split}.csv"
        if p.exists():
            try:
                df = pd.read_csv(p)
                if not df.empty and {"label", "label_id"} <= set(df.columns):
                    df = df.sort_values("label_id").drop_duplicates("label_id")
                    return df["label"].astype(str).tolist()
            except Exception:
                pass
    try:
        from src.models.metadata import class_names_for_crop

        names = class_names_for_crop(crop, get_model_specs(), load_app_config().datasets)
        if names and not names[0].startswith("class"):
            return names
    except Exception:
        pass
    try:
        classes = (load_app_config().datasets.get("crops", {}).get(crop, {}) or {}).get("classes", [])
        if classes:
            return list(classes)
    except Exception:
        pass
    return [f"class{i}" for i in range(4)]


# ---------------------------------------------------------------- checkpoints
@st.cache_data(show_spinner=False)
def find_checkpoint(crop: str, arch: str):
    """Newest matching checkpoint for (crop, arch), or None when absent (never fabricated)."""
    from src.models.metadata import discover_checkpoint

    return discover_checkpoint(crop, arch, (REPO_ROOT / "models" / "checkpoints", REPO_ROOT / "outputs" / "checkpoints"))


@st.cache_resource(show_spinner="Loading model weights…")
def get_model(arch: str, num_classes: int, checkpoint, device: str):
    from src.inference.model_loader import load_model

    return load_model(arch, max(1, int(num_classes)), checkpoint, device)


def is_demo(arch: str, checkpoint) -> bool:
    return arch == "mock_demo" or not checkpoint


# ---------------------------------------------------------------- runs / evals
@st.cache_data(show_spinner=False)
def load_runs_table() -> pd.DataFrame:
    p = REPO_ROOT / "outputs" / "metrics" / "runs.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_eval_record(crop: str, arch: str) -> dict:
    import json

    p = REPO_ROOT / "outputs" / "metrics" / f"eval_{crop}_{arch}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def dataset_info() -> dict:
    """Row counts per crop/split from manifests (empty when unavailable — never faked)."""
    info = {}
    for crop_dir in (REPO_ROOT / "data" / "splits").glob("*"):
        if not crop_dir.is_dir():
            continue
        info[crop_dir.name] = {}
        for split in ("train", "val", "test"):
            p = crop_dir / f"{split}.csv"
            if p.exists():
                try:
                    info[crop_dir.name][split] = len(pd.read_csv(p))
                except Exception:
                    info[crop_dir.name][split] = "error"
    return info


@st.cache_data(show_spinner=False)
def env_info() -> dict:
    from src.utils.reproducibility import env_report

    try:
        return env_report()
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------- inference/xai
def pil_from_upload(upload):
    from PIL import Image

    return Image.open(upload).convert("RGB")


def predict_pil(model, class_names: list, pil_img, device: str, demo: bool, arch: str, crop: str, top_k: int = 3):
    from src.inference.predictor import Predictor

    return Predictor(model, class_names, device, demo=demo, arch=arch, crop=crop).predict(pil_img, top_k)


XAI_METHODS = ("gradcam_pp", "scorecam", "saliency")


def compute_cam(model, adapter, tensor, method: str = "gradcam_pp", class_idx=None, max_masks: int = 16):
    """Return (heatmap np.ndarray in [0,1], class_idx). Raises XAIError-style RuntimeError on failure."""
    from src.explainability.target_layers import resolve_target_layers

    model.eval()
    layer = resolve_target_layers(adapter, model)[-1]
    try:
        if method == "scorecam":
            from src.explainability.scorecam import ScoreCAM

            return ScoreCAM(model, layer, max_masks=max_masks)(tensor, class_idx)
        if method == "saliency":
            from src.explainability.saliency import Saliency

            return Saliency(model)(tensor, class_idx)
        from src.explainability.gradcam import GradCAMpp

        return GradCAMpp(model, layer)(tensor, class_idx)
    except Exception as e:
        raise RuntimeError(
            f"WHAT: {method} failed for this model\nWHY: {e}\n"
            f"HOW: check adapter target_layers() for this architecture, or choose a "
            f"method this model supports. XAI method unavailable for this model is a "
            f"valid outcome — the rest of the app keeps working."
        ) from e


def target_layer_names(model, adapter) -> list:
    from src.explainability.target_layers import resolve_target_layers

    try:
        layers = resolve_target_layers(adapter, model)
    except Exception as e:
        return [f"unresolved ({e})"]
    ids = {id(x) for x in layers}
    names = [n for n, m in model.named_modules() if id(m) in ids]
    return names or [type(x).__name__ for x in layers]


def blend_overlay(pil_img, cam, alpha: float = 0.45):
    import numpy as np

    from src.explainability.heatmap import overlay

    return overlay(np.asarray(pil_img.convert("RGB")), cam, alpha=alpha)


def colorize_cam(cam):
    """Jet-coloured RGB rendering of a [0,1] heatmap."""
    import numpy as np

    try:
        from matplotlib import colormaps

        cmap = colormaps["jet"]
    except Exception:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("jet")
    return (cmap(np.asarray(cam))[:, :, :3] * 255).astype("uint8")


def save_overlay_artifact(overlay_arr, meta: dict, prefix: str = "ui"):
    import json
    import time

    from PIL import Image

    out = REPO_ROOT / "outputs" / "xai"
    out.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    img_p = out / f"{prefix}_{ts}_overlay.jpg"
    Image.fromarray(overlay_arr).save(img_p)
    meta_p = out / f"{prefix}_{ts}_meta.json"
    meta_p.write_text(json.dumps({**meta, "overlay": img_p.name, "timestamp": ts}, indent=2, default=str))
    return img_p, meta_p


# ---------------------------------------------------------------- banners
def demo_banner():
    st.warning("DEMO / TEST MODEL — not a research result. Supply real checkpoints per docs/model_integration.md.")


def unavailable_banner(msg: str = "Results not available yet."):
    st.info(f"UNAVAILABLE RESULT — {msg}")


def error_box(e: Exception):
    st.error(str(e))