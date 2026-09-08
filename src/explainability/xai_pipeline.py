from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from src.explainability.gradcam import GradCAMpp
from src.explainability.scorecam import ScoreCAM
from src.explainability.heatmap import overlay
from src.explainability.target_layers import resolve_target_layers

def run_xai(model, adapter, pil_img, input_tensor, out_dir="outputs/xai", methods=("gradcam_pp",), class_idx=None, meta=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    layers = resolve_target_layers(adapter, model)
    layer = layers[-1]
    orig = np.asarray(pil_img.convert("RGB").resize((input_tensor.shape[3], input_tensor.shape[2])))
    ts = int(time.time()); results = {}
    model.eval()
    if "gradcam_pp" in methods:
        cam, c = GradCAMpp(model, layer)(input_tensor, class_idx)
        vis = overlay(orig, cam)
        Image.fromarray(vis).save(out / f"xai_{ts}_gradcampp_c{c}.jpg")
        results["gradcam_pp"] = {"class": c, "file": f"xai_{ts}_gradcampp_c{c}.jpg"}
    if "scorecam" in methods:
        cam, c = ScoreCAM(model, layer)(input_tensor, class_idx)
        vis = overlay(orig, cam)
        Image.fromarray(vis).save(out / f"xai_{ts}_scorecam_c{c}.jpg")
        results["scorecam"] = {"class": c, "file": f"xai_{ts}_scorecam_c{c}.jpg"}
    (out / f"xai_{ts}_meta.json").write_text(json.dumps({**(meta or {}), **results, "timestamp": ts}, indent=2))
    return results
