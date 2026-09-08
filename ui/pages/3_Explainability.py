"""Explainability page: Grad-CAM++ vs Score-CAM vs Saliency, overlay controls, faithfulness."""
from __future__ import annotations

import hashlib
import io

import streamlit as st
from PIL import Image

from ui.common import (
    XAI_METHODS,
    blend_overlay,
    colorize_cam,
    compute_cam,
    demo_banner,
    error_box,
    get_class_names,
    get_crops,
    get_device,
    get_model,
    get_model_choices,
    model_info,
    predict_pil,
    target_layer_names,
    unavailable_banner,
)
from ui.theme import inject

inject()
st.title("🔥 Explainability")
st.caption("Grad-CAM++ with Score-CAM and saliency cross-check. Heatmaps explain the model — they are not diagnoses.")

crops = get_crops()
device = get_device()

c1, c2 = st.columns([1, 2])
with c1:
    crop = st.selectbox("Crop", crops, key="xai_crop")
    choices = get_model_choices(crop)
    default_idx = choices.index("mock_demo") if "mock_demo" in choices else 0
    key = st.selectbox("Model", choices, index=default_idx, key="xai_model")
    upload = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp", "bmp"], key="xai_upload")
    true_label = st.text_input("True label (optional, if known)", key="xai_true")
    run = st.button("Explain", type="primary", disabled=upload is None, key="xai_go")

info = model_info(crop, key)
class_names = get_class_names(crop)
if info["demo"] and not info["checkpoint"]:
    st.warning("WHAT: no checkpoint for this model — WHY: weights not supplied — "
               "HOW: models/checkpoints/ or configure an instance in configs/models.yaml. "
               "Using random weights (DEMO).")

if run and upload is not None:
    try:
        with st.spinner("Predicting + computing explanations…"):
            img_bytes = upload.getvalue()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            model = get_model(info["architecture"], len(class_names), info["checkpoint"], device)
            from src.inference.preprocessing import preprocess_pil
            from src.models.factory import get_adapter

            adapter = get_adapter(info["architecture"])
            tensor = preprocess_pil(img, info["input_size"])
            pred = predict_pil(model, class_names, img, device, info["demo"], info["architecture"], crop, top_k=3)
            cams = {}
            for m in XAI_METHODS:
                try:
                    cams[m] = compute_cam(model, adapter, tensor, method=m)
                except Exception as e:
                    cams[m] = ("error", str(e))
            st.session_state["xai"] = {"pred": pred, "cams": cams, "arch": info["architecture"],
                                       "model_name": info["name"], "crop": crop,
                                       "ckpt": info["checkpoint"], "demo": info["demo"],
                                       "classes": class_names,
                                       "layers": target_layer_names(model, adapter),
                                       "img_hash": hashlib.md5(img_bytes).hexdigest()}
            st.session_state["xai_bytes"] = img_bytes
    except Exception as e:
        error_box(e)

res = st.session_state.get("xai")
if res is None:
    st.info("Upload an image and press **Explain**.")
    # gallery of past artifacts
    from pathlib import Path

    files = sorted(str(x) for x in Path("outputs/xai").glob("*.jpg"))[-8:]
    if files:
        st.markdown("**Past artifacts (outputs/xai/)**")
        cols = st.columns(4)
        for i, f in enumerate(files):
            cols[i % 4].image(f, width="stretch")
    st.stop()
    raise SystemExit

if res["demo"]:
    demo_banner()
pred = res["pred"]
m1, m2 = st.columns(2)
m1.metric("Prediction", pred["label"])
m2.metric("Confidence", f"{pred['confidence']:.1%}")
if true_label:
    if true_label.strip() == pred["label"]:
        st.success("Prediction matches the supplied true label (correct-prediction case).")
    else:
        st.error(f"Prediction differs from true label '{true_label.strip()}' (incorrect-prediction case).")
if pred["confidence"] < 0.6:
    st.warning("Low-confidence prediction — treat the heatmap as exploratory, not evidence.")
st.markdown(
    "<span class='small-note'>Highlighted regions indicate image areas contributing strongly to the "
    "selected prediction. Not agronomic certainty.</span>",
    unsafe_allow_html=True)

alpha = st.slider("Overlay opacity α", 0.0, 1.0, 0.45, 0.05, key="xai_alpha")
img = Image.open(io.BytesIO(st.session_state["xai_bytes"])).convert("RGB")

LABELS = {"gradcam_pp": "Grad-CAM++", "scorecam": "Score-CAM", "saliency": "Saliency"}
cams = res["cams"]
tabs = st.tabs(["Side-by-side"] + [LABELS[m] for m in XAI_METHODS])
for ti, m in enumerate(XAI_METHODS, start=1):
    with tabs[ti]:
        if isinstance(cams.get(m), tuple):
            error_box(RuntimeError(cams[m][1]))
            st.caption("XAI method unavailable for this model — other methods above remain usable.")
        else:
            cam, cls_idx = cams[m]
            a, b, c_ = st.columns(3)
            a.image(img, caption="Original", width="stretch")
            b.image(colorize_cam(cam), caption=f"{LABELS[m]} (class {cls_idx})", width="stretch")
            c_.image(blend_overlay(img, cam, alpha=alpha), caption=f"Overlay α={alpha:.2f}", width="stretch")
with tabs[0]:
    ok = {m: cams[m] for m in XAI_METHODS if not isinstance(cams.get(m), tuple)}
    if len(ok) >= 2:
        cols = st.columns(len(ok))
        first_cls = None
        agree = True
        for i, (m, (cam, cls_idx)) in enumerate(ok.items()):
            cols[i].image(blend_overlay(img, cam, alpha=alpha), caption=f"{LABELS[m]} overlay", width="stretch")
            first_cls = cls_idx if first_cls is None else first_cls
            agree = agree and cls_idx == first_cls
        st.caption("All available methods agree on the explained class — cross-check passes." if agree
                   else "Methods disagree on the explained class — inspect both before trusting either.")
    else:
        st.info("Fewer than two methods succeeded — see individual tabs for the errors.")

with st.expander("🧬 Target-layer metadata", expanded=False):
    st.json({"model": res["model_name"], "architecture": res["arch"], "target_layers": res["layers"],
             "checkpoint": res["ckpt"] or "none (random init)", "device": device})

# ------------------------------------------------ faithfulness (pointing game)
st.divider()
st.subheader("🎯 Faithfulness (pointing game)")
st.caption("Heatmap maximum inside the lesion mask? Requires a binary lesion mask — never fabricated.")
mask_up = st.file_uploader("Lesion mask (optional, PNG, white = lesion)", type=["png", "jpg", "jpeg"],
                           key="xai_mask")
g = cams.get("gradcam_pp")
if mask_up is not None and not isinstance(g, tuple):
    try:
        import numpy as np

        from src.explainability.faithfulness import faithfulness_score, pointing_hit

        mask = Image.open(mask_up).convert("L").resize((g[0].shape[1], g[0].shape[0]))
        marr = (np.asarray(mask) > 127).astype("uint8")
        hit = pointing_hit(g[0], marr)
        st.json(faithfulness_score([hit]))
        st.success("HIT — heatmap maximum lies inside the lesion region." if hit else
                   "MISS — heatmap maximum lies outside the lesion region.")
    except Exception as e:
        error_box(e)
else:
    st.info("Quantitative lesion-based evaluation cannot be performed without masks. "
            "Upload one above, or place masks under data/raw/<crop>/masks/. Qualitative visualizations above remain valid.")