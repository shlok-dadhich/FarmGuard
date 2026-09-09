"""Diagnosis page: crop + image + model -> prediction + XAI + optional AI explanation."""
from __future__ import annotations

import hashlib

import streamlit as st

from ui.common import (
    XAI_METHODS,
    XAI_METHOD_LABELS,
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
    load_app_config,
    model_info,
    pil_from_upload,
    predict_pil,
    save_overlay_artifact,
    target_layer_names,
)
from ui.theme import inject

inject()
st.title("🔬 Diagnosis")
st.caption("Single-image inference. The classifier is the source of prediction truth; AI text is assistive only.")

try:
    cfg = load_app_config()
except Exception as e:
    error_box(e)
    st.stop()

crops = get_crops()
device = get_device()

# ---------------------------------------------------------------- controls
c_left, c_right = st.columns([1, 2])
with c_left:
    crop = st.selectbox("Crop", crops, key="diag_crop")
    choices = get_model_choices(crop)
    default_idx = choices.index("tomato_regnet_y_4gf") if "tomato_regnet_y_4gf" in choices else 0
    key = st.selectbox("Model", choices, index=default_idx, key="diag_model",
                       help="Configured model instances (configs/models.yaml -> instances:) and "
                            "registered architectures. External checkpoints load automatically.")
    upload = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png", "webp", "bmp"], key="diag_upload")
    top_k = st.slider("Top-k", 1, 5, 3, key="diag_topk")
    go = st.button("Predict", type="primary", disabled=upload is None, key="diag_go")

info = model_info(crop, key)
class_names = get_class_names(crop)
demo = info["demo"]

with c_left:
    with st.expander("🧩 Model information", expanded=False):
        st.json({
            "model": info["name"],
            "architecture": info["architecture"],
            "crop": crop,
            "classes": class_names,
            "input_size": info["input_size"],
            "checkpoint": info["checkpoint"] or "none — random init (DEMO)",
            "device": device,
            "demo": demo,
        })
    if not demo and not info["checkpoint"]:
        st.warning("No checkpoint found for this model — running with random weights as DEMO until "
                   "weights are placed under models/checkpoints/ (docs/model_integration.md).")

# ---------------------------------------------------------------- inference
if go and upload is not None:
    img_bytes = upload.getvalue()
    key_hash = hashlib.md5(img_bytes + f"{crop}|{key}|{info['checkpoint'] or 'none'}".encode()).hexdigest()
    try:
        with st.spinner("Running inference…"):
            from PIL import Image

            import io

            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            model = get_model(info["architecture"], len(class_names), info["checkpoint"], device)
            pred = predict_pil(model, class_names, img, device, demo, info["architecture"], crop, top_k)
            st.session_state["diag"] = {"key": key_hash, "pred": pred, "key_model": key,
                                        "arch": info["architecture"], "crop": crop,
                                        "ckpt": info["checkpoint"], "demo": demo,
                                        "classes": class_names, "input_size": info["input_size"],
                                        "model_name": info["name"]}
            st.session_state.pop("diag_cam", None)  # stale heatmaps
    except Exception as e:
        error_box(e)

res = st.session_state.get("diag")
with c_right:
    if res is None:
        st.info("Upload a leaf image and press **Predict**.")
    else:
        pred = res["pred"]
        if res["demo"]:
            demo_banner()
        m1, m2 = st.columns(2)
        m1.metric("Predicted class", pred["label"])
        m2.metric("Confidence", f"{pred['confidence']:.1%}")
        # uncertainty
        probs_sorted = sorted(pred["probs"], reverse=True)
        margin = probs_sorted[0] - (probs_sorted[1] if len(probs_sorted) > 1 else 0.0)
        if pred["confidence"] < 0.6:
            st.warning(f"⚠️ Low confidence ({pred['confidence']:.1%}) — verify with an expert before acting.")
        elif margin < 0.10:
            st.warning(f"⚠️ Ambiguous: top-2 margin only {margin:.1%} — consider Score-CAM cross-check.")
        else:
            st.success("Confidence looks reasonable — still verify visually (XAI below).")
        # top-k
        st.markdown("**Top-k probabilities**")
        import pandas as pd

        df = pd.DataFrame(res["pred"]["top_k"], columns=["class", "probability"])
        t1, t2 = st.columns(2)
        with t1:
            st.dataframe(df.assign(probability=df["probability"].map(lambda p: f"{p:.2%}")),
                         width="stretch", hide_index=True)
        with t2:
            st.bar_chart(df.set_index("class"))

# ---------------------------------------------------------------- XAI
if res is not None:
    st.divider()
    st.subheader("🔥 Visual evidence (XAI)")
    method = st.radio("Method", list(XAI_METHODS), horizontal=True, key="diag_method",
                      format_func=lambda m: {"gradcam_pp": "Grad-CAM++", "scorecam": "Score-CAM",
                                             "saliency": "Saliency"}[m],
                      help="Score-CAM is perturbation-based and slower on CPU. Saliency needs "
                           "gradients through the input and may be unavailable for some models.")
    alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.45, 0.05, key="diag_alpha")
    cam_key = f"{res['key']}|{method}"
    if st.session_state.get("diag_cam_key") != cam_key:
        try:
            with st.spinner(f"Computing {method}…"):
                from src.inference.preprocessing import preprocess_pil
                from src.models.factory import get_adapter

                from PIL import Image
                import io

                model = get_model(res["arch"], len(res["classes"]), res["ckpt"], device)
                adapter = get_adapter(res["arch"])
                img = Image.open(io.BytesIO(upload.getvalue())).convert("RGB")
                tensor = preprocess_pil(img, res["input_size"])
                cam, cls_idx = compute_cam(model, adapter, tensor, method=method)
                st.session_state["diag_cam"] = {"cam": cam, "class_idx": cls_idx,
                                                "layers": target_layer_names(model, adapter)}
                st.session_state["diag_cam_key"] = cam_key
        except Exception as e:
            error_box(e)
            st.session_state["diag_cam"] = None
    cam_res = st.session_state.get("diag_cam")
    if cam_res is not None:
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(upload.getvalue())).convert("RGB")
        ov = blend_overlay(img, cam_res["cam"], alpha=alpha)
        cls_idx = cam_res["class_idx"]
        cls_name = res["classes"][cls_idx] if 0 <= cls_idx < len(res["classes"]) else str(cls_idx)
        probs = pred.get("probs", [])
        conf = probs[cls_idx] if 0 <= cls_idx < len(probs) else None
        conf_txt = f" · {cls_name} {conf:.1%}" if conf is not None else f" · {cls_name}"
        o1, o2, o3 = st.columns(3)
        o1.image(img, caption="Original image", width="stretch")
        o2.image(colorize_cam(cam_res["cam"]), caption=f"{XAI_METHOD_LABELS[method]} heatmap{conf_txt}",
                 width="stretch")
        o3.image(ov, caption=f"Overlay on original (α={alpha:.2f})", width="stretch")
        st.markdown(
            "<span class='small-note'>Highlighted regions indicate image areas contributing strongly to "
            "the selected prediction. This is model attribution, not a diagnosis.</span>",
            unsafe_allow_html=True)
        with st.expander("🧬 Target-layer metadata", expanded=False):
            st.json({"model": res["model_name"], "architecture": res["arch"],
                     "target_layers": cam_res["layers"], "explained_class": cls_name,
                     "explained_class_idx": cls_idx})
        if st.button("💾 Save XAI artifact", key="diag_save"):
            try:
                ip, mp = save_overlay_artifact(ov, {"crop": res["crop"], "model": res["key_model"],
                                                    "architecture": res["arch"],
                                                    "prediction": pred["label"], "confidence": pred["confidence"],
                                                    "method": method, "explained_class": cls_name,
                                                    "target_layers": cam_res["layers"]})
                st.success(f"Saved {ip.name} (+ meta) to outputs/xai/")
            except Exception as e:
                error_box(e)

# ---------------------------------------------------------------- AI layer
if res is not None:
    st.divider()
    st.subheader("💬 AI explanation (optional, assistive only)")
    if st.checkbox("Generate AI explanation", key="diag_ai"):
        try:
            from src.ai.explanation import explain_prediction

            with st.spinner("Asking AI provider (fallback when offline)…"):
                ctx = {"label": pred["label"], "confidence": pred["confidence"], "top_k": pred["top_k"],
                       "crop": res["crop"], "model": res["model_name"],
                       "xai_note": f"{method} heatmap available" if st.session_state.get("diag_cam") else "no XAI",
                       "next_step": "compare lesion close-up with reference images; confirm with expert"}
                st.markdown(explain_prediction(ctx, cfg.ai))
        except Exception as e:
            error_box(e)