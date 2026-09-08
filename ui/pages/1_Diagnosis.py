"""Diagnosis page."""
import streamlit as st
from PIL import Image
from src.core.config import load_config
from src.models.factory import build_model
from src.inference.predictor import Predictor
from src.ai.explanation import explain_prediction
from src.explainability.xai_pipeline import run_xai
from ui.theme import inject
inject()
st.title("Diagnosis")
cfg = load_config()
arch = st.selectbox("Model", list(cfg.models_cfg["models"].keys()), index=len(cfg.models_cfg["models"])-1)
crop = st.selectbox("Crop", list(cfg.datasets["crops"].keys()))
f = st.file_uploader("Leaf image", type=["jpg","jpeg","png","webp","bmp"])
if f and st.button("Predict"):
    img = Image.open(f).convert("RGB")
    st.image(img, width=320)
    demo = arch == "mock_demo"
    if demo: st.warning("DEMO / TEST MODEL - not a research result")
    model = build_model(arch, 4)
    from src.models.factory import get_adapter
    ad = get_adapter(arch)
    import torch
    pred = Predictor(model, [f"{crop}_class{i}" for i in range(4)], demo=demo, arch=arch, crop=crop).predict(img)
    st.metric(pred["label"], f"{pred['confidence']:.1%}")
    st.bar_chart({l: p for l, p in pred["top_k"]})
    if pred["confidence"] < 0.6: st.warning("Low confidence - verify with expert")
    with torch.no_grad():
        from src.inference.preprocessing import preprocess_pil
        x = preprocess_pil(img, ad.input_size())
        res = run_xai(model, ad, img, x, meta={"crop": crop, "model": arch})
        st.write(res)
    if st.checkbox("AI explanation"):
        st.markdown(explain_prediction({"label": pred["label"], "confidence": pred["confidence"], "top_k": pred["top_k"], "crop": crop, "model": arch}, cfg.ai))
