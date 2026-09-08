"""Streamlit entry point."""
import streamlit as st
from ui.theme import inject
inject()
st.set_page_config(page_title="AgroVision", layout="wide")
st.title("AgroVision - Plant Disease Research Dashboard")
st.markdown("Use the sidebar pages: Diagnosis, Model Comparison, Explainability, Ensemble, About.")
st.info("DEMO mode uses mock_demo model until external checkpoints are supplied under models/checkpoints/. See docs/model_contract.md")
try:
    from src.core.config import load_config
    cfg = load_config()
    st.json({"crops": list(cfg.datasets["crops"].keys()), "architectures": list(cfg.models_cfg["models"].keys()), "demo_mode": cfg.demo_mode})
except Exception as e:
    st.error(f"Config failed: {e}")
