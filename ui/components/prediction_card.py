from __future__ import annotations
def show_prediction(label, conf, demo=False):
    try:
        import streamlit as st
        if demo: st.warning("DEMO / TEST MODEL - not a research result")
        st.metric("Prediction", label, f"{conf:.1%}")
    except Exception: pass
