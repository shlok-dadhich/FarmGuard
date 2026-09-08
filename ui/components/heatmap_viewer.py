from __future__ import annotations
def show_heatmap(path_or_array, caption="Grad-CAM++"):
    try:
        import streamlit as st
        st.image(path_or_array, caption=caption, use_column_width=True)
    except Exception: pass
