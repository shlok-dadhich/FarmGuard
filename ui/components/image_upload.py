from __future__ import annotations
def upload_widget(key="img"):
    try:
        import streamlit as st
        return st.file_uploader("Upload leaf image", type=["jpg","jpeg","png","webp","bmp"], key=key)
    except Exception: return None
