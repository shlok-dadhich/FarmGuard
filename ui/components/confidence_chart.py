from __future__ import annotations
def show_topk(topk):
    try:
        import streamlit as st
        st.bar_chart({l: p for l, p in topk})
    except Exception: pass
