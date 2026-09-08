from __future__ import annotations
def show_metrics(m: dict):
    try:
        import streamlit as st
        c = st.columns(4)
        c[0].metric("Accuracy", f"{m.get('accuracy',0):.3f}")
        c[1].metric("Macro F1", f"{m.get('macro_f1',0):.3f}")
        c[2].metric("Precision", f"{m.get('macro_precision',0):.3f}")
        c[3].metric("Recall", f"{m.get('macro_recall',0):.3f}")
    except Exception: pass
