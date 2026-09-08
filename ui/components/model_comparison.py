from __future__ import annotations
def show_comparison(rows: list):
    try:
        import streamlit as st, pandas as pd
        if rows: st.dataframe(pd.DataFrame(rows))
        else: st.info("No runs yet. Train models or see outputs/metrics/runs.csv")
    except Exception: pass
