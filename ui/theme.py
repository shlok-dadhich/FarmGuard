"""Research dashboard styling."""
from __future__ import annotations

CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.2rem;}
.metric-card {background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:12px;}
.demo-banner {background:#fef3c7;border:1px solid #f59e0b;border-radius:8px;padding:8px;font-weight:600;}
.small-note {color:#64748b;font-size:0.85rem;}
.stExpander {border-radius: 8px;}
</style>
"""

_injected = False


def inject():
    global _injected
    if _injected:
        return
    try:
        import streamlit as st

        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
        _injected = True
    except Exception:
        pass
