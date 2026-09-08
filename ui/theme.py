"""Research dashboard styling."""
CUSTOM_CSS = """
<style>
.metric-card {background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:12px;}
.demo-banner {background:#fef3c7;border:1px solid #f59e0b;border-radius:8px;padding:8px;font-weight:600;}
</style>
"""
def inject(): 
    try:
        import streamlit as st
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    except Exception: pass
