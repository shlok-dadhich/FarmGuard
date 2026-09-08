import streamlit as st, pandas as pd
from pathlib import Path
st.title("Model Comparison")
p = Path("outputs/metrics/runs.csv")
if p.exists(): st.dataframe(pd.read_csv(p))
else: st.info("UNAVAILABLE RESULT - no runs yet. Run scripts/train.py")
