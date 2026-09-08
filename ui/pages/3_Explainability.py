import streamlit as st
st.title("Explainability")
st.info("Upload an image on Diagnosis page to generate Grad-CAM++ / Score-CAM. See outputs/xai/")
from pathlib import Path
files = sorted(str(x) for x in Path("outputs/xai").glob("*.jpg"))[-12:]
for f in files: st.image(f, width=260)
