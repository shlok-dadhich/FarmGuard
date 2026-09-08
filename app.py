"""AgroVision Streamlit entry point with explicit navigation (ui/pages/*)."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="AgroVision", page_icon="🌱", layout="wide")

from ui.theme import inject

inject()


def _home():
    st.title("🌱 AgroVision — Plant Disease Research Dashboard")
    st.caption(
        "Multi-crop classification · controlled comparison · efficiency analysis · "
        "ensembles · Grad-CAM++/Score-CAM · tracking (file / MLflow / TensorBoard)"
    )
    try:
        from ui.common import (
            dataset_info,
            env_info,
            get_architectures,
            get_crops,
            get_device,
            load_app_config,
            load_runs_table,
        )
    except Exception as e:
        st.error(f"WHAT: configuration failed to load\nWHY: {e}\nHOW: run from repo root; check configs/*.yaml")
        return

    cfg = load_app_config()
    device = get_device()
    runs = load_runs_table()
    d_info = dataset_info()

    if cfg.demo_mode:
        st.warning("DEMO mode is ON (configs/project.yaml). Mock results are clearly labelled and are never research results.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Crops", len(get_crops()))
    c2.metric("Architectures", len(get_architectures()))
    c3.metric("Logged runs", len(runs))
    c4.metric("Device", device)

    with st.expander("📁 Dataset status (manifests)", expanded=False):
        if d_info:
            st.json(d_info)
        else:
            st.info("UNAVAILABLE RESULT — no split manifests yet. Run: python scripts/prepare_data.py --crop <crop>")

    with st.expander("🔬 Recent runs", expanded=False):
        if runs.empty:
            st.info("UNAVAILABLE RESULT — no runs logged yet. Run: python scripts/train.py --crop tomato --model mock_demo")
        else:
            cols = [c for c in ("crop", "architecture", "seed", "accuracy", "macro_f1", "run_id") if c in runs.columns]
            st.dataframe(runs[cols].tail(10), width="stretch")

    with st.expander("🧾 Reproducibility", expanded=False):
        st.json(
            {
                "seed": cfg.seed,
                "image_size": cfg.image_size,
                "split": cfg.datasets.get("split_ratios"),
                "augmentation": cfg.datasets.get("augmentation_version"),
                "tracking_backend": cfg.tracking.get("backend"),
                "env": env_info(),
            }
        )

    st.markdown("#### Quickstart")
    st.code(
        "pip install -r requirements.txt\n"
        "streamlit run app.py\n"
        "# evaluation pipeline (models from configs/models.yaml -> instances:)\n"
        "python scripts/run_experiments.py --crop tomato --split test\n"
        "python scripts/generate_comparison.py --crop tomato\n"
        "python scripts/smoke_test.py",
        language="bash",
    )
    st.markdown(
        "<span class='small-note'>External model checkpoints go under "
        "models/checkpoints/ — see docs/model_contract.md. Nothing here fabricates weights or metrics.</span>",
        unsafe_allow_html=True,
    )


pg_home = st.Page(_home, title="Overview", icon="🏠", default=True)
pg_diag = st.Page("ui/pages/1_Diagnosis.py", title="Diagnosis", icon="🔬")
pg_eval = st.Page("ui/pages/2_Model_Evaluation.py", title="Model Evaluation", icon="🧪")
pg_cmp = st.Page("ui/pages/2_Model_Comparison.py", title="Model Comparison", icon="📊")
pg_exp = st.Page("ui/pages/6_Experiments.py", title="Experiments", icon="🗂️")
pg_xai = st.Page("ui/pages/3_Explainability.py", title="Explainability", icon="🔥")
pg_ens = st.Page("ui/pages/4_Ensemble.py", title="Ensemble", icon="🤝")
pg_about = st.Page("ui/pages/5_About.py", title="About", icon="ℹ️")

nav = st.navigation(
    {
        "AgroVision": [pg_home],
        "Research": [pg_diag, pg_eval, pg_cmp, pg_exp, pg_xai, pg_ens],
        "Project": [pg_about],
    }
)
# nav.run() needs a Streamlit runtime: executed when launched via
# `streamlit run app.py` (__main__), skipped on plain `import app` (tests).
if __name__ == "__main__":
    nav.run()
