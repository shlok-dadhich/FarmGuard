"""About page: methodology, models, datasets, reproducibility, limitations."""
from __future__ import annotations

import streamlit as st

from ui.common import dataset_info, env_info, get_architectures, get_crops, load_app_config
from ui.theme import inject

inject()
st.title("ℹ️ About AgroVision")

cfg = load_app_config()

st.markdown("## Methodology")
st.markdown(
    "1. **Deterministic stratified splits** (75/10/15) saved to `data/splits/<crop>/*.csv` and reused across "
    "architectures and seeds.\n"
    "2. **Shared preprocessing + augmentation** for all architectures (deterministic val/test).\n"
    "3. **Screening then full training**: short 1-seed screening → 3-seed full runs reported as mean ± std "
    "(a single seed is never the final result).\n"
    "4. **Primary metric macro F1** — accuracy alone is never sufficient.\n"
    "5. **Efficiency-normalized analysis**: accuracy/F1 per M params and per FLOP, plus measured latency.\n"
    "6. **Ensembles** from top 2–3 models with measured agreement, diversity check, and cost-benefit verdict.\n"
    "7. **Explainability**: Grad-CAM++ with Score-CAM cross-check and pointing-game faithfulness when masks exist.\n"
    "8. **Tracking**: file backend always (`outputs/metrics/`), optional MLflow + TensorBoard."
)

st.markdown("## Candidate architectures")
rows = []
for arch in get_architectures():
    meta = (cfg.models_cfg.get("models", {}).get(arch, {}) or {})
    rows.append({"architecture": arch, "family": meta.get("family", "?"),
                 "input": meta.get("input_size", "?"), "notes": meta.get("description", "")})
st.dataframe(rows, width="stretch", hide_index=True)

st.markdown("## Datasets")
crops_cfg = cfg.datasets.get("crops", {})
drows = [{"crop": c, "root": (v or {}).get("root", "?"), "notes": (v or {}).get("description", "")}
         for c, v in crops_cfg.items()]
st.dataframe(drows, width="stretch", hide_index=True)
manifests = dataset_info()
if manifests:
    with st.expander("Manifest row counts", expanded=False):
        st.json(manifests)
else:
    st.info("UNAVAILABLE RESULT — no manifests yet. Run scripts/prepare_data.py --crop <crop>.")

st.markdown("## Reproducibility")
st.json({
    "seed": cfg.seed,
    "image_size": cfg.image_size,
    "split_ratios": cfg.datasets.get("split_ratios"),
    "augmentation_version": cfg.datasets.get("augmentation_version"),
    "optimizer": cfg.training.get("optimizer"),
    "scheduler": cfg.training.get("scheduler"),
    "tracking": cfg.tracking,
    "env": env_info(),
})

st.markdown("## Limitations & integrity rules")
st.markdown(
    "- `models/` is externally owned: no invented architectures, no fake checkpoints, no silent substitutions.\n"
    "- Anything labelled **DEMO / TEST MODEL** is plumbing output, never a research result.\n"
    "- Missing data is reported as **UNAVAILABLE RESULT**, never filled in.\n"
    "- Ensemble superiority and XAI faithfulness are claimed only from measurements.\n"
    "- AI text is assistive: it never overrides the classifier and never states agronomic certainty."
)
