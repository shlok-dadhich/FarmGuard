"""Ensemble page: top-model selection, agreement, voting, cost-benefit (measured only)."""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st
from PIL import Image

from ui.common import (
    demo_banner,
    error_box,
    get_class_names,
    get_crops,
    get_device,
    get_model,
    get_model_choices,
    load_runs_table,
    model_info,
    research_runs,
    unavailable_banner,
)
from ui.theme import inject

inject()
st.title("🤝 Ensemble")
st.caption("Majority vote · soft average · validation-weighted voting. Superiority is claimed only from measurements.")

crops = get_crops()
device = get_device()
runs = research_runs(load_runs_table())

crop = st.selectbox("Crop", crops, key="ens_crop")
choices = get_model_choices(crop)
# Top-3 tomato models by test macro F1 when available, otherwise the first choices.
default = [c for c in ("tomato_regnet_y_4gf", "tomato_resnet50", "tomato_convnext_tiny") if c in choices][:3] or choices[:3]
chosen = st.multiselect("Choose 2–3 models", choices, default=default, key="ens_models")
w_mode = st.radio("Weighted-voting weights", ["uniform", "val macro-F1 (runs.csv)"],
                  horizontal=True, key="ens_w",
                  help="Validation-based weights come from logged runs; uniform when runs are absent.")
max_n = st.slider("Dataset-level eval cap (val images)", 8, 128, 32, key="ens_n")
upload = st.file_uploader("Leaf image (single-image consensus)", type=["jpg", "jpeg", "png", "webp", "bmp"],
                          key="ens_upload")

if len(chosen) < 2:
    st.info("Select at least 2 models.")
    st.stop()
if len(chosen) > 3:
    st.warning("More than 3 models selected — only the first 3 will be used.")
    chosen = chosen[:3]

infos = {c: model_info(crop, c) for c in chosen}
demos = {c: infos[c]["demo"] for c in chosen}
if any(demos.values()):
    demo_banner()
    st.caption("Ensemble contains DEMO member(s): " + ", ".join(c for c in chosen if demos[c]))
class_names = get_class_names(crop)

weights = [1.0 / len(chosen)] * len(chosen)
if w_mode.startswith("val") and not runs.empty:
    w = []
    for c in chosen:
        arch = infos[c]["architecture"]
        m = runs[(runs.get("crop", "") == crop) & (runs.get("architecture", "") == arch)] if {"crop", "architecture"} <= set(runs.columns) else runs[runs.get("architecture", "") == arch]
        try:
            w.append(float(m["macro_f1"].max()) if not m.empty else 0.0)
        except Exception:
            w.append(0.0)
    if sum(w) > 0:
        weights = [x / sum(w) for x in w]
        st.caption("Weights from val macro-F1: " + ", ".join(f"{infos[c]['architecture']}={x:.2f}" for c, x in zip(chosen, weights)))
    else:
        st.caption("No usable val macro-F1 in runs.csv — falling back to uniform weights.")
elif w_mode.startswith("val"):
    st.caption("No runs logged — uniform weights.")

# ------------------------------------------------ single image
if upload is not None:
    try:
        from src.inference.ensemble import soft_average, weighted_soft
        from src.inference.predictor import Predictor

        img = Image.open(io.BytesIO(upload.getvalue())).convert("RGB")
        st.image(img, width=280)
        with st.spinner("Running ensemble members…"):
            indiv, prob_lists = [], []
            for c in chosen:
                info = infos[c]
                m = get_model(info["architecture"], len(class_names), info["checkpoint"], device)
                p = Predictor(m, class_names, device, demo=info["demo"], arch=info["architecture"], crop=crop).predict(img, top_k=3)
                indiv.append({"model": info["name"], "architecture": info["architecture"],
                              "label": p["label"], "confidence": p["confidence"],
                              "top_k": ", ".join(f"{l} {c0:.0%}" for l, c0 in p["top_k"])})
                prob_lists.append(p["probs"])
        st.markdown("**Individual predictions**")
        st.dataframe(pd.DataFrame(indiv), width="stretch", hide_index=True)
        import numpy as np

        avg = np.asarray(prob_lists).mean(axis=0)[0]
        wavg = np.asarray(prob_lists)
        wv = (wavg * np.asarray(weights)[:, None]).sum(axis=0)[0]
        st.markdown("**Ensemble prediction**")
        e1, e2 = st.columns(2)
        e1.metric("Soft average", class_names[int(avg.argmax())], f"{float(avg.max()):.1%}")
        e2.metric("Weighted vote", class_names[int(wv.argmax())], f"{float(wv.max()):.1%}")
        labels = [d["label"] for d in indiv]
        if len(set(labels)) == 1:
            st.info(f"All {len(chosen)} members agree on **{labels[0]}** for this image.")
        else:
            st.warning("Members disagree on this image — see per-model confidences above.")
    except Exception as e:
        error_box(e)

# ------------------------------------------------ dataset level
st.divider()
st.subheader("📏 Dataset-level agreement & cost-benefit")
from pathlib import Path

split_p = Path(f"data/splits/{crop}/val.csv")
if not split_p.exists():
    split_p = Path(f"data/splits/{crop}/test.csv")
if not split_p.exists():
    unavailable_banner("no val/test manifest for this crop. Run scripts/prepare_data.py first.")
    st.stop()

if st.button("Run dataset-level ensemble eval", type="primary", key="ens_run"):
    try:
        import time

        import torch

        from src.data.datasets import CropDataset
        from src.evaluation.metrics import compute_metrics
        from src.inference.ensemble import agreement_rate, diversity_warning, majority_vote
        from src.models.factory import get_adapter

        df = pd.read_csv(split_p).head(max_n)
        if df.empty:
            unavailable_banner("manifest is empty.")
            st.stop()
        with st.spinner(f"Evaluating {len(chosen)} models on {len(df)} images…"):
            from src.data.preprocessing import build_transforms

            pred_lists, prob_lists, lat = [], [], {}
            for c in chosen:
                info = infos[c]
                adapter = get_adapter(info["architecture"])
                tf = build_transforms(info["input_size"], train=False)
                ds = CropDataset(df, transform=tf)
                loader = torch.utils.data.DataLoader(ds, batch_size=8)
                m = get_model(info["architecture"], len(class_names), info["checkpoint"], device).eval()
                ps, pr, t0, n = [], [], time.time(), 0
                with torch.no_grad():
                    for x, y in loader:
                        out = m(x.to(device))
                        ps += out.argmax(1).cpu().tolist()
                        pr += torch.softmax(out, 1).cpu().tolist()
                        n += len(y)
                lat[c] = (time.time() - t0) / max(1, n) * 1000
                pred_lists.append(ps)
                prob_lists.append(pr)
            y_true = df["label_id"].tolist()
        rate = agreement_rate(pred_lists)
        st.metric("Prediction agreement", f"{rate:.1%}")
        note = diversity_warning(rate)
        if note:
            st.warning(note + " (measured on this subset)")
        mv = majority_vote(pred_lists)
        import numpy as np

        sv = np.asarray(prob_lists).mean(axis=0).argmax(1).tolist()
        m_mv, m_sv = compute_metrics(y_true, mv), compute_metrics(y_true, sv)
        comp = pd.DataFrame([
            {"method": "majority vote", "accuracy": m_mv["accuracy"], "macro_f1": m_mv["macro_f1"]},
            {"method": "soft average", "accuracy": m_sv["accuracy"], "macro_f1": m_sv["macro_f1"]},
        ])
        # best single (measured here)
        singles = [compute_metrics(y_true, p) for p in pred_lists]
        bi = max(range(len(chosen)), key=lambda i: singles[i]["macro_f1"])
        comp = pd.concat([comp, pd.DataFrame([{"method": f"best single ({infos[chosen[bi]]['name']})",
                                               "accuracy": singles[bi]["accuracy"],
                                               "macro_f1": singles[bi]["macro_f1"]}])], ignore_index=True)
        st.dataframe(comp.style.format({"accuracy": "{:.4f}", "macro_f1": "{:.4f}"}),
                     width="stretch", hide_index=True)
        st.markdown("**Cost (measured mean latency/image)**")
        st.dataframe(pd.DataFrame([{"model": infos[c]["name"], "latency_ms": lat[c]} for c in chosen]),
                     width="stretch", hide_index=True)
        st.caption(f"Ensemble total ≈ {sum(lat.values()):.1f} ms vs best single ≈ {lat[chosen[bi]]:.1f} ms.")
        best_ens = max(m_mv["macro_f1"], m_sv["macro_f1"])
        if best_ens > singles[bi]["macro_f1"]:
            st.success(f"Ensemble beats best single by {best_ens - singles[bi]['macro_f1']:.4f} macro-F1 on this subset.")
        else:
            st.info("Ensemble does NOT beat the best single model on this subset — no free lunch; report as measured.")
    except Exception as e:
        error_box(e)